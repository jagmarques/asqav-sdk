// Offline whole-pack check mirroring Python pack_verify, no network
// No RFC3161 or OTS evaluation: anchored receipts stay not verifiable

import { verifyReceiptOffline } from "./index.js";
import type { AxisResult } from "./verifier/core.js";
import { asqavJcs, parseJsonPreservingFloats } from "./verifier/canonical.js";
import { sha256Hex, verifySignature } from "./verifier/crypto.js";
import { b64decode, checkAnchors, normaliseEnvelope } from "./verifier/vrShim.js";

export type PackStatus = "verified" | "not_verifiable_offline" | "invalid";

export interface PackEntryResult {
  signature_id: unknown;
  agent_id: unknown;
  status: PackStatus;
  reasons: string[];
  signature: string | null;
  chain: string | null;
  notes: string[];
}

export interface PackReport {
  status: PackStatus;
  exit_code: number;
  bundle: { status: PackStatus; reason: string };
  receipts: PackEntryResult[];
  summary: { total: number; verified: number; not_verifiable_offline: number; invalid: number };
  trust: { keys: string; key_count: number };
}

type Obj = Record<string, unknown>;

const RANK: Record<PackStatus, number> = { verified: 0, not_verifiable_offline: 1, invalid: 2 };
const EXIT_CODES: Record<PackStatus, number> = { verified: 0, not_verifiable_offline: 2, invalid: 1 };
const PRESENCE_MEMBERS = [
  "content_disclosure",
  "regime_mapping_disclaimer",
  "regime_provenance",
  "jwks",
  "exit_manifest",
  "tsa_trust",
];
const FLAT_REASON =
  "non-compliance-mode receipt: it has no signed receipt envelope (issuer, chain, expiry), " +
  "so it cannot be verified offline";
const WITHHELD_REASON =
  "signed bytes withheld at export; re-export with --include-signed-bytes to verify it";

function isObj(v: unknown): v is Obj {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

/** The pack projection the bundle signature covers (export contract) */
export function signedBundlePayload(pack: Obj): Obj {
  const receipts = Array.isArray(pack.receipts) ? pack.receipts : [];
  const out: Obj = {
    organization_id: pack.organization_id ?? null,
    start: pack.start ?? null,
    end: pack.end ?? null,
    agent_id: pack.agent_id ?? null,
    only_compliance: pack.only_compliance ?? null,
    algorithm_registry_version: pack.algorithm_registry_version ?? null,
    receipt_count: "receipt_count" in pack ? pack.receipt_count : receipts.length,
    receipts,
    revocation_manifest: pack.revocation_manifest || [],
    regime_mapping: pack.regime_mapping || {},
  };
  for (const key of PRESENCE_MEMBERS) {
    if (pack[key] !== undefined && pack[key] !== null) out[key] = pack[key];
  }
  const attested = pack.attested_framework_coverage;
  if (isObj(attested) && Object.keys(attested).length > 0) out.attested_framework_coverage = attested;
  return out;
}

function jwksKeys(jwks: unknown): Obj[] {
  const keys = isObj(jwks) ? jwks.keys : undefined;
  return Array.isArray(keys) ? keys.filter(isObj) : [];
}

function keyBytes(entry: Obj): Uint8Array | null {
  const value = entry.public_key ?? entry.pub;
  return typeof value === "string" ? b64decode(value) : null;
}

function sameBytes(a: Uint8Array | null, b: Uint8Array): boolean {
  return a !== null && Buffer.from(a).equals(Buffer.from(b));
}

/** Recompute the bundle digest, check its signature and key */
export function checkBundle(pack: Obj, jwks: unknown): { status: PackStatus; reason: string } {
  const { bundle_digest: digest, bundle_signature: sig, bundle_public_key: keyB64 } = pack;
  if (typeof digest !== "string" || typeof sig !== "string" || typeof keyB64 !== "string") {
    return { status: "not_verifiable_offline", reason: "pack carries no bundle signature" };
  }
  const msg = asqavJcs(signedBundlePayload(pack));
  if (`sha256:${sha256Hex(msg)}` !== digest) {
    return { status: "invalid", reason: "bundle_digest does not match the pack contents" };
  }
  const key = b64decode(keyB64);
  const alg = typeof pack.bundle_signature_algorithm === "string" ? pack.bundle_signature_algorithm : "ML-DSA-65";
  const outcome = verifySignature(alg, key, msg, b64decode(sig));
  if (outcome.result !== "PASS") {
    const status = outcome.result === "FAIL" ? "invalid" : "not_verifiable_offline";
    return { status, reason: `bundle signature: ${outcome.note}` };
  }
  const entry = jwksKeys(jwks).find((k) => sameBytes(keyBytes(k), key));
  if (entry === undefined) {
    return { status: "not_verifiable_offline", reason: "bundle signature valid, but its key is not in the key set" };
  }
  if (pack.organization_id && entry.org_id && entry.org_id !== pack.organization_id) {
    return { status: "invalid", reason: "bundle signing key belongs to another organization" };
  }
  return { status: "verified", reason: `bundle signature valid under key ${String(entry.kid)}` };
}

function chainOrder(p: Obj): [number, string] {
  const seq = typeof p.seq === "number" && Number.isInteger(p.seq) ? p.seq : Number.POSITIVE_INFINITY;
  return [seq, String(p.issued_at ?? "")];
}

function consecutive(prev: Obj, cur: Obj): boolean {
  if (typeof prev.seq === "number" && typeof cur.seq === "number") return cur.seq === prev.seq + 1;
  return true;
}

/** Receipt index to its predecessor payload, per agent chain */
export function chainPredecessors(receipts: unknown[]): Map<number, Obj | null> {
  const chains = new Map<string, number[]>();
  receipts.forEach((entry, i) => {
    if (!isObj(entry) || !isObj(entry.payload) || entry.signed_bytes_withheld) return;
    const agent = String(entry.payload.agent_id ?? entry.agent_id ?? "");
    chains.set(agent, [...(chains.get(agent) ?? []), i]);
  });
  const preds = new Map<number, Obj | null>();
  const payloadAt = (i: number): Obj => (receipts[i] as Obj).payload as Obj;
  for (const idxs of chains.values()) {
    idxs.sort((a, b) => {
      const [sa, ta] = chainOrder(payloadAt(a));
      const [sb, tb] = chainOrder(payloadAt(b));
      return sa !== sb ? sa - sb : ta < tb ? -1 : ta > tb ? 1 : 0;
    });
    idxs.forEach((cur, n) => {
      const prev = n === 0 ? null : payloadAt(idxs[n - 1]);
      preds.set(cur, prev !== null && consecutive(prev, payloadAt(cur)) ? prev : null);
    });
  }
  return preds;
}

function expiryNote(payload: Obj, axes: AxisResult[]): string | null {
  const axis = axes.find((a) => a.axis === "expiry");
  if (!axis || axis.result !== "FAIL" || !axis.note.includes("lapsed")) return null;
  const issued = Date.parse(String(payload.issued_at));
  const expires = Date.parse(String(payload.expires_at));
  if (!Number.isNaN(issued) && !Number.isNaN(expires) && issued <= expires) {
    return `expires_at ${String(payload.expires_at)} has passed; signed before it, so not a failure`;
  }
  return `expires_at ${String(payload.expires_at)} is not after issued_at`;
}

function blockingReasons(status: PackStatus, axes: AxisResult[]): string[] {
  const rows = axes.filter((a) =>
    status === "invalid"
      ? a.failureClass === "invalid" && a.axis !== "expiry"
      : a.result !== "PASS" && a.axis !== "expiry" && !(a.axis === "chain" && a.result === "SKIPPED"),
  );
  return rows.map((a) => `${a.axis}: ${a.note.split(/\s+/).join(" ")}`);
}

function bare(status: PackStatus, reason: string): Omit<PackEntryResult, "signature_id" | "agent_id"> {
  return { status, reasons: [reason], signature: null, chain: null, notes: [] };
}

function envelopeResult(entry: Obj, jwks: Obj, predecessor: Obj | null): Omit<PackEntryResult, "signature_id" | "agent_id"> {
  const res = verifyReceiptOffline(entry, jwks, predecessor);
  const [anchorState, anchorNote] = checkAnchors(normaliseEnvelope(entry));
  const axes: AxisResult[] = [
    ...res.axes,
    { axis: "anchors", result: anchorState, note: anchorNote, failureClass: anchorState === "FAIL" ? "invalid" : anchorState === "SKIPPED" ? "unverifiable" : null },
  ];
  let status: PackStatus = "verified";
  if (res.failureClass === "invalid" || anchorState === "FAIL") status = "invalid";
  else if (res.verdict === "unverified" || anchorState !== "PASS") status = "not_verifiable_offline";
  const named = (n: string): string | null => axes.find((a) => a.axis === n)?.note ?? null;
  const expiry = expiryNote(isObj(entry.payload) ? entry.payload : {}, axes);
  return { status, reasons: blockingReasons(status, axes), signature: named("signature"), chain: named("chain"), notes: expiry ? [expiry] : [] };
}

function flatResult(entry: Obj, jwks: Obj): Omit<PackEntryResult, "signature_id" | "agent_id"> {
  const out = bare("not_verifiable_offline", FLAT_REASON);
  const agent = entry.agent_id;
  const keys = jwksKeys(jwks).filter((k) => agent && k.agent_id === agent);
  const { message_b64: msg, signature_b64: sig } = entry;
  if (keys.length === 0 || typeof msg !== "string" || !msg || typeof sig !== "string" || !sig) return out;
  const alg = String(entry.algorithm ?? "");
  const results = keys.map((k) => verifySignature(alg, keyBytes(k) ?? new Uint8Array(), b64decode(msg), b64decode(sig)).result);
  if (results.includes("PASS")) out.signature = "signature valid under the agent's published key";
  else if (results.every((r) => r === "FAIL")) {
    return { ...out, status: "invalid", reasons: [`signature does not verify under any key of agent ${String(agent)}`] };
  }
  return out;
}

/** Status, reasons and notes for one pack entry. */
export function verifyPackEntry(entry: unknown, jwks: Obj, predecessor: Obj | null = null): PackEntryResult {
  if (!isObj(entry)) return { signature_id: null, agent_id: null, ...bare("invalid", "entry is not a JSON object") };
  const head = { signature_id: entry.signature_id ?? null, agent_id: entry.agent_id ?? null };
  if (entry.signed_bytes_withheld) return { ...head, ...bare("not_verifiable_offline", WITHHELD_REASON) };
  if (isObj(entry.payload)) return { ...head, ...envelopeResult(entry, jwks, predecessor) };
  if (entry.compliance_mode === false) return { ...head, ...flatResult(entry, jwks) };
  return { ...head, ...envelopeResult(entry, jwks, null) };
}

/** Verify raw pack JSON offline; `jwksText` replaces the key snapshot
 *  exit_code: 0 all verified, 2 some not verifiable offline, 1 invalid */
export function verifyAuditPackOffline(packText: string, jwksText?: string): PackReport {
  const pack = parseJsonPreservingFloats(packText);
  if (!isObj(pack)) throw new SyntaxError("pack is not a JSON object");
  const supplied = jwksText === undefined ? undefined : parseJsonPreservingFloats(jwksText);
  const keySet: Obj = isObj(supplied) ? supplied : isObj(pack.jwks) ? pack.jwks : { keys: [] };
  const receipts = Array.isArray(pack.receipts) ? pack.receipts : [];
  const preds = chainPredecessors(receipts);
  const results = receipts.map((r, i) => verifyPackEntry(r, keySet, preds.get(i) ?? null));
  const bundle = checkBundle(pack, keySet);
  const count = (s: PackStatus): number => results.filter((r) => r.status === s).length;
  const status = [bundle.status, ...results.map((r) => r.status)].reduce<PackStatus>(
    (worst, s) => (RANK[s] > RANK[worst] ? s : worst),
    "verified",
  );
  return {
    status,
    exit_code: EXIT_CODES[status],
    bundle,
    receipts: results,
    summary: {
      total: results.length,
      verified: count("verified"),
      not_verifiable_offline: count("not_verifiable_offline"),
      invalid: count("invalid"),
    },
    trust: { keys: supplied === undefined ? "pack key snapshot" : "supplied key set", key_count: jwksKeys(keySet).length },
  };
}
