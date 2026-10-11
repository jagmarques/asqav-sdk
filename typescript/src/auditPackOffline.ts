// Offline whole-pack check mirroring Python pack_verify, no network
// No RFC3161 or OTS evaluation: anchored receipts stay not verifiable

import { verifyReceiptOffline } from "./index.js";
import type { AxisResult } from "./verifier/core.js";
import { asqavJcs, parseJsonPreservingFloats } from "./verifier/canonical.js";
import { sha256Hex, verifySignature } from "./verifier/crypto.js";
import { b64decode, checkAnchors, normaliseEnvelope, thumbprintForKey } from "./verifier/vrShim.js";

export type PackStatus = "verified" | "not_verifiable_offline" | "invalid";
/** Pack status when nothing failed but the keys came from the pack itself */
export type PackOverall = PackStatus | "self_consistent";
export const SELF_CONSISTENT_LABEL = "self-consistent, not anchored to a trusted key";

export interface PackEntryResult {
  signature_id: unknown;
  agent_id: unknown;
  status: PackStatus;
  reasons: string[];
  signature: string | null;
  chain: string | null;
  notes: string[];
}

export interface BundleResult {
  status: PackStatus;
  reason: string;
  key_thumbprint: string | null;
}

export interface PackReport {
  status: PackOverall;
  exit_code: number;
  bundle: BundleResult;
  receipts: PackEntryResult[];
  summary: { total: number; verified: number; not_verifiable_offline: number; invalid: number };
  trust: { anchored: boolean; keys: string; key_count: number; tsa_carried_not_trusted: number };
}

type Obj = Record<string, unknown>;
type EntryBody = Omit<PackEntryResult, "signature_id" | "agent_id">;

const RANK: Record<PackStatus, number> = { verified: 0, not_verifiable_offline: 1, invalid: 2 };
const EXIT_CODES: Record<PackOverall, number> = {
  verified: 0,
  not_verifiable_offline: 2,
  self_consistent: 2,
  invalid: 1,
};
const BUNDLE_ALGS = new Set(["ML-DSA-65", "ED25519", "ES256"]);
const REVOKED_STATUSES = new Set(["revoked", "suspended", "compromised"]);
const FIRST_SEED = "0".repeat(64);
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
export const UNVOUCHED_REASON = "bundle signature not verified, so nothing vouches for this entry";

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

function outcome(status: PackStatus, reason: string, thumb: string | null = null): BundleResult {
  return { status, reason, key_thumbprint: thumb };
}

type Material = { msg: Uint8Array; key: Uint8Array; sig: Uint8Array; alg: string };

function bundleMaterial(pack: Obj): Material | BundleResult {
  const { bundle_digest: digest, bundle_signature: sig, bundle_public_key: keyB64 } = pack;
  if (![digest, sig, keyB64].every((v) => typeof v === "string" && v.length > 0)) {
    return outcome("invalid", "pack carries no bundle signature");
  }
  const alg = pack.bundle_signature_algorithm;
  if (typeof alg !== "string" || !BUNDLE_ALGS.has(alg.toUpperCase())) {
    return outcome("invalid", `unsupported bundle signature algorithm '${String(alg)}'`);
  }
  const msg = asqavJcs(signedBundlePayload(pack));
  if (`sha256:${sha256Hex(msg)}` !== digest) {
    return outcome("invalid", "bundle_digest does not match the pack contents");
  }
  return { msg, key: b64decode(keyB64 as string), sig: b64decode(sig as string), alg };
}

function bundleKeyProblem(pack: Obj, entry: Obj | undefined, alg: string, anchored: boolean): BundleResult | null {
  if (entry === undefined) return outcome("invalid", "bundle key is not in the key set");
  if (typeof entry.alg === "string" && entry.alg.toUpperCase() !== alg.toUpperCase()) {
    return outcome("invalid", `bundle algorithm ${alg} differs from its key's alg ${entry.alg}`);
  }
  if (pack.organization_id && entry.org_id && entry.org_id !== pack.organization_id) {
    return outcome("invalid", "bundle signing key belongs to another organization");
  }
  const status = String(entry.status ?? "").toLowerCase();
  if (REVOKED_STATUSES.has(status) || entry.revoked_at) {
    return outcome(anchored ? "invalid" : "not_verifiable_offline", `bundle key is ${status || "revoked"}`);
  }
  return null;
}

/** Recompute the bundle digest, check its signature and key */
export function checkBundle(pack: Obj, jwks: unknown, anchored = false): BundleResult {
  const material = bundleMaterial(pack);
  if ("status" in material) return material;
  const { msg, key, sig, alg } = material;
  const entry = jwksKeys(jwks).find((k) => sameBytes(keyBytes(k), key));
  const problem = bundleKeyProblem(pack, entry, alg, anchored);
  if (problem !== null || entry === undefined) return problem as BundleResult;
  const res = verifySignature(alg, key, msg, sig);
  if (res.result !== "PASS") {
    return outcome(res.result === "FAIL" ? "invalid" : "not_verifiable_offline", `bundle signature: ${res.note}`);
  }
  const thumb = typeof entry.key_thumbprint === "string" ? entry.key_thumbprint : safeThumbprint(alg, key);
  return outcome("verified", `bundle signature valid under key ${String(entry.kid)}`, thumb);
}

function safeThumbprint(alg: string, key: Uint8Array): string | null {
  try {
    return thumbprintForKey(alg, key);
  } catch {
    return null;
  }
}

function seqOf(p: Obj): number | null {
  return typeof p.seq === "number" && Number.isInteger(p.seq) ? p.seq : null;
}

function agentOf(e: Obj): string {
  return String((e.payload as Obj).agent_id ?? e.agent_id ?? "");
}

function linkHash(e: Obj): string | null {
  if (e.signed_bytes_withheld) return typeof e.message_sha256 === "string" ? e.message_sha256 : null;
  return sha256Hex(asqavJcs(e.payload));
}

interface ChainIndex {
  envelopes: Map<number, Obj>;
  bySeq: Map<string, number[]>;
  byHash: Map<string, number>;
}

function chainIndex(receipts: unknown[]): ChainIndex {
  const envelopes = new Map<number, Obj>();
  receipts.forEach((e, i) => {
    if (isObj(e) && isObj(e.payload)) envelopes.set(i, e);
  });
  const bySeq = new Map<string, number[]>();
  const byHash = new Map<string, number>();
  for (const [i, e] of envelopes) {
    const seq = e.signed_bytes_withheld ? null : seqOf(e.payload as Obj);
    if (seq !== null) {
      const k = JSON.stringify([agentOf(e), seq]);
      bySeq.set(k, [...(bySeq.get(k) ?? []), i]);
    }
    const h = linkHash(e);
    if (h !== null) byHash.set(JSON.stringify([agentOf(e), h]), i);
  }
  return { envelopes, bySeq, byHash };
}

function link(e: Obj, idx: ChainIndex): [Obj | null, string | null] {
  const payload = e.payload as Obj;
  const seq = seqOf(payload);
  const agent = agentOf(e);
  const prev = payload.previousReceiptHash;
  if (prev === FIRST_SEED) return [null, null];
  const before = seq === null ? undefined : idx.bySeq.get(JSON.stringify([agent, seq - 1]));
  if (before !== undefined && before.length === 1) return [(idx.envelopes.get(before[0]) as Obj).payload as Obj, null];
  const j = idx.byHash.get(JSON.stringify([agent, prev]));
  if (j !== undefined) {
    const pred = idx.envelopes.get(j) as Obj;
    return [pred.signed_bytes_withheld ? null : (pred.payload as Obj), null];
  }
  const lower = [...idx.bySeq.keys()].some((k) => {
    const [a, s] = JSON.parse(k) as [string, number];
    return a === agent && seq !== null && s < seq;
  });
  return [null, lower ? `chain: gap, the predecessor of seq ${seq} is missing from the pack` : null];
}

function duplicates(pairs: Array<[number, string | null]>): Map<number, string> {
  const groups = new Map<string, number[]>();
  for (const [i, k] of pairs) if (k !== null) groups.set(k, [...(groups.get(k) ?? []), i]);
  const out = new Map<number, string>();
  for (const [k, idxs] of groups) if (idxs.length > 1) for (const i of idxs) out.set(i, k);
  return out;
}

function chainIssues(receipts: unknown[], idx: ChainIndex): Map<number, string[]> {
  const issues = new Map<number, string[]>();
  const add = (i: number, msg: string): void => void issues.set(i, [...(issues.get(i) ?? []), msg]);
  for (const [k, idxs] of idx.bySeq) {
    const [agent, seq] = JSON.parse(k) as [string, number];
    if (idxs.length > 1) for (const i of idxs) add(i, `chain: fork, seq ${seq} appears twice for agent ${agent}`);
  }
  const prevs = [...idx.envelopes].map(([i, e]): [number, string] => [i, JSON.stringify([agentOf(e), (e.payload as Obj).previousReceiptHash ?? null])]);
  for (const [i, k] of duplicates(prevs)) add(i, `chain: fork, two receipts of ${(JSON.parse(k) as string[])[0]} share a predecessor`);
  const ids = receipts.map((e, i): [number, string | null] => [i, isObj(e) && typeof e.signature_id === "string" ? e.signature_id : null]);
  for (const [i, sid] of duplicates(ids)) add(i, `duplicate entry: signature_id ${sid}`);
  return issues;
}

/** Predecessor payloads and invalid chain reasons, by receipt index */
export function chainContext(receipts: unknown[]): [Map<number, Obj | null>, Map<number, string[]>] {
  const idx = chainIndex(receipts);
  const issues = chainIssues(receipts, idx);
  const preds = new Map<number, Obj | null>();
  for (const [i, e] of idx.envelopes) {
    const [pred, issue] = link(e, idx);
    preds.set(i, pred);
    if (issue) issues.set(i, [...(issues.get(i) ?? []), issue]);
  }
  return [preds, issues];
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

function bare(status: PackStatus, reason: string): EntryBody {
  return { status, reasons: [reason], signature: null, chain: null, notes: [] };
}

function envelopeResult(entry: Obj, jwks: Obj, predecessor: Obj | null): EntryBody {
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

function flatResult(entry: Obj, jwks: Obj): EntryBody {
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

function applyContext(r: PackEntryResult, issues: string[], bundleOk: boolean): PackEntryResult {
  if (issues.length > 0) return { ...r, status: "invalid", reasons: [...issues, ...r.reasons] };
  if (r.status === "verified" && !bundleOk) return { ...r, status: "not_verifiable_offline", reasons: [UNVOUCHED_REASON] };
  return r;
}

function tsaCount(pack: Obj): number {
  const raw = pack.tsa_trust;
  return Array.isArray(raw) ? raw.length : raw ? 1 : 0;
}

/** Verify raw pack JSON offline; keys are trusted only from `jwksText`
 *  Without it the best result is self_consistent, exit 2 */
export function verifyAuditPackOffline(packText: string, jwksText?: string): PackReport {
  const pack = parseJsonPreservingFloats(packText);
  if (!isObj(pack)) throw new SyntaxError("pack is not a JSON object");
  const supplied = jwksText === undefined ? undefined : parseJsonPreservingFloats(jwksText);
  if (supplied !== undefined && !isObj(supplied)) throw new SyntaxError("key set is not a JSON object");
  const anchored = supplied !== undefined;
  const keySet: Obj = isObj(supplied) ? supplied : isObj(pack.jwks) ? pack.jwks : { keys: [] };
  const receipts = Array.isArray(pack.receipts) ? pack.receipts : [];
  const [preds, issues] = chainContext(receipts);
  const bundle = checkBundle(pack, keySet, anchored);
  const results = receipts.map((r, i) =>
    applyContext(verifyPackEntry(r, keySet, preds.get(i) ?? null), issues.get(i) ?? [], bundle.status === "verified"),
  );
  const count = (s: PackStatus): number => results.filter((r) => r.status === s).length;
  const worst = [bundle.status, ...results.map((r) => r.status)].reduce<PackStatus>(
    (w, s) => (RANK[s] > RANK[w] ? s : w),
    "verified",
  );
  const status: PackOverall = worst !== "invalid" && !anchored ? "self_consistent" : worst;
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
    trust: {
      anchored,
      keys: anchored ? "supplied key set" : "pack key snapshot (self-asserted)",
      key_count: jwksKeys(keySet).length,
      tsa_carried_not_trusted: tsaCount(pack),
    },
  };
}
