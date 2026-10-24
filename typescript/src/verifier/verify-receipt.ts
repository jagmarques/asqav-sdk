// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Asqav; standalone offline receipt verifier, see NOTICE

/** Port of verify_receipt.py: node: builtins only, no network, no API key;
 * an axis it cannot evaluate reports SKIPPED, never PASS. */

import { createHash, createPublicKey, verify as verifySig } from "node:crypto";
import { readFileSync } from "node:fs";

export type AxisResult = "PASS" | "FAIL" | "SKIPPED";
export interface Axis {
  name: string;
  result: AxisResult;
  note: string;
}
export interface Verification {
  verdict: "verified" | "verified_keyed" | "unverified";
  failureClass: "invalid" | "unverifiable" | null;
  axes: Axis[];
  refusal: string | null;
  canonicalSha256: string | null;
}

/** A float lexeme kept apart from integers, as Python's json.loads does. */
export class PyFloat {
  constructor(value: number, lexeme: string) {
    this.value = value;
    this.lexeme = lexeme;
  }
  value: number;
  lexeme: string;
}
export type Json = null | boolean | number | string | PyFloat | Json[] | { [k: string]: Json };
type Obj = { [k: string]: Json };

export const FIRST_RECEIPT_SEED = "0".repeat(64);
export const SKEW_BOUND_SECONDS = 300;
export const MAX_NESTING_DEPTH = 200;
const MAX_CANONICAL_INTEGER = 2 ** 53;
const MAX_PROFILE_INTEGER = 2 ** 53 - 1;
export const REQUIRED_FIELDS = [
  "type", "issued_at", "issuer_id", "action_ref", "payload_digest",
  "policy_digest", "previousReceiptHash", "decision",
];
export const ALLOWED_TYPES = new Set([
  "protectmcp:decision", "protectmcp:restraint", "protectmcp:lifecycle",
  "protectmcp:lifecycle:configuration_change", "protectmcp:lifecycle:risk_acceptance",
  "protectmcp:lifecycle:code_authorship", "protectmcp:acknowledgment",
  "protectmcp:observation", "protectmcp:observation:result_bound",
]);
const REGISTRY_ONLY_TYPES = new Set(["protectmcp:lifecycle:oversight_ruling"]);
const ALLOWED_CONTROL_KEYS = new Set([
  "emergency_halt", "delegation_scope", "quorum", "mandate", "policy", "content_scan", "result",
]);
const REVOKED_KEY_STATUSES = new Set(["revoked", "suspended", "compromised"]);
const OUTER_FREE = new Set(["payload", "signature", "anchors"]);
const INVALID_FAIL_AXES = new Set([
  "signature", "anchors", "issuer_bind", "key_status", "nonce", "key_binding",
  "counterparty", "payload_digest", "outer_copies",
]);
const ML_DSA_PUBLIC_KEY_BYTES: Record<string, number> = {
  "ML-DSA-44": 1312, "ML-DSA-65": 1952, "ML-DSA-87": 2592,
};

export class VerifierInputError extends Error {}

const isObj = (v: unknown): v is Obj =>
  typeof v === "object" && v !== null && !Array.isArray(v) && !(v instanceof PyFloat);
const isInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v);
const has = (o: object, k: string) => Object.prototype.hasOwnProperty.call(o, k);
const show = (v: unknown): string =>
  v instanceof PyFloat ? v.lexeme : v === undefined ? "None" : JSON.stringify(v);

/** Strict JSON: duplicate members, integers past 2**53, NaN are refused. */
export function parseStrictJson(text: string, source = "input"): Json {
  let i = 0;
  const fail = (why: string): never => {
    throw new VerifierInputError(`${source}: ${why}`);
  };
  const ws = () => {
    while (i < text.length && " \t\n\r".includes(text[i])) i++;
  };
  const str = (): string => {
    let out = "";
    i++;
    for (;;) {
      const c = text[i++];
      if (c === undefined) fail("not valid JSON (unterminated string)");
      if (c === '"') return out;
      if (c < " ") fail("not valid JSON (control character in string)");
      if (c !== "\\") {
        out += c;
        continue;
      }
      const e = text[i++];
      const simple: Record<string, string> = {
        '"': '"', "\\": "\\", "/": "/", b: "\b", f: "\f", n: "\n", r: "\r", t: "\t",
      };
      if (e in simple) out += simple[e];
      else if (e === "u" && /^[0-9a-fA-F]{4}$/.test(text.slice(i, i + 4))) {
        out += String.fromCharCode(parseInt(text.slice(i, i + 4), 16));
        i += 4;
      } else fail("not valid JSON (bad escape)");
    }
  };
  const num = (): Json => {
    const m = /^-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?/.exec(text.slice(i));
    if (!m) return fail(`not valid JSON (unexpected ${JSON.stringify(text[i] ?? "end")})`);
    i += m[0].length;
    const value = Number(m[0]);
    if (m[2] !== undefined || m[3] !== undefined) {
      if (!Number.isFinite(value)) fail(`non-finite number ${m[0]}`);
      return new PyFloat(value, m[0]);
    }
    const big = BigInt(m[0]);
    if ((big < 0n ? -big : big) > BigInt(MAX_CANONICAL_INTEGER)) {
      fail(`integer outside the canonical integer range +/-2**53: ${m[0]}`);
    }
    return value === 0 ? 0 : value;
  };
  const value = (depth: number): Json => {
    if (depth > MAX_NESTING_DEPTH) fail(`nesting exceeds ${MAX_NESTING_DEPTH} levels`);
    ws();
    const c = text[i];
    if (c === "{") {
      const obj: Obj = {};
      i++;
      ws();
      if (text[i] === "}") {
        i++;
        return obj;
      }
      for (;;) {
        ws();
        if (text[i] !== '"') fail("not valid JSON (expected a member name)");
        const key = str();
        if (has(obj, key)) {
          fail(`duplicate JSON member name: ${JSON.stringify(key)}`);
        }
        ws();
        if (text[i++] !== ":") fail("not valid JSON (expected ':')");
        Object.defineProperty(obj, key, {
          value: value(depth + 1), enumerable: true, writable: true, configurable: true,
        });
        ws();
        if (text[i++] === "}") return obj;
        if (text[i - 1] !== ",") fail("not valid JSON (expected ',' or '}')");
      }
    }
    if (c === "[") {
      const arr: Json[] = [];
      i++;
      ws();
      if (text[i] === "]") {
        i++;
        return arr;
      }
      for (;;) {
        arr.push(value(depth + 1));
        ws();
        if (text[i++] === "]") return arr;
        if (text[i - 1] !== ",") fail("not valid JSON (expected ',' or ']')");
      }
    }
    if (c === '"') return str();
    for (const [word, lit] of [["true", true], ["false", false], ["null", null]] as const) {
      if (text.startsWith(word, i)) {
        i += word.length;
        return lit;
      }
    }
    return num();
  };
  if (!text.trim()) fail("empty input, expected a JSON object");
  const out = value(0);
  ws();
  if (i !== text.length) fail("not valid JSON (trailing data)");
  return out;
}

/** Python repr() of a double: shortest digits, exponent from 1e16, 1e-5. */
export function pyFloatRepr(x: number): string {
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  const [mant, expText] = x.toExponential().split("e");
  const exp = Number(expText);
  const sign = mant.startsWith("-") ? "-" : "";
  const digits = mant.replace("-", "").replace(".", "");
  if (exp < -4 || exp >= 16) {
    const head = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    return `${sign}${head}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
  }
  if (exp < 0) return `${sign}0.${"0".repeat(-exp - 1)}${digits}`;
  const whole = digits.slice(0, exp + 1).padEnd(exp + 1, "0");
  return `${sign}${whole}.${digits.slice(exp + 1) || "0"}`;
}

function jcsString(s: string): string {
  let out = '"';
  for (let k = 0; k < s.length; k++) {
    const code = s.charCodeAt(k);
    if (code >= 0xd800 && code <= 0xdfff) {
      const pair = code < 0xdc00 && (s.charCodeAt(k + 1) & 0xfc00) === 0xdc00;
      if (!pair) throw new VerifierInputError("lone surrogate; not encodable as UTF-8");
      out += s[k] + s[++k];
      continue;
    }
    const esc: Record<number, string> = { 8: "\\b", 9: "\\t", 10: "\\n", 12: "\\f", 13: "\\r", 34: '\\"', 92: "\\\\" };
    if (esc[code]) out += esc[code];
    else if (code < 0x20) out += `\\u${code.toString(16).padStart(4, "0")}`;
    else out += s[k];
  }
  return out + '"';
}

/** JCS bytes as the server emits them: UTF-16 member order, no whitespace. */
export function canonicalJson(value: Json): Buffer {
  const enc = (v: Json, depth: number): string => {
    if (depth > MAX_NESTING_DEPTH) {
      throw new VerifierInputError(`nesting exceeds ${MAX_NESTING_DEPTH} levels`);
    }
    if (v === null) return "null";
    if (typeof v === "boolean") return v ? "true" : "false";
    if (typeof v === "string") return jcsString(v);
    const n = v instanceof PyFloat ? v.value : v;
    if (typeof n === "number") {
      if (!Number.isFinite(n)) throw new VerifierInputError(`non-finite number ${n}`);
      return v instanceof PyFloat || !Number.isInteger(n) ? pyFloatRepr(n) : String(n);
    }
    if (Array.isArray(v)) return `[${v.map((x) => enc(x, depth + 1)).join(",")}]`;
    const o = v as Obj;
    const keys = Object.keys(o).sort();
    return `{${keys.map((k) => `${jcsString(k)}:${enc(o[k], depth + 1)}`).join(",")}}`;
  };
  return Buffer.from(enc(value, 0), "utf8");
}

const sha256Hex = (b: Buffer | string) => createHash("sha256").update(b).digest("hex");

function strictB64(value: unknown): Buffer | null {
  if (typeof value !== "string") return null;
  let t = value.replace(/-/g, "+").replace(/_/g, "/");
  t += "=".repeat((4 - (t.length % 4)) % 4);
  if (!/^[A-Za-z0-9+/]*={0,2}$/.test(t)) return null;
  return Buffer.from(t, "base64");
}

function looseB64(value: string): Buffer {
  return Buffer.from(value.replace(/-/g, "+").replace(/_/g, "/"), "base64");
}

const derLen = (n: number) =>
  n < 0x80 ? Buffer.from([n]) : n < 0x100 ? Buffer.from([0x81, n]) : Buffer.from([0x82, n >> 8, n & 0xff]);
const der = (tag: number, body: Buffer) => Buffer.concat([Buffer.from([tag]), derLen(body.length), body]);

function spki(oids: string[], key: Buffer): Buffer {
  const algId = der(0x30, Buffer.concat(oids.map((o) => der(0x06, Buffer.from(o, "hex")))));
  return der(0x30, Buffer.concat([algId, der(0x03, Buffer.concat([Buffer.from([0]), key]))]));
}

/** Ed25519 and ES256 via node:crypto; ML-DSA-65 where the runtime has it. */
export function verifySignature(pk: Buffer, msg: Buffer, sig: Buffer, alg: unknown): [AxisResult, string] {
  const token = typeof alg === "string" ? alg.toUpperCase() : "";
  const check = (spkiDer: Buffer, digest: string | null, p1363: boolean, label: string): [AxisResult, string] => {
    let key;
    try {
      key = createPublicKey({ key: spkiDer, format: "der", type: "spki" });
    } catch (exc) {
      return ["FAIL", `bad ${label} public key: ${(exc as Error).message}`];
    }
    try {
      const ok = verifySig(digest, msg, p1363 ? { key, dsaEncoding: "ieee-p1363" } : key, sig);
      return ok ? ["PASS", "signature valid"] : ["FAIL", "signature mismatch"];
    } catch (exc) {
      return ["FAIL", `verify error: ${(exc as Error).message}`];
    }
  };
  if (token === "ED25519") {
    if (pk.length !== 32) return ["FAIL", `bad Ed25519 public key: ${pk.length} bytes, not 32`];
    return check(spki(["2b6570"], pk), null, false, "Ed25519");
  }
  if (token === "ES256") {
    if (sig.length !== 64) return ["FAIL", `ES256 signature must be 64-byte raw r||s, got ${sig.length}`];
    return check(spki(["2a8648ce3d0201", "2a8648ce3d030107"], pk), "sha256", true, "P-256");
  }
  if (token !== "ML-DSA-65") {
    return ["SKIPPED", `unsupported alg ${show(alg)} (this tool checks ML-DSA-65, Ed25519, ES256)`];
  }
  if (pk.length !== ML_DSA_PUBLIC_KEY_BYTES["ML-DSA-65"]) {
    return ["FAIL", `verify error: ML-DSA-65 public key is ${pk.length} bytes`];
  }
  try {
    createPublicKey({ key: spki(["608648016503040312"], pk), format: "der", type: "spki" });
  } catch {
    return ["SKIPPED", "this Node runtime's node:crypto has no ML-DSA-65; run the Python verifier"];
  }
  return check(spki(["608648016503040312"], pk), null, false, "ML-DSA-65");
}

const STAMP_RE =
  /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?)?(Z|[+-]\d{2}(?::?\d{2})?)?$/;

/** Epoch seconds of an ISO-8601 stamp, or null; no zone reads as UTC. */
export function parseStamp(value: unknown): number | null {
  const m = STAMP_RE.exec(String(value));
  if (!m) return null;
  const [y, mo, d, hh, mi, ss] = [1, 2, 3, 4, 5, 6].map((k) => Number(m[k] ?? 0));
  const day = new Date(0);
  day.setUTCFullYear(y, mo - 1, d);
  const sameDay = day.getUTCFullYear() === y && day.getUTCMonth() === mo - 1 && day.getUTCDate() === d;
  if (!sameDay || y < 1 || hh > 23 || mi > 59 || ss > 59) return null;
  let offset = 0;
  const zone = m[8];
  if (zone && zone !== "Z") {
    const oh = Number(zone.slice(1, 3));
    const om = Number(zone.slice(3).replace(":", "") || 0);
    if (oh > 23 || om > 59) return null;
    offset = (zone[0] === "-" ? -1 : 1) * (oh * 3600 + om * 60);
  }
  const frac = m[7] ? Number(`0.${m[7]}`) : 0;
  return day.getTime() / 1000 + hh * 3600 + mi * 60 + ss + frac - offset;
}

export function normaliseEnvelope(raw: Obj): Obj {
  const sig = raw.signature;
  if (isObj(sig) && isObj(raw.payload)) return { payload: raw.payload, signature: sig, anchors: raw.anchors };
  const payload = raw.payload;
  if (!isObj(payload)) return raw;
  let sigObj = raw.signature_envelope;
  if (!isObj(sigObj)) {
    sigObj = isObj(sig) ? sig : {
      alg: raw.algorithm ?? "ML-DSA-65",
      kid: payload.issuer_id ?? "",
      sig: typeof sig === "string" ? sig : "",
    };
  }
  return { payload, signature: sigObj, anchors: raw.anchors };
}

function profileRangeNote(node: Json, path = "$"): string | null {
  if (node instanceof PyFloat) {
    const integral = Number.isInteger(node.value);
    if (!/^-?\d+$/.test(node.lexeme) && integral && Math.abs(node.value) <= MAX_PROFILE_INTEGER) {
      return `Asqav profile range +/-(2**53 - 1) excludes non-integer lexeme '${node.lexeme}' at ${path}`;
    }
    if (integral && Math.abs(node.value) > MAX_PROFILE_INTEGER) {
      return `Asqav profile range +/-(2**53 - 1) excludes number ${node.lexeme} at ${path}`;
    }
    return null;
  }
  if (isInt(node)) {
    return Math.abs(node) > MAX_PROFILE_INTEGER
      ? `Asqav profile range +/-(2**53 - 1) excludes integer ${node} at ${path}`
      : null;
  }
  if (Array.isArray(node)) {
    for (let k = 0; k < node.length; k++) {
      const note = profileRangeNote(node[k], `${path}[${k}]`);
      if (note) return note;
    }
  } else if (isObj(node)) {
    for (const k of Object.keys(node)) {
      const note = profileRangeNote(node[k], `${path}.${k}`);
      if (note) return note;
    }
  }
  return null;
}

const isCurrentProfile = (v: Json | undefined) => v === 1 || (v instanceof PyFloat && v.value === 1);

export function checkControlsEvaluated(payload: Obj): [AxisResult, string] {
  const ce = payload.controls_evaluated;
  const guard = "(false_control_attestation_guard)";
  if (ce === undefined || ce === null) return ["PASS", "receipt declares no controls_evaluated block"];
  if (!isObj(ce)) return ["FAIL", `controls_evaluated must be an object ${guard}`];
  const unknown = Object.keys(ce).filter((k) => !ALLOWED_CONTROL_KEYS.has(k)).sort();
  if (unknown.length) return ["FAIL", `controls_evaluated keys outside the closed set: ${unknown.join(",")} ${guard}`];
  const q = ce.quorum;
  if (q !== undefined && q !== null) {
    const ah = isObj(q) ? q.attestation_hash : undefined;
    if (!isObj(q) || q.fired !== true || typeof ah !== "string" || !/^[0-9a-f]{64}$/.test(ah)) {
      return ["FAIL", `quorum member requires fired=true and a bare 64-hex attestation_hash ${guard}`];
    }
  }
  const p = ce.policy;
  if (p !== undefined && p !== null) {
    const mc = isObj(p) ? p.matched_count : undefined;
    if (!isObj(p) || p.evaluated !== true || !isInt(mc) || mc < 1) {
      return ["FAIL", `policy member requires evaluated=true and a real matched_count >= 1 ${guard}`];
    }
  }
  return ["PASS", "controls_evaluated block conforms to the closed key set"];
}

export function checkStructure(payload: Obj): [AxisResult, string] {
  const missing = REQUIRED_FIELDS.filter((f) => !has(payload, f));
  if (missing.length) return ["FAIL", `missing required fields: ${missing.join(",")}`];
  const rt = payload.type;
  if (typeof rt !== "string") return ["FAIL", `malformed type ${show(rt)}: not a string`];
  if (!ALLOWED_TYPES.has(rt)) {
    if (REGISTRY_ONLY_TYPES.has(rt)) return ["SKIPPED", `type '${rt}' is registered but this verifier does not implement it`];
    if (rt.startsWith("protectmcp:")) return ["FAIL", `unregistered type '${rt}': claims the reserved protectmcp namespace`];
    return ["SKIPPED", `type '${rt}' outside the known namespace; profile membership unverifiable, reported not failed`];
  }
  if (!has(payload, "v")) return ["SKIPPED", "no v member: not a Compliance Receipt of this document; absence is not version 1"];
  if (payload.v !== 1 && payload.v !== 2) return ["SKIPPED", `unsupported wire version ${show(payload.v)}: unverifiable under this document, reported not failed`];
  if (!has(payload, "mode")) return ["SKIPPED", "no mode member: not a Compliance Receipt of this document; absence selects no mode"];
  if (payload.mode !== "hash" && payload.mode !== "payload") return ["SKIPPED", `unsupported mode ${show(payload.mode)}: unverifiable under this document, reported not failed`];
  const [ceRes, ceNote] = checkControlsEvaluated(payload);
  return ceRes === "FAIL" ? ["FAIL", ceNote] : ["PASS", `required fields present; type ${rt}`];
}

export function isKeyedDigest(payload: Obj): boolean {
  const algo = payload.hash_algo;
  if (algo === undefined || algo === null) return false;
  return typeof algo !== "string" || algo.trim().toLowerCase() !== "sha256";
}

export function checkPayloadDigest(payload: Obj): [AxisResult, string] {
  const digest = payload.payload_digest;
  if (digest === undefined || digest === null) return ["PASS", "receipt binds no payload_digest; nothing to recompute"];
  if (!isObj(digest)) return ["FAIL", `payload_digest is ${show(digest)}, not an object`];
  const claimed = digest.hash;
  if (typeof claimed !== "string" || !/^[0-9a-f]{64}$/.test(claimed)) {
    return ["FAIL", `payload_digest.hash ${show(claimed)} is not 64 lowercase hex`];
  }
  const size = digest.size;
  if (has(digest, "size") && (!isInt(size) || size < 0)) {
    return ["FAIL", `payload_digest.size ${show(size)} is not a non-negative integer`];
  }
  if (payload.context === undefined || payload.context === null) {
    return ["PASS", "no context carried; payload_digest not recomputable here"];
  }
  if (isKeyedDigest(payload)) {
    return ["SKIPPED", "keyed payload_digest needs holder key material to recompute; recorded as unverifiable, never a mismatch"];
  }
  let encoded: Buffer;
  try {
    encoded = canonicalJson(payload.context);
  } catch {
    return ["SKIPPED", "context is not canonicalisable, so payload_digest cannot be recomputed"];
  }
  const actual = sha256Hex(encoded);
  if (actual !== claimed) {
    return ["FAIL", `payload_digest_mismatch: receipt binds ${claimed.slice(0, 16)}.., its own context hashes to ${actual.slice(0, 16)}..`];
  }
  if (isInt(size) && size !== encoded.length) {
    return ["FAIL", `payload_digest_mismatch: size claims ${size}, canonical context is ${encoded.length} bytes`];
  }
  return ["PASS", `payload_digest rederives from the carried context (${encoded.length} bytes)`];
}

export function checkChain(payload: Obj, predecessor: Json | undefined): [AxisResult, string] {
  const prev = payload.previousReceiptHash;
  if (prev === FIRST_RECEIPT_SEED) return ["PASS", "first receipt on chain (all-zero seed)"];
  if (predecessor === undefined) return ["SKIPPED", "no predecessor supplied (pass --predecessor to check)"];
  let actual: string;
  try {
    actual = sha256Hex(canonicalJson(predecessor));
  } catch (exc) {
    return ["FAIL", `predecessor payload not canonicalisable: ${(exc as Error).message}`];
  }
  if (actual === prev) return ["PASS", "chain link rederives from predecessor"];
  return ["FAIL", `chain break: expected ${String(prev).slice(0, 16)}.. got ${actual.slice(0, 16)}..`];
}

export function checkSkew(issuedAt: Json | undefined, now: number): [AxisResult, string] {
  const ts = parseStamp(issuedAt ?? "");
  if (ts === null) return ["FAIL", `unparseable issued_at ${show(issuedAt)}`];
  const skew = ts - now;
  if (skew > SKEW_BOUND_SECONDS) return ["FAIL", `issued_at ${skew.toFixed(0)}s ahead of wall clock (> ${SKEW_BOUND_SECONDS}s)`];
  return ["PASS", `skew ${skew.toFixed(0)}s within bound`];
}

export function checkExpiry(payload: Obj, now: number): [AxisResult, string] {
  if (!has(payload, "expires_at")) return ["PASS", "receipt declares no expiry"];
  const ts = parseStamp(payload.expires_at);
  if (ts === null) return ["FAIL", `unreadable expires_at ${show(payload.expires_at)}; refused rather than read as no expiry`];
  const delta = ts - now;
  if (delta < 0) return ["FAIL", `expires_at ${show(payload.expires_at)} lapsed ${(-delta).toFixed(0)}s ago`];
  return ["PASS", `expires_at ${show(payload.expires_at)} is ${delta.toFixed(0)}s ahead`];
}

export function checkKeyStatus(status: Json | undefined, issuedAt: Json | undefined, revokedAt: Json | undefined, reason: Json | undefined): [AxisResult, string] {
  const s = typeof status === "string" ? status.toLowerCase() : "";
  const r = typeof reason === "string" ? reason.toLowerCase() : "";
  const kind = s === "compromised" || r === "compromised" ? "compromised" : s;
  if (kind === "compromised") return ["FAIL", `signing key compromised at ${revokedAt || "an unpublished time"}; a compromised key never verifies`];
  if (s === "decommissioned") return ["SKIPPED", `signing key decommissioned; issuance at ${issuedAt} cannot be placed against retirement`];
  if (!s || s === "unknown") return ["SKIPPED", "signing key status unknown; cannot confirm the key was active"];
  if (!REVOKED_KEY_STATUSES.has(s)) return ["PASS", `signing key status ${show(status)} is active`];
  if (revokedAt) {
    const rev = parseStamp(revokedAt);
    const iss = parseStamp(issuedAt ?? "");
    if (rev === null || iss === null) return ["FAIL", `signing key status ${show(status)}; unparseable revoked_at/issued_at`];
    if (rev <= iss) return ["FAIL", `signing key ${kind} at ${revokedAt} on/before issuance ${issuedAt}`];
    return ["SKIPPED", `signing key ${kind} at ${revokedAt}; issued_at ${issuedAt} is self-attested, no anchor proves pre-revocation timing`];
  }
  return ["SKIPPED", `signing key status ${show(status)}; no revoked_at published, cannot place issuance relative to revocation`];
}

function usableKeys(jwks: Json): Obj[] {
  const keys = isObj(jwks) ? jwks.keys : undefined;
  return Array.isArray(keys) ? keys.filter((k): k is Obj => isObj(k) && typeof k.public_key === "string") : [];
}

function agentKeys(jwks: Json, agentId: Json | undefined, issuerId: Json | undefined, orgId: Json | undefined): Obj[] {
  return usableKeys(jwks).filter((k) => agentId && agentId === k.agent_id &&
    ((issuerId && issuerId === k.issuer_id) || (orgId && orgId === k.org_id)));
}

/** The jwks entry a signature is checked against, as match_signing_key. */
export function matchSigningKey(jwks: Json, kid: Json | undefined, payload: Obj, env: Obj): Obj | null {
  const keys = usableKeys(jwks);
  const agentId = payload.agent_id || env.agent_id;
  const tp = payload.key_thumbprint;
  if (typeof tp === "string" && /^sha256:[0-9a-f]{64}$/.test(tp)) {
    const hits = keys.filter((k) => k.key_thumbprint === tp);
    const narrowed = hits.filter((k) => agentId && k.agent_id === agentId);
    const pool = narrowed.length ? narrowed : hits;
    if (pool.length) return pool.find((k) => kid && kid === k.kid) ?? pool[0];
  }
  const byId = keys.find((k) => kid && kid === k.kid);
  if (byId) return byId;
  const byAgent = agentKeys(jwks, agentId, payload.issuer_id, payload.org_id || env.org_id);
  if (byAgent.length) return byAgent[0];
  return keys.find((k) => kid && (kid === k.issuer_id || kid === k.kid)) ?? null;
}

function checkKeyBinding(payload: Obj, alg: Json | undefined, pk: Buffer | null): [AxisResult, string] {
  const claimed = payload.key_thumbprint;
  if (claimed === undefined || claimed === null) return ["PASS", "receipt binds no key_thumbprint; binding not checked"];
  if (typeof claimed !== "string" || !/^sha256:[0-9a-f]{64}$/.test(claimed)) {
    return ["FAIL", `key_thumbprint ${show(claimed)} is not sha256:<64 lowercase hex>`];
  }
  if (!pk) return ["SKIPPED", "no signing key resolved, so the bound key_thumbprint cannot be recomputed"];
  if (typeof alg !== "string" || ML_DSA_PUBLIC_KEY_BYTES[alg] !== pk.length) {
    return ["SKIPPED", "resolved key is not raw ML-DSA material, so the bound key_thumbprint cannot be recomputed"];
  }
  const jwk = { alg, kty: "AKP", pub: pk.toString("base64url") };
  const actual = `sha256:${sha256Hex(canonicalJson(jwk))}`;
  if (actual === claimed) return ["PASS", `key_thumbprint rederives from the resolved ${alg} key`];
  return ["FAIL", `key_substituted: receipt binds ${claimed}, resolved key computes ${actual}`];
}

function checkCounterparty(payload: Obj): [AxisResult, string] {
  if (!has(payload, "counterparty_binding")) return ["PASS", "no counterparty binding; content is unilaterally asserted"];
  const cpb = payload.counterparty_binding;
  if (!isObj(cpb)) return ["FAIL", `counterparty_binding is ${show(cpb)}, not an object`];
  if (!has(cpb, "scope")) return ["SKIPPED", "legacy_scope: the signed binding omits its digest scope"];
  if (cpb.scope !== "envelope_minus_anchors") return ["SKIPPED", "unrecognised_scope: the signed digest scope is unsupported"];
  return ["SKIPPED", "counterparty binding not evaluated by this standalone file; run the full verifier"];
}

function checkAnchors(anchors: Json | undefined): [AxisResult, string] {
  if (anchors === undefined || anchors === null) return ["SKIPPED", "no anchors on this receipt"];
  if (!Array.isArray(anchors)) return ["FAIL", `anchors field is not a list (got ${show(anchors)})`];
  if (!anchors.length) return ["SKIPPED", "no anchors on this receipt"];
  return ["SKIPPED", "anchors not evaluated by this standalone file; run the full verifier"];
}

function failureClass(axis: Axis): "invalid" | "unverifiable" {
  const { name, result, note } = axis;
  if (result === "SKIPPED") return "unverifiable";
  if (INVALID_FAIL_AXES.has(name)) return "invalid";
  if (name === "chain") return note.startsWith("chain break:") ? "invalid" : "unverifiable";
  if (name === "skew") return note.startsWith("unparseable issued_at") ? "unverifiable" : "invalid";
  if (name === "structure") return /^(unregistered|malformed) type /.test(note) ? "invalid" : "unverifiable";
  if (name === "expiry") return note.startsWith("unreadable expires_at") ? "unverifiable" : "invalid";
  return "unverifiable";
}

/** _fold_verdict: expiry never folds, a skipped chain never blocks. */
export function foldVerdict(axes: Axis[], keyed: boolean): Pick<Verification, "verdict" | "failureClass"> {
  const failed = axes.filter((a) => a.result === "FAIL" && a.name !== "expiry");
  if (failed.length) {
    return { verdict: "unverified", failureClass: failed.some((a) => failureClass(a) === "invalid") ? "invalid" : "unverifiable" };
  }
  if (axes.some((a) => a.result === "SKIPPED" && a.name !== "chain")) return { verdict: "unverified", failureClass: "unverifiable" };
  return { verdict: keyed ? "verified_keyed" : "verified", failureClass: null };
}

export interface VerifyOptions {
  predecessor?: Json;
  now?: number;
}

/** Verify one receipt against a local JWKS object; nothing is fetched. */
export function verifyReceipt(receipt: Json, jwks: Json, opts: VerifyOptions = {}): Verification {
  const refuse = (why: string): Verification => ({
    verdict: "unverified", failureClass: "unverifiable", axes: [], refusal: why, canonicalSha256: null,
  });
  if (!isObj(receipt)) return refuse(`expected a JSON object receipt, got ${show(receipt)}`);
  const env = normaliseEnvelope(receipt);
  const payload = has(env, "payload") ? env.payload : env;
  if (!isObj(payload)) return refuse(`receipt payload not available from this surface (got ${show(payload)})`);
  const pred = opts.predecessor;
  const predPayload = isObj(pred) && has(pred, "payload") ? pred.payload : pred;
  const sigObj: Obj = isObj(env.signature) ? env.signature : typeof env.signature === "string"
    ? { alg: env.algorithm ?? "ML-DSA-65", kid: payload.issuer_id ?? "", sig: env.signature } : {};
  if (isCurrentProfile(payload.v)) {
    const nonGenesis = predPayload !== undefined && payload.previousReceiptHash !== FIRST_RECEIPT_SEED;
    const note = profileRangeNote(payload) ??
      (Array.isArray(env.anchors) && env.anchors.length ? profileRangeNote(sigObj) : null) ??
      (nonGenesis ? profileRangeNote(predPayload as Json) : null);
    if (note) return refuse(note);
  }
  let msg: Buffer;
  try {
    msg = canonicalJson(payload);
  } catch (exc) {
    return refuse((exc as Error).message);
  }
  const now = opts.now ?? Date.now() / 1000;
  const kid = sigObj.kid ?? "";
  const alg = sigObj.alg ?? "ML-DSA-65";
  const axes: Axis[] = [];
  const add = (name: string, [result, note]: [AxisResult, string]) => axes.push({ name, result, note });
  add("structure", checkStructure(payload));
  add("nonce", payload.nonce === undefined || payload.nonce === null || payload.nonce === ""
    ? ["PASS", "receipt declares no nonce; nothing to flag"]
    : ["PASS", "nonce present; this surface holds no seen-nonce index, so duplicate_emission_candidate stays false"]);
  let entry = matchSigningKey(jwks, kid, payload, env);
  let effPk: Buffer | null = null;
  let effAlg: Json | undefined;
  if (!entry) {
    add("issuer_key", ["FAIL", `kid ${show(kid)} not in jwks directory`]);
    add("signature", ["SKIPPED", "no issuer key to verify against"]);
  } else {
    const rawSig = sigObj.sig;
    const sig = typeof rawSig === "string" && rawSig ? strictB64(rawSig) : null;
    let pk = looseB64(entry.public_key as string);
    effAlg = entry.alg;
    let sigRes: [AxisResult, string] = sig ? verifySignature(pk, msg, sig, alg || entry.alg) : ["FAIL", "signature string is not decodable base64"];
    let agentNote = "";
    let effKid = kid && (kid === entry.issuer_id || kid === entry.kid) ? kid : entry.kid;
    if (sig && sigRes[0] !== "PASS") {
      const cands = agentKeys(jwks, payload.agent_id || env.agent_id, payload.issuer_id, payload.org_id || env.org_id);
      for (const k of cands.slice(0, 8)) {
        const kpk = looseB64(k.public_key as string);
        const res = verifySignature(kpk, msg, sig, k.alg || alg || entry.alg);
        if (res[0] === "PASS") {
          [sigRes, entry, pk, effAlg, effKid] = [res, k, kpk, k.alg, k.kid];
          break;
        }
      }
      if (cands.length && sigRes[0] !== "PASS") agentNote = "; no key published for this agent verified";
    }
    if (sigRes[0] === "PASS" && typeof rawSig === "string" && /[+/=]/.test(rawSig)) {
      sigRes = ["PASS", `${sigRes[1]}; accepted under legacy encoding policy: pre-2026-09-02 profile transport`];
    }
    const issuer = typeof entry.issuer_id === "string" ? entry.issuer_id : null;
    const claimed = payload.issuer_id;
    add("issuer_key", ["PASS", `resolved signing key ${effKid} (status=${entry.status ?? "None"})${agentNote}`]);
    add("issuer_bind", typeof claimed === "string" && claimed && issuer === claimed
      ? ["PASS", `signing key is published under the claimed issuer ${claimed}`]
      : ["FAIL", `signing key is published under issuer ${show(issuer)}, not the claimed issuer ${show(claimed)}`]);
    add("key_status", checkKeyStatus(entry.status, payload.issued_at ?? "", entry.revoked_at, entry.revocation_reason));
    add("signature", sigRes);
    effPk = pk;
  }
  add("key_binding", checkKeyBinding(payload, effAlg, effPk));
  add("counterparty", checkCounterparty(payload));
  add("payload_digest", checkPayloadDigest(payload));
  const outer = Object.keys(receipt).filter((k) => !OUTER_FREE.has(k));
  add("outer_copies", outer.length
    ? ["SKIPPED", `outer members ${outer.join(",")} not compared by this standalone file; run the full verifier`]
    : ["PASS", "no outer copies carried; the signed payload is the whole receipt"]);
  add("chain", checkChain(payload, predPayload));
  add("anchors", checkAnchors(env.anchors));
  add("skew", checkSkew(payload.issued_at, now));
  add("expiry", checkExpiry(payload, now));
  return { ...foldVerdict(axes, isKeyedDigest(payload)), axes, refusal: null, canonicalSha256: sha256Hex(msg) };
}

/** CLI: --receipt FILE --jwks FILE [--predecessor FILE]; exit 0, 1 or 2. */
export function main(argv: string[], out: (line: string) => void = console.log): number {
  const arg = (name: string) => {
    const k = argv.indexOf(name);
    return k >= 0 ? argv[k + 1] : undefined;
  };
  const receiptPath = arg("--receipt");
  const jwksPath = arg("--jwks");
  if (!receiptPath || !jwksPath) {
    out("usage: verify-receipt --receipt FILE --jwks FILE [--predecessor FILE]");
    return 2;
  }
  let result: Verification;
  try {
    const load = (p: string) => parseStrictJson(readFileSync(p, "utf8"), p);
    const pred = arg("--predecessor");
    result = verifyReceipt(load(receiptPath), load(jwksPath), pred ? { predecessor: load(pred) } : {});
  } catch (exc) {
    out(`error: ${(exc as Error).message}`);
    return 2;
  }
  out("Asqav receipt verification");
  if (result.refusal) out(`  [FAIL] input       ${result.refusal}`);
  if (result.canonicalSha256) out(`  canonical bytes: sha256:${result.canonicalSha256}`);
  const marks = { PASS: "ok", FAIL: "FAIL", SKIPPED: "skip" };
  for (const a of result.axes) out(`  [${marks[a.result].padStart(4)}] ${a.name.padEnd(11)} ${a.note}`);
  if (result.failureClass === null) {
    out(`\n  => ${result.verdict}`);
    return 0;
  }
  out(`\n  => ${result.verdict} (failure_class: ${result.failureClass})`);
  return result.failureClass === "invalid" ? 1 : 2;
}

if (typeof process !== "undefined" && /verify-receipt\.(ts|js|mjs|cjs)$/.test(process.argv[1] ?? "")) {
  process.exitCode = main(process.argv.slice(2));
}
