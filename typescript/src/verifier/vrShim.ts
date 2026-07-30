/**
 * A focused port of the standalone `verify_receipt.py` helpers the Asqav-native
 * and ACTA adapters reuse, so the TS oracle reproduces the same bytes and the
 * same structure verdict as the Python surface.
 *
 * Only the pieces the adapters touch are ported: base64 decoding, envelope
 * normalisation, the Asqav-native structure check, JWKS key resolution, and the
 * first-receipt seed constant.
 */

import { asqavJcs } from "./canonical.js";
import { sha256Hex, type VerifyState } from "./crypto.js";

/** Mirrors `core/integrity.py` FIRST_RECEIPT_SEED (64 zeros). */
export const FIRST_RECEIPT_SEED = "0".repeat(64);

/** Wall-clock bound on `issued_at`, mirrors Python `SKEW_BOUND_SECONDS`. */
export const SKEW_BOUND_SECONDS = 300;

const REQUIRED_FIELDS = [
  "type",
  "issued_at",
  "issuer_id",
  "action_ref",
  "payload_digest",
  "policy_digest",
  "previousReceiptHash",
  "decision",
] as const;

const ALLOWED_TYPES = new Set([
  "protectmcp:decision",
  "protectmcp:restraint",
  "protectmcp:lifecycle",
  "protectmcp:lifecycle:configuration_change",
  "protectmcp:lifecycle:risk_acceptance",
  "protectmcp:lifecycle:code_authorship",
  "protectmcp:acknowledgment",
  "protectmcp:observation",
  "protectmcp:observation:result_bound",
]);

/** Decode standard or url-safe base64, padding-tolerant (mirrors `_b64decode`). */
export function b64decode(value: string): Uint8Array {
  let s = value.replace(/-/g, "+").replace(/_/g, "/");
  s += "=".repeat((-s.length % 4 + 4) % 4);
  return new Uint8Array(Buffer.from(s, "base64"));
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

/**
 * Remap a hosted /verify response into the canonical 3-key envelope. Already
 * canonical or non-hosted shapes pass through unchanged (mirrors
 * `normalise_envelope`).
 */
export function normaliseEnvelope(raw: Record<string, unknown>): Record<string, unknown> {
  const sig = raw.signature;
  if (isRecord(sig) && isRecord(raw.payload)) {
    return raw;
  }
  const payload = raw.payload;
  if (!isRecord(payload)) {
    return raw;
  }
  let sigObj = raw.signature_envelope;
  if (!isRecord(sigObj)) {
    if (isRecord(sig)) {
      sigObj = sig;
    } else {
      sigObj = {
        alg: raw.algorithm ?? "ML-DSA-65",
        kid: payload.issuer_id ?? "",
        sig: typeof sig === "string" ? sig : "",
      };
    }
  }
  return {
    payload,
    signature: sigObj,
    anchors: raw.anchors ?? [],
  };
}

/** Asqav-native structure check; returns `[result, note]` (mirrors `check_structure`). */
export function checkStructure(payload: Record<string, unknown>): readonly ["PASS" | "FAIL", string] {
  const missing = REQUIRED_FIELDS.filter((f) => !(f in payload));
  if (missing.length > 0) {
    return ["FAIL", `missing required fields: ${missing.join(",")}`];
  }
  const rt = payload.type;
  if (typeof rt !== "string" || !ALLOWED_TYPES.has(rt)) {
    return ["FAIL", `type ${JSON.stringify(rt)} outside the allowed namespace`];
  }
  return ["PASS", `required fields present; type ${rt}`];
}

// One matcher (mirrors `_match_key`), so key bytes and the key's published
// issuer always come from the same JWKS entry.
function matchKey(
  jwks: Record<string, unknown> | null,
  kid: string,
): Record<string, unknown> | null {
  const keys = jwks?.keys;
  if (!Array.isArray(keys)) return null;
  for (const k of keys as Array<Record<string, unknown>>) {
    if (k === null || typeof k !== "object") continue;
    if (kid && (kid === k.issuer_id || kid === k.kid)) {
      if (typeof k.public_key !== "string") continue;
      return k;
    }
  }
  return null;
}

// Exact key id (mirrors `_match_key_by_id`). A key id names one key, while the
// issuer match above answers for every key an org publishes, which is a set.
function matchKeyById(
  jwks: Record<string, unknown> | null,
  kid: string,
): Record<string, unknown> | null {
  const keys = jwks?.keys;
  if (!Array.isArray(keys)) return null;
  for (const k of keys as Array<Record<string, unknown>>) {
    if (k === null || typeof k !== "object") continue;
    if (kid && kid === k.kid) {
      if (typeof k.public_key !== "string") continue;
      return k;
    }
  }
  return null;
}

// An agent key bound to the issuer the receipt claims (mirrors `_match_key_by_agent`).
// agent_id is attacker-controlled, so the key's published issuer must equal the one
// the receipt names inside its signed bytes.
function matchKeyByAgent(
  jwks: Record<string, unknown> | null,
  agentId: unknown,
  issuerId: unknown,
): Record<string, unknown> | null {
  const keys = jwks?.keys;
  if (!Array.isArray(keys)) return null;
  for (const k of keys as Array<Record<string, unknown>>) {
    if (k === null || typeof k !== "object") continue;
    if (agentId && agentId === k.agent_id && issuerId && issuerId === k.issuer_id) {
      if (typeof k.public_key !== "string") continue;
      return k;
    }
  }
  return null;
}

/**
 * The one JWKS entry a receipt's signature is checked against (mirrors `match_signing_key`).
 *
 * Exact key id, then the agent bind, then the bare-kid issuer match. A cloud receipt
 * puts the org id in kid and signs with the agent's own key, and the directory
 * publishes issuer_id on every key that org owns, so an org-shaped kid matches each
 * sibling alike and list position picks one. Position binds nothing: agent_id and
 * issuer_id both sit inside the signed bytes, so the pair names the signer, while the
 * sibling it lands on holds other key bytes and another revocation status.
 */
export function matchSigningKey(
  jwks: Record<string, unknown> | null,
  kid: string,
  agentId?: unknown,
  issuerId?: unknown,
): Record<string, unknown> | null {
  return (
    matchKeyById(jwks, kid) ?? matchKeyByAgent(jwks, agentId, issuerId) ?? matchKey(jwks, kid)
  );
}

/** Return the issuer id a JWKS entry publishes (mirrors `key_issuer_of`). */
export function keyIssuerOf(entry: Record<string, unknown> | null): string | null {
  return entry !== null && typeof entry.issuer_id === "string" ? entry.issuer_id : null;
}

/** Return the org id a JWKS entry publishes (mirrors `key_org_of`). */
export function keyOrgOf(entry: Record<string, unknown> | null): string | null {
  // A value that is not an org id cannot serve as one, so it reads as unpublished.
  return entry !== null && isOrgId(entry.org_id) ? (entry.org_id as string) : null;
}

/** Return the revoked_at a JWKS entry publishes (mirrors `revoked_at_of`). */
export function revokedAtOf(entry: Record<string, unknown> | null): string | null {
  return entry !== null && typeof entry.revoked_at === "string" ? entry.revoked_at : null;
}

/** Return `[publicKeyBytes, status, alg]` for `kid` from a JWKS dict (mirrors `resolve_key`). */
export function resolveKey(
  jwks: Record<string, unknown> | null,
  kid: string,
): readonly [Uint8Array | null, string | null, string | null] {
  const k = matchKey(jwks, kid);
  if (k === null) return [null, null, null];
  return [b64decode(k.public_key as string), (k.status as string) ?? null, (k.alg as string) ?? null];
}

/** Return the issuer id the JWKS publishes for `kid` (mirrors `resolve_key_issuer`). */
export function resolveKeyIssuer(jwks: Record<string, unknown> | null, kid: string): string | null {
  const k = matchKey(jwks, kid);
  return k !== null && typeof k.issuer_id === "string" ? k.issuer_id : null;
}

/** PASS only when the JWKS publishes the verifying key under the claimed
 *  issuer, so a signature alone never proves authorship (`check_issuer_binding`). */
export function checkIssuerBinding(
  keyIssuerId: string | null,
  claimedIssuerId: unknown,
): readonly [VerifyState, string] {
  if (
    typeof claimedIssuerId === "string" &&
    claimedIssuerId !== "" &&
    keyIssuerId === claimedIssuerId
  ) {
    return ["PASS", `signing key is published under the claimed issuer ${claimedIssuerId}`];
  }
  return [
    "FAIL",
    `signing key is published under issuer ${JSON.stringify(keyIssuerId)}, not the claimed issuer ${JSON.stringify(claimedIssuerId ?? null)}`,
  ];
}

/** Return the JWKS revoked_at for kid if published, else null (mirrors `resolve_revoked_at`). */
export function resolveRevokedAt(jwks: Record<string, unknown> | null, kid: string): string | null {
  const k = matchKey(jwks, kid);
  return k !== null && typeof k.revoked_at === "string" ? k.revoked_at : null;
}

/** Return the org id the JWKS publishes for `kid` (mirrors `resolve_key_org`). */
export function resolveKeyOrg(jwks: Record<string, unknown> | null, kid: string): string | null {
  const k = matchKey(jwks, kid);
  // A value that is not an org id cannot serve as one, so it reads as unpublished.
  return k !== null && isOrgId(k.org_id) ? (k.org_id as string) : null;
}

// Same body text as the Python _ORG_ID_RE, which anchors with fullmatch instead,
// since python's $ also matches before a trailing newline and this one does not.
const ORG_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** True for a canonical dashed UUID, the only form an org id takes on the wire. */
function isOrgId(value: unknown): boolean {
  return typeof value === "string" && ORG_ID_RE.test(value);
}

// Hex case carries no meaning in a UUID, so two spellings name one org. Folding
// is limited to org ids, which are ASCII, so both languages fold identically.
function orgKey(value: unknown): unknown {
  return isOrgId(value) ? (value as string).toLowerCase() : value;
}

/** Bind a hash-mode receipt's org_id to the org the JWKS names (mirrors `check_org_binding`). */
export function checkOrgBinding(
  keyIssuerId: string | null,
  keyOrgId: string | null,
  claimedOrgId: unknown,
): readonly [VerifyState, string] {
  if (typeof claimedOrgId !== "string" || claimedOrgId === "") {
    return ["FAIL", `receipt org_id is ${JSON.stringify(claimedOrgId ?? null)}, so there is no org to bind`];
  }
  const claimKey = orgKey(claimedOrgId);
  if (claimKey === orgKey(keyOrgId) || claimKey === orgKey(keyIssuerId)) {
    return ["PASS", `signing key is published under the claimed org ${claimedOrgId}`];
  }
  if (keyOrgId === null && !isOrgId(keyIssuerId)) {
    return [
      "SKIPPED",
      `jwks names issuer ${JSON.stringify(keyIssuerId)} for this key, a label rather than an org id, so org ${claimedOrgId} cannot be confirmed offline; publish org_id per key to close this`,
    ];
  }
  return [
    "FAIL",
    `signing key is published under org ${JSON.stringify(keyOrgId ?? keyIssuerId)}, not the claimed ${JSON.stringify(claimedOrgId)}`,
  ];
}

// Python spells these three differently and the note text is read side by side.
function pyStr(v: unknown): string {
  if (v === null || v === undefined) return "None";
  if (v === true) return "True";
  if (v === false) return "False";
  return String(v);
}

// Python's repr, for notes read side by side with the Python half's output.
// Strings carry single quotes there, so JSON.stringify would print a different note.
function pyRepr(v: unknown): string {
  if (typeof v !== "string") return pyStr(v);
  const body = v.replace(/\\/g, "\\\\");
  // Python switches to double quotes when the text holds a single quote and no double.
  if (body.includes("'") && !body.includes('"')) return `"${body}"`;
  return `'${body.replace(/'/g, "\\'")}'`;
}

// Python names the type in the malformed-value notes, so mirror those names.
function pyTypeName(v: unknown): string {
  if (v === null || v === undefined) return "NoneType";
  if (Array.isArray(v)) return "list";
  if (typeof v === "boolean") return "bool";
  if (typeof v === "number") return Number.isInteger(v) ? "int" : "float";
  if (typeof v === "string") return "str";
  return "dict";
}

// Python truthiness, which differs from JavaScript's for [] and {}.
function pyTruthy(v: unknown): boolean {
  if (v === null || v === undefined || v === false) return false;
  if (typeof v === "number") return v !== 0 && !Number.isNaN(v);
  if (typeof v === "string") return v.length > 0;
  if (Array.isArray(v)) return v.length > 0;
  if (typeof v === "object") return Object.keys(v as object).length > 0;
  return true;
}

// Mirrors base64.b64decode(validate=True): alphabet characters, then at most two pads
const B64_STRICT = /^[A-Za-z0-9+/]*={0,2}$/;

/**
 * True when Python's `_safe_b64` accepts `value`, false otherwise.
 *
 * Strict on purpose: an out-of-alphabet character is refused, not dropped, so a
 * forged all-punctuation anchor cannot read as present. A non-ASCII codepoint
 * falls outside the alphabet here, matching the ascii encode Python raises on.
 */
function safeB64(value: unknown): boolean {
  if (typeof value !== "string") return false;
  const s = value.replace(/-/g, "+").replace(/_/g, "/");
  const padded = s + "=".repeat(((-s.length % 4) + 4) % 4);
  if (!B64_STRICT.test(padded)) return false;
  // Padding carries no byte, so an all-pad value is not an anchor
  return padded.replace(/=+$/, "").length > 0;
}

/** JCS bytes of the envelope with `anchors` removed (mirrors `envelope_minus_anchors_jcs`). */
export function envelopeMinusAnchorsJcs(env: Record<string, unknown>): Uint8Array {
  const e = { ...env };
  delete e.anchors;
  return asqavJcs(e);
}

// ISO 8601 shapes Python's fromisoformat accepts, spelled out so a lenient JS date
// string Python rejects also FAILs. Only an uppercase Z is a zone designator.
const ISO_STAMP =
  /^(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?))?(Z|[+-]\d{2}(?::?\d{2})?)?$/;

// Python treats a stamp with no zone as UTC, where JS would read it as local time.
// Ranges are checked here because Date.parse accepts hour 24 and Python does not.
function parseIsoMs(issuedAt: unknown): number | null {
  if (typeof issuedAt !== "string") return null;
  const m = ISO_STAMP.exec(issuedAt);
  if (m === null) return null;
  const [, date, time, zone] = m;
  if (time !== undefined) {
    const [hh, mm, ss] = time.split(":");
    if (Number(hh) > 23 || Number(mm) > 59 || Number(ss ?? "0") > 59) return null;
  }
  let tz = zone ?? "Z";
  if (tz !== "Z") {
    const digits = tz.replace(":", "");
    const oh = Number(digits.slice(1, 3));
    const om = digits.length > 3 ? Number(digits.slice(3)) : 0;
    if (oh > 23 || om > 59) return null;
    tz = `${digits[0]}${String(oh).padStart(2, "0")}:${String(om).padStart(2, "0")}`;
  }
  const ms = Date.parse(`${date}T${time ?? "00:00:00"}${tz}`);
  return Number.isNaN(ms) ? null : ms;
}

/** `issued_at` within the wall-clock bound; returns `[result, note]` (mirrors `check_skew`). */
export function checkSkew(issuedAt: unknown): readonly [VerifyState, string] {
  const ms = parseIsoMs(issuedAt);
  if (ms === null) {
    return ["FAIL", `unparseable issued_at ${JSON.stringify(issuedAt ?? null)}`];
  }
  const skew = (ms - Date.now()) / 1000;
  if (skew > SKEW_BOUND_SECONDS) {
    return ["FAIL", `issued_at ${skew.toFixed(0)}s ahead of wall clock (> ${SKEW_BOUND_SECONDS}s)`];
  }
  return ["PASS", `skew ${skew.toFixed(0)}s within bound`];
}

/**
 * Refuse a receipt past the expiry its own signer committed to (mirrors `check_expiry`).
 *
 * Mirrors the hosted signature_expired verdict. The window is read from inside
 * the signed bytes, so an unsigned envelope field can never move it, and an
 * unreadable value fails closed instead of reading as no window at all.
 */
export function checkExpiry(payload: unknown): readonly [VerifyState, string] {
  if (!isRecord(payload) || !("expires_at" in payload)) {
    return ["PASS", "receipt declares no expiry"];
  }
  const raw = payload.expires_at;
  const ms = parseIsoMs(raw);
  if (ms === null) {
    return ["FAIL", `unreadable expires_at ${pyRepr(raw)}; refused rather than read as no expiry`];
  }
  const delta = (ms - Date.now()) / 1000;
  if (delta < 0) {
    return ["FAIL", `expires_at ${pyStr(raw)} lapsed ${(-delta).toFixed(0)}s ago`];
  }
  return ["PASS", `expires_at ${pyStr(raw)} is ${delta.toFixed(0)}s ahead`];
}

/**
 * Report which envelope each anchor binds (mirrors `check_anchors`).
 *
 * Absent or null anchors is a legitimate no-anchors receipt (SKIPPED). A present
 * non-list value is malformed and FAILs, never laundered to an empty list.
 * `anchors` sits outside the signed bytes, so a forged envelope can move it.
 */
export function checkAnchors(envelope: Record<string, unknown>): readonly [VerifyState, string] {
  const anchors = envelope.anchors;
  if (anchors === null || anchors === undefined) {
    return ["SKIPPED", "no anchors on this receipt"];
  }
  if (!Array.isArray(anchors)) {
    return ["FAIL", `anchors field is not a list (got ${pyTypeName(anchors)})`];
  }
  if (anchors.length === 0) {
    return ["SKIPPED", "no anchors on this receipt"];
  }
  let bound: string;
  try {
    bound = sha256Hex(envelopeMinusAnchorsJcs(envelope));
  } catch {
    // Defense in depth: core's depth gate should already have rejected this.
    return ["FAIL", "envelope too deeply nested to canonicalise for anchor binding"];
  }
  const lines = [`anchors bind envelope digest sha256:${bound.slice(0, 16)}..`];
  let allOk = true;
  for (const a of anchors as unknown[]) {
    if (a === null || typeof a !== "object" || Array.isArray(a)) {
      allOk = false;
      lines.push(`    - malformed anchor entry (got ${pyTypeName(a)}, expected an object)`);
      continue;
    }
    const entry = a as Record<string, unknown>;
    const atype = "type" in entry ? entry.type : "?";
    const val = entry.value;
    const ok = pyTruthy(val) && safeB64(val);
    allOk = allOk && ok;
    const state = ok ? "present, base64-ok" : "MISSING or malformed";
    lines.push(`    - ${pyStr(atype)}: value ${state}`);
  }
  return [allOk ? "PASS" : "FAIL", lines.join("; ")];
}

// Mirrors Python REVOKED_KEY_STATUSES; receipts from these keys must not PASS offline.
const REVOKED_KEY_STATUSES = new Set(["revoked", "suspended", "compromised"]);

// Gate on the key's JWKS status (mirrors `check_key_status`). With revoked_at and a trusted
// anchor, pre-revocation receipts PASS, and without revoked_at any revoked-status key FAILs.
// Without an anchor, issued_at is self-attested (backdateable), so downgrade to SKIPPED.
export function checkKeyStatus(
  status: string | null,
  issuedAt: string,
  revokedAt: string | null,
  hasTrustedAnchor: boolean,
): readonly [VerifyState, string] {
  const s = (status ?? "").toLowerCase();
  if (!REVOKED_KEY_STATUSES.has(s)) {
    return ["PASS", `signing key status ${JSON.stringify(status)} is active`];
  }
  if (revokedAt) {
    try {
      const rev = new Date(revokedAt).getTime();
      const iss = new Date(issuedAt).getTime();
      if (isNaN(rev) || isNaN(iss)) {
        return ["FAIL", `signing key status ${JSON.stringify(status)}; unparseable revoked_at/issued_at`];
      }
      if (rev <= iss) {
        return ["FAIL", `signing key revoked at ${revokedAt} on/before issuance ${issuedAt}`];
      }
      if (!hasTrustedAnchor) {
        return ["SKIPPED", `signing key revoked at ${revokedAt}; issued_at ${issuedAt} is self-attested, no anchor proves pre-revocation timing`];
      }
      return ["PASS", `signing key revoked at ${revokedAt}, after issuance ${issuedAt}`];
    } catch {
      return ["FAIL", `signing key status ${JSON.stringify(status)}; unparseable revoked_at/issued_at`];
    }
  }
  return ["FAIL", `signing key status ${JSON.stringify(status)}; receipt cannot be trusted`];
}
