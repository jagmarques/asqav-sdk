/**
 * A focused port of the standalone `verify_receipt.py` helpers the Asqav-native
 * and ACTA adapters reuse, so the TS oracle reproduces the same bytes and the
 * same structure verdict as the Python surface.
 *
 * Only the pieces the adapters touch are ported: base64 decoding, envelope
 * normalisation, the Asqav-native structure check, JWKS key resolution, and the
 * first-receipt seed constant.
 */

import type { VerifyState } from "./crypto.js";

/** Mirrors `core/integrity.py` FIRST_RECEIPT_SEED (64 zeros). */
export const FIRST_RECEIPT_SEED = "0".repeat(64);

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
  const keys = (jwks?.keys as Array<Record<string, unknown>> | undefined) ?? [];
  for (const k of keys) {
    if (kid && (kid === k.issuer_id || kid === k.kid)) {
      if (typeof k.public_key !== "string") continue;
      return k;
    }
  }
  return null;
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
  return k !== null && typeof k.org_id === "string" ? k.org_id : null;
}

// org.id is a UUID, so a published issuer that does not parse as one is a
// legal-entity label the verifier cannot resolve to an org offline.
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** Bind a hash-mode receipt's org_id to the org the JWKS names (mirrors `check_org_binding`). */
export function checkOrgBinding(
  keyIssuerId: string | null,
  keyOrgId: string | null,
  claimedOrgId: unknown,
): readonly [VerifyState, string] {
  if (typeof claimedOrgId !== "string" || claimedOrgId === "") {
    return ["FAIL", `receipt org_id is ${JSON.stringify(claimedOrgId ?? null)}, so there is no org to bind`];
  }
  if (claimedOrgId === keyOrgId || claimedOrgId === keyIssuerId) {
    return ["PASS", `signing key is published under the claimed org ${claimedOrgId}`];
  }
  if (keyOrgId === null && !(typeof keyIssuerId === "string" && UUID_RE.test(keyIssuerId))) {
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

// Mirrors Python REVOKED_KEY_STATUSES; receipts from these keys must not PASS offline.
const REVOKED_KEY_STATUSES = new Set(["revoked", "suspended", "compromised"]);

// Gate on the key's JWKS status (mirrors `check_key_status`).
// With revoked_at and a trusted anchor, receipts signed before revocation PASS.
// Without revoked_at, any revoked-status key FAILs the axis.
// Without an anchor, a pre-revocation issued_at is self-attested and cannot be
// trusted: a holder of the compromised key can backdate. Downgrade to SKIPPED so
// the verdict is INCOMPLETE, never a hiding PASS.
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
