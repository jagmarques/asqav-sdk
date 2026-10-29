/** Oversight-attestation states the platform reports, signature-gated. */

import { asqavJcs } from "./canonical.js";
import { PASS, verifySignature } from "./crypto.js";
import { b64decode, checkKeyStatus, revokedAtOf } from "./vrShim.js";

export const STATES = {
  notApplicable: "not_applicable",
  missing: "missing/unverified",
  asserted: "asserted/unverified",
  signatureVerified: "signature-verified",
} as const;

export const NOT_APPLICABLE = STATES.notApplicable;
export const MISSING = STATES.missing;
export const ASSERTED = STATES.asserted;
export const SIGNATURE_VERIFIED = STATES.signatureVerified;

export const METHODS: ReadonlySet<string> = new Set([
  "manual_assertion",
  "sso",
  "hardware_key",
  "signed_statement",
]);

/** The attestation signature members, each a non-blank string. */
const SIGNATURE_MEMBERS = ["alg", "kid", "sig"] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** The ruling members the attestation signature binds beside the principal. */
const RULING_MEMBERS = ["issued_at", "judged_action_refs", "ruling"] as const;

/** Canonical bytes of the attestation minus its own signature envelope,
 * bound to the reviewer's principal and the ruling it backs. */
function attestationMessage(payload: Record<string, unknown>, attestation: Record<string, unknown>): Uint8Array {
  const reviewer = payload.reviewer;
  return asqavJcs({
    attestation: Object.fromEntries(Object.entries(attestation).filter(([k]) => k !== "signature")),
    principal: isRecord(reviewer) ? (reviewer.principal ?? null) : null,
    ...Object.fromEntries(RULING_MEMBERS.map((k) => [k, payload[k] ?? null])),
  });
}

/** The one trusted key bound to the reviewer's principal, never a wire key. */
function reviewerKey(keys: unknown, principal: unknown): Record<string, unknown> | null {
  if (!isRecord(keys) || typeof principal !== "string" || principal.trim() === "") return null;
  const entries = Array.isArray(keys.keys) ? keys.keys : [];
  const bound = entries.filter((entry) => isRecord(entry) && entry.principal === principal);
  return bound.length === 1 ? (bound[0] as Record<string, unknown>) : null;
}

function signatureVerifies(
  payload: Record<string, unknown>,
  attestation: Record<string, unknown>,
  reviewerKey: Record<string, unknown>,
): boolean {
  const signature = attestation.signature;
  if (!isRecord(signature)) return false;
  const complete = SIGNATURE_MEMBERS.every((k) => {
    const value = signature[k];
    return typeof value === "string" && value.trim() !== "";
  });
  if (!complete) return false;
  const alg = reviewerKey.alg;
  if (typeof alg !== "string" || alg.toUpperCase() !== (signature.alg as string).toUpperCase()) return false;
  const publicKey = reviewerKey.public_key;
  if (typeof publicKey !== "string") return false;
  const outcome = verifySignature(
    alg,
    b64decode(publicKey),
    attestationMessage(payload, attestation),
    b64decode(signature.sig as string),
  );
  return outcome.result === PASS;
}

/** Verified only when the attestation signature checks under the trusted
 * key bound to the reviewer; every other listed method is asserted. */
export function oversightAttestation(payload: unknown, keys: unknown = null): string {
  if (!isRecord(payload)) return NOT_APPLICABLE;
  if (payload.type !== "protectmcp:lifecycle:oversight_ruling") return NOT_APPLICABLE;
  const reviewer = payload.reviewer;
  if (!isRecord(reviewer)) return MISSING;
  const attestation = reviewer.attestation;
  if (!isRecord(attestation)) return MISSING;
  const method = attestation.method;
  if (typeof method !== "string" || !METHODS.has(method)) return MISSING;
  const key = reviewerKey(keys, reviewer.principal);
  if (method === "manual_assertion" || key === null) return ASSERTED;
  const status = typeof key.status === "string" ? key.status : null;
  const issuedAt = typeof payload.issued_at === "string" ? payload.issued_at : "";
  if (checkKeyStatus(status, issuedAt, revokedAtOf(key), false)[0] !== PASS) return ASSERTED;
  return signatureVerifies(payload, attestation, key) ? SIGNATURE_VERIFIED : ASSERTED;
}
