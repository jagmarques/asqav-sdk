/** Oversight-attestation states the platform reports, signature-gated. */

import { asqavJcs } from "./canonical.js";
import { PASS, verifySignature } from "./crypto.js";
import { b64decode } from "./vrShim.js";

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

/** The vector file pinning the reviewer key; the receipt never supplies it. */
export const REVIEWER_KEY_FILE = "reviewer_key.json";
/** The attestation signature members, each a non-blank string. */
const SIGNATURE_MEMBERS = ["alg", "kid", "sig"] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Canonical bytes of the attestation minus its own signature envelope. */
export function attestationMessage(attestation: Record<string, unknown>): Uint8Array {
  return asqavJcs(Object.fromEntries(Object.entries(attestation).filter(([k]) => k !== "signature")));
}

function signatureVerifies(attestation: Record<string, unknown>, reviewerKey: Record<string, unknown>): boolean {
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
    attestationMessage(attestation),
    b64decode(signature.sig as string),
  );
  return outcome.result === PASS;
}

/** Verified only when the attestation signature checks under the pinned
 * reviewer key; every other listed method is asserted. */
export function oversightAttestation(payload: unknown, reviewerKey: unknown = null): string {
  if (!isRecord(payload)) return NOT_APPLICABLE;
  if (payload.type !== "protectmcp:lifecycle:oversight_ruling") return NOT_APPLICABLE;
  const reviewer = payload.reviewer;
  if (!isRecord(reviewer)) return MISSING;
  const attestation = reviewer.attestation;
  if (!isRecord(attestation)) return MISSING;
  const method = attestation.method;
  if (typeof method !== "string" || !METHODS.has(method)) return MISSING;
  if (method === "manual_assertion" || !isRecord(reviewerKey)) return ASSERTED;
  return signatureVerifies(attestation, reviewerKey) ? SIGNATURE_VERIFIED : ASSERTED;
}
