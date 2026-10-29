/** Oversight-attestation states the platform reports, signature-gated. */

import type { VerifyResult } from "./core.js";
import { PASS } from "./crypto.js";

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

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Report the reviewer's claim; only a verified signature backs it. */
export function oversightAttestation(payload: unknown, result: VerifyResult): string {
  if (!isRecord(payload)) return NOT_APPLICABLE;
  if (payload.type !== "protectmcp:lifecycle:oversight_ruling") return NOT_APPLICABLE;
  const reviewer = payload.reviewer;
  if (!isRecord(reviewer)) return MISSING;
  const attestation = reviewer.attestation;
  if (!isRecord(attestation)) return MISSING;
  const method = attestation.method;
  if (typeof method !== "string" || !METHODS.has(method)) return MISSING;
  if (method === "manual_assertion") return ASSERTED;
  const signature = result.axes.find((axis) => axis.axis === "signature");
  const signatureOk = signature?.result === PASS;
  return signatureOk ? SIGNATURE_VERIFIED : ASSERTED;
}
