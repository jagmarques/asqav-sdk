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

/** The receipt type whose reviewer claim these states report. */
export const TYPE = "protectmcp:lifecycle:oversight_ruling";
export const MANUAL_ASSERTION = "manual_assertion";
export const METHODS: ReadonlySet<string> = new Set([
  MANUAL_ASSERTION,
  "sso",
  "hardware_key",
  "signed_statement",
]);
/** The expected.json member a vector declares its oversight state in. */
export const EXPECTED_KEY = "oversight_attestation";

/** Report the reviewer's claim; only a verified signature backs it. */
export function oversightAttestation(payload: unknown, result: VerifyResult): string {
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) return NOT_APPLICABLE;
  const record = payload as Record<string, unknown>;
  if (record.type !== TYPE) return NOT_APPLICABLE;
  const reviewer = record.reviewer;
  if (typeof reviewer !== "object" || reviewer === null || Array.isArray(reviewer)) return MISSING;
  const attestation = (reviewer as Record<string, unknown>).attestation;
  if (typeof attestation !== "object" || attestation === null || Array.isArray(attestation)) return MISSING;
  const method = (attestation as Record<string, unknown>).method;
  if (typeof method !== "string" || !METHODS.has(method)) return MISSING;
  if (method === MANUAL_ASSERTION) return ASSERTED;
  const signature = result.axes.find((axis) => axis.axis === "signature");
  const signatureOk = signature?.result === PASS;
  return signatureOk ? SIGNATURE_VERIFIED : ASSERTED;
}
