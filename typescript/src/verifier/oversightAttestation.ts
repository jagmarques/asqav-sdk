/** Oversight-attestation states the platform reports, signature-gated. */

export const NOT_APPLICABLE = "not_applicable";
export const MISSING = "missing/unverified";
export const ASSERTED = "asserted/unverified";
export const SIGNATURE_VERIFIED = "signature-verified";

export const METHODS: ReadonlySet<string> = new Set([
  "manual_assertion",
  "sso",
  "hardware_key",
  "signed_statement",
]);

const MANUAL_ASSERTION = "manual_assertion";

/** Report the reviewer's claim; only a verified signature backs it. */
export function oversightAttestation(payload: unknown, signatureOk: boolean): string {
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) return NOT_APPLICABLE;
  const record = payload as Record<string, unknown>;
  if (record.type !== "protectmcp:lifecycle:oversight_ruling") return NOT_APPLICABLE;
  const reviewer = record.reviewer;
  if (typeof reviewer !== "object" || reviewer === null || Array.isArray(reviewer)) return MISSING;
  const attestation = (reviewer as Record<string, unknown>).attestation;
  if (typeof attestation !== "object" || attestation === null || Array.isArray(attestation)) return MISSING;
  const method = (attestation as Record<string, unknown>).method;
  if (typeof method !== "string" || !METHODS.has(method)) return MISSING;
  if (method === MANUAL_ASSERTION) return ASSERTED;
  return signatureOk ? SIGNATURE_VERIFIED : ASSERTED;
}
