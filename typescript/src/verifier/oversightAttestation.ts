/** Oversight-attestation states the platform reports, signature-gated. */

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
  if (method === "manual_assertion") return ASSERTED;
  return signatureOk ? SIGNATURE_VERIFIED : ASSERTED;
}
