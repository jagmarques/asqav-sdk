/**
 * Tri-state failure taxonomy (criterion 418) - one shared vocabulary.
 * A port of the Python oracle's `verifier/oracle/taxonomy.py`; the closed
 * reason-code table is pinned cross-language by `verifier/failure-classification.json`.
 *
 * Every verifier surface reports failures in exactly two classes that must
 * never collapse into one:
 *
 *   - INVALID: a check RAN and a cryptographic or policy binding FAILED
 *     (signature mismatch, chain mismatch, invalid anchor, counterparty
 *     mismatch, key changed, algorithm mismatch, skew-bound violation...).
 *   - UNVERIFIABLE: recomputation COULD NOT COMPLETE (unresolvable key,
 *     missing chain predecessor, malformed or duplicate member, parse or
 *     canonicalisation failure, unsupported algorithm, pending anchor
 *     without proof...).
 *
 * A receipt is NEVER reported valid while any recomputation failed: verdict
 * derivation orders INVALID above UNVERIFIABLE above PASS.
 *
 * `SKIPPED` survives as a fourth AXIS result with no failure class: the axis
 * does not apply to this receipt, so there is nothing to bind and it never
 * blocks PASS.
 */

/** Verdict tokens - the overall outcome of verifying one receipt. */
export const PASS = "PASS";
export const INVALID = "INVALID";
export const UNVERIFIABLE = "UNVERIFIABLE";

/** Fourth axis-only result: the axis does not apply, nothing to bind. */
export const SKIPPED = "SKIPPED";

export type VerdictToken = "PASS" | "INVALID" | "UNVERIFIABLE";
export type AxisToken = VerdictToken | "SKIPPED";
export type Classification = "valid" | "invalid" | "unverifiable";

/** Verdict -> wire classification token; the failure classes stay distinct. */
export const CLASSIFICATION: Record<VerdictToken, Classification> = {
  PASS: "valid",
  INVALID: "invalid",
  UNVERIFIABLE: "unverifiable",
};

/** Reason code carried by PASS and not-applicable SKIPPED axes. */
export const NO_REASON = "none";

/**
 * Reason code -> failure class. Closed set; both language suites pin it
 * against `verifier/failure-classification.json`.
 */
export const REASON_CLASSES: Record<string, "INVALID" | "UNVERIFIABLE"> = {
  // INVALID: a check ran and a cryptographic or policy binding failed
  signature_mismatch: "INVALID",
  chain_mismatch: "INVALID",
  invalid_anchor: "INVALID",
  counterparty_mismatch: "INVALID",
  key_changed: "INVALID",
  algorithm_mismatch: "INVALID",
  skew_bound_violation: "INVALID",
  false_control_attestation: "INVALID",
  duplicate_emission: "INVALID",
  countersign_missing: "INVALID",
  pdp_binding_mismatch: "INVALID",
  expiry_lapsed: "INVALID",
  claim_mismatch: "INVALID",
  inclusion_mismatch: "INVALID",
  policy_binding_violation: "INVALID",
  // UNVERIFIABLE: recomputation could not complete
  parse_failed: "UNVERIFIABLE",
  duplicate_member: "UNVERIFIABLE",
  member_malformed: "UNVERIFIABLE",
  canonicalization_failed: "UNVERIFIABLE",
  format_unrecognized: "UNVERIFIABLE",
  key_unresolvable: "UNVERIFIABLE",
  key_malformed: "UNVERIFIABLE",
  signature_malformed: "UNVERIFIABLE",
  algorithm_unsupported: "UNVERIFIABLE",
  crypto_dependency_missing: "UNVERIFIABLE",
  chain_predecessor_missing: "UNVERIFIABLE",
  issuer_unresolvable: "UNVERIFIABLE",
  anchor_unproven: "UNVERIFIABLE",
  timing_unproven: "UNVERIFIABLE",
  inclusion_unchecked: "UNVERIFIABLE",
  signature_unchecked: "UNVERIFIABLE",
};

/** True when `reasonCode` belongs to `failureClass` in the closed vocabulary. */
export function reasonHasClass(reasonCode: string, failureClass: string): boolean {
  return REASON_CLASSES[reasonCode] === failureClass;
}

/**
 * Fold per-axis results into the receipt verdict without collapsing classes.
 *
 * INVALID dominates UNVERIFIABLE: one proved binding failure outweighs every
 * incomplete check, and an incomplete check can never clear it. Axes named in
 * `exemptAxes` report on their own and never fold the verdict (426). A
 * SKIPPED (not-applicable) axis never blocks; an UNVERIFIABLE one always does.
 */
export function deriveVerdict(
  namedResults: Iterable<readonly [string, string]>,
  exemptAxes: ReadonlySet<string> = new Set(["expiry"]),
): VerdictToken {
  let sawUnverifiable = false;
  for (const [name, result] of namedResults) {
    if (exemptAxes.has(name)) continue;
    if (result === INVALID) return INVALID;
    if (result === UNVERIFIABLE) sawUnverifiable = true;
  }
  return sawUnverifiable ? UNVERIFIABLE : PASS;
}
