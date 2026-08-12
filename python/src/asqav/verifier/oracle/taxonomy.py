"""Tri-state failure taxonomy (criterion 418) - one shared vocabulary.

Every verifier surface in the SDK reports failures in exactly two classes that
must never collapse into one:

  - INVALID: a check RAN and a cryptographic or policy binding FAILED. The bytes
    were read, the recomputation completed, and it proved the receipt wrong:
    a signature that does not match, a chain link that does not re-derive, an
    anchor that does not bind, a counterparty the key does not belong to, a key
    that was changed or revoked, an algorithm that conflicts, a clock skew past
    the bound.

  - UNVERIFIABLE: recomputation COULD NOT COMPLETE, so no binding was proved in
    either direction: an unresolvable or malformed key, a missing chain
    predecessor, a malformed or duplicate JSON member, a canonicalisation or
    parse failure, an algorithm with no implementation, an anchor declared
    without its proof.

A receipt is NEVER reported valid while any recomputation failed: the verdict
derivation orders INVALID above UNVERIFIABLE above PASS, so an incomplete check
downgrades the verdict and can never be hidden by the axes that did pass.

``SKIPPED`` survives as a fourth AXIS result with no failure class at all: the
axis does not apply to this receipt (no anchors declared, no nonce, genesis
chain, no impact tags), so there is nothing to bind and it never blocks PASS.

The reason codes are a CLOSED vocabulary pinned cross-language by
``verifier/failure-classification.json``; both suites assert every code's class.
"""
from __future__ import annotations

#: Verdict tokens - the overall outcome of verifying one receipt.
PASS = "PASS"
INVALID = "INVALID"
UNVERIFIABLE = "UNVERIFIABLE"

#: Fourth axis-only result: the axis does not apply, nothing to bind, never blocks.
SKIPPED = "SKIPPED"

#: Verdict -> wire classification token; the two failure classes stay distinct.
CLASSIFICATION = {PASS: "valid", INVALID: "invalid", UNVERIFIABLE: "unverifiable"}

#: Reason code -> failure class. Closed set; tests pin it cross-language.
REASON_CLASSES: dict[str, str] = {
    # INVALID: a check ran and a cryptographic or policy binding failed
    "signature_mismatch": INVALID,
    "chain_mismatch": INVALID,
    "invalid_anchor": INVALID,
    "counterparty_mismatch": INVALID,
    "key_changed": INVALID,
    "algorithm_mismatch": INVALID,
    "skew_bound_violation": INVALID,
    "false_control_attestation": INVALID,
    "duplicate_emission": INVALID,
    "countersign_missing": INVALID,
    "pdp_binding_mismatch": INVALID,
    "expiry_lapsed": INVALID,
    "claim_mismatch": INVALID,
    "inclusion_mismatch": INVALID,
    "policy_binding_violation": INVALID,
    # UNVERIFIABLE: recomputation could not complete
    "parse_failed": UNVERIFIABLE,
    "duplicate_member": UNVERIFIABLE,
    "member_malformed": UNVERIFIABLE,
    "canonicalization_failed": UNVERIFIABLE,
    "format_unrecognized": UNVERIFIABLE,
    "key_unresolvable": UNVERIFIABLE,
    "key_malformed": UNVERIFIABLE,
    "signature_malformed": UNVERIFIABLE,
    "algorithm_unsupported": UNVERIFIABLE,
    "crypto_dependency_missing": UNVERIFIABLE,
    "chain_predecessor_missing": UNVERIFIABLE,
    "issuer_unresolvable": UNVERIFIABLE,
    "anchor_unproven": UNVERIFIABLE,
    "timing_unproven": UNVERIFIABLE,
    "inclusion_unchecked": UNVERIFIABLE,
    "signature_unchecked": UNVERIFIABLE,
}

#: Reason code carried by PASS and not-applicable SKIPPED axes.
NO_REASON = "none"


    # True when ``reason_code`` belongs to ``failure_class`` in the closed vocabulary.
def reason_has_class(reason_code: str, failure_class: str) -> bool:
    return REASON_CLASSES.get(reason_code) == failure_class


def derive_verdict(axis_results, exempt_axes=("expiry",)) -> str:
    """Fold per-axis results into the receipt verdict without collapsing classes.

    INVALID dominates UNVERIFIABLE: one proved binding failure outweighs every
    incomplete check, and an incomplete check can never clear it. Axes named in
    ``exempt_axes`` report on their own and never fold the verdict (426). A
    SKIPPED (not-applicable) axis never blocks; an UNVERIFIABLE one always does,
    so a receipt is never PASS while any recomputation failed.
    """
    weighed = [r for name, r in axis_results if name not in exempt_axes]
    if any(r == INVALID for r in weighed):
        return INVALID
    if any(r == UNVERIFIABLE for r in weighed):
        return UNVERIFIABLE
    return PASS
