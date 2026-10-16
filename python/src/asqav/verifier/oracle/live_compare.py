"""Compare two live VerifyResults, then classify (488 C19, C5 core).

The C12 compare and C13 classify steps never ran over live oracle output.
This adapter maps both results to the retained shape and runs that same
compare-before-classify path. Retention stays optional: digests ride as
caller-stated input bytes, and predicate terms attach as AgreementRef.
"""
from __future__ import annotations

import hashlib

from asqav.verifier.oracle.disagreement import Disagreement, SpecCase, classify, spec_case
from asqav.verifier.oracle.retain import axes_dict, context_dict


def _live_view(digest: str, result: object, agreement: object | None) -> dict:
    """Map one live result onto the retained-assessment view shape."""
    return {
        "doc_sha256": digest,
        "axes": axes_dict(result),
        "context": context_dict(result),
        "agreement": agreement,
    }


def compare_live(
    first_digest: str,
    first: object,
    second_digest: str,
    second: object,
    first_agreement: object | None = None,
    second_agreement: object | None = None,
    *,
    first_bytes: bytes | None = None,
    second_bytes: bytes | None = None,
) -> Disagreement:
    """Map two live results to views, then compare and classify.

    Digests ride as caller-stated input bytes by default; pass the raw
    receipt bytes to bind them (a stated digest that is not the bytes'
    sha256 raises instead of classifying)."""
    for tag, digest, raw in (("first", first_digest, first_bytes),
                             ("second", second_digest, second_bytes)):
        if raw is not None and hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"{tag} digest is not the sha256 of the stated bytes")
    first_view = _live_view(first_digest, first, first_agreement)
    second_view = _live_view(second_digest, second, second_agreement)
    return classify(first_view, second_view, getattr(first, "verdict"), getattr(second, "verdict"))


def attribute_split(
    first_view: dict,
    second_view: dict,
    verdict_a: str,
    verdict_b: str,
    rerun_a_view: dict | None = None,
    rerun_b_view: dict | None = None,
) -> tuple[Disagreement, SpecCase]:
    """Classify retained views, then run the spec-bug bar (488 C25, C2 wiring).

    The pipeline over retained evidence: compare-before-classify, then
    attribute any unexplained split from the rerun views. Live results
    enter through attribute_live_split; stored entries already are views."""
    return (
        classify(first_view, second_view, verdict_a, verdict_b),
        spec_case(first_view, second_view, verdict_a, verdict_b, rerun_a_view, rerun_b_view),
    )


def attribute_live_split(
    first_digest: str,
    first: object,
    second_digest: str,
    second: object,
    rerun_a: object | None = None,
    rerun_b: object | None = None,
    first_agreement: object | None = None,
    second_agreement: object | None = None,
    *,
    first_bytes: bytes | None = None,
    second_bytes: bytes | None = None,
    rerun_a_bytes: bytes | None = None,
    rerun_b_bytes: bytes | None = None,
) -> tuple[Disagreement, SpecCase]:
    """Map live results (plus live reruns) to views, then attribute (488 C25).

    Rerun digests are the original digests (a rerun re-executes the same
    bytes); pass explicit rerun bytes to bind them like C24 binds the
    originals."""
    for tag, digest, raw in (("first", first_digest, first_bytes),
                             ("second", second_digest, second_bytes),
                             ("rerun_a", first_digest, rerun_a_bytes),
                             ("rerun_b", second_digest, rerun_b_bytes)):
        if raw is not None and hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"{tag} digest is not the sha256 of the stated bytes")
    first_view = _live_view(first_digest, first, first_agreement)
    second_view = _live_view(second_digest, second, second_agreement)
    rerun_a_view = None if rerun_a is None else _live_view(first_digest, rerun_a, first_agreement)
    rerun_b_view = None if rerun_b is None else _live_view(second_digest, rerun_b, second_agreement)
    return attribute_split(
        first_view, second_view, getattr(first, "verdict"), getattr(second, "verdict"),
        rerun_a_view, rerun_b_view)
