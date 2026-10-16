"""Compare two live VerifyResults, then classify (488 C19, C5 core).

The C12 compare and C13 classify steps never ran over live oracle output.
This adapter maps both results to the retained shape and runs that same
compare-before-classify path. Retention stays optional: digests ride as
caller-stated input bytes, and predicate terms attach as AgreementRef.
"""
from __future__ import annotations

import hashlib

from asqav.verifier.oracle.disagreement import Disagreement, classify
from asqav.verifier.oracle.retain import axes_dict, context_dict


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
    first_view = {
        "doc_sha256": first_digest,
        "axes": axes_dict(first),
        "context": context_dict(first),
        "agreement": first_agreement,
    }
    second_view = {
        "doc_sha256": second_digest,
        "axes": axes_dict(second),
        "context": context_dict(second),
        "agreement": second_agreement,
    }
    return classify(first_view, second_view, getattr(first, "verdict"), getattr(second, "verdict"))
