"""Compare two live VerifyResults, then classify (488 C19, C5 core).
The C12 compare and C13 classify steps never ran over live oracle output.
This adapter maps both results to the retained shape and runs that same
compare-before-classify path. Retention stays optional: digests ride as
caller-stated input bytes, and predicate terms attach as AgreementRef."""
from __future__ import annotations

import hashlib
import json

from asqav.verifier.oracle import ADAPTERS, verify
from asqav.verifier.oracle.adapter import VerificationContext
from asqav.verifier.oracle.agreement import bind_envelope
from asqav.verifier.oracle.assessment_store import AssessmentStore
from asqav.verifier.oracle.disagreement import (
    Disagreement,
    SpecCase,
    classify,
    mark_legacy_gap,
    spec_case,
)
from asqav.verifier.oracle.retain import axes_dict, context_dict


def _retained_view(entry: object) -> dict:
    """Map a retained assessment entry onto the assessment view shape."""
    return {
        "doc_sha256": getattr(entry, "doc_sha256"),
        "axes": dict(getattr(entry, "axes")),
        "context": dict(getattr(entry, "context")),
        # Stored envelope ref (488 C39a); old entries predate it.
        "agreement": getattr(entry, "agreement", None),
    }


def _live_view(digest: str, result: object, agreement: object | None) -> dict:
    """Map one live result onto the retained-assessment view shape."""
    return {
        "doc_sha256": digest,
        "axes": axes_dict(result),
        "context": context_dict(result),
        "agreement": agreement,
    }


def _has_stated_envelope(agreement: object | None) -> bool:
    """True when the caller states a kid or signature (needs bytes)."""
    if agreement is None:
        return False
    if isinstance(agreement, dict):
        kid = agreement.get("kid")
        sig = agreement.get("signature")
    else:
        kid = getattr(agreement, "kid", None)
        sig = getattr(agreement, "signature", None)
    return kid is not None or sig is not None


def _bound_agreement(agreement: object | None, raw: bytes | None) -> object | None:
    """Bind the caller terms to the bytes-derived envelope (488 C30).

    Bytes present with stated terms: parse and bind (contradiction
    raises, match derives). No stated agreement: keep None so
    digest-only callers see no behavior change. Bytes absent: pass
    through (a stated envelope already raised above)."""
    if raw is None or agreement is None:
        return agreement
    return bind_envelope(agreement, json.loads(raw), ADAPTERS)


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
    sha256 raises instead of classifying). Envelopes default-bind the
    same way (C30): a stated kid/signature without bytes raises, and
    bytes present bind the terms to the derived envelope."""
    for tag, digest, raw, agr in (("first", first_digest, first_bytes, first_agreement),
                                  ("second", second_digest, second_bytes, second_agreement)):
        if raw is not None and hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"{tag} digest is not the sha256 of the stated bytes")
        if raw is None and _has_stated_envelope(agr):
            raise ValueError(f"{tag} envelope without bytes; pass the receipt bytes")
    first_view = _live_view(first_digest, first,
                            _bound_agreement(first_agreement, first_bytes))
    second_view = _live_view(second_digest, second,
                             _bound_agreement(second_agreement, second_bytes))
    return classify(first_view, second_view, getattr(first, "verdict"), getattr(second, "verdict"))


def compare_live_strict(
    first_bytes: bytes,
    second_bytes: bytes,
    first_agreement: object | None = None,
    second_agreement: object | None = None,
    *,
    context: VerificationContext | None = None,
    first_context: VerificationContext | None = None,
    second_context: VerificationContext | None = None,
) -> Disagreement:
    """Strict live compare: the entry verifies the bytes itself (488 C26, C29).
    No digest or result parameter exists: no unbound or foreign-result call is expressible; digests
    derive from the bytes; both results come from re-verifying them under one context. Envelopes derive
    from the same bytes (C29): caller terms attach, but a contradicting kid/signature raises. Per-side
    trust (C33): first_context/second_context verify each side under its own context, so differing
    trust dims classify. Omitted sides fall back to ``context``; single-context calls unchanged."""
    first_doc, second_doc = json.loads(first_bytes), json.loads(second_bytes)
    first = verify(first_doc, ADAPTERS,
                   context=first_context if first_context is not None else context)
    second = verify(second_doc, ADAPTERS,
                    context=second_context if second_context is not None else context)
    first_digest = hashlib.sha256(first_bytes).hexdigest()
    second_digest = hashlib.sha256(second_bytes).hexdigest()
    first_view = _live_view(
        first_digest, first, bind_envelope(first_agreement, first_doc, ADAPTERS))
    second_view = _live_view(
        second_digest, second, bind_envelope(second_agreement, second_doc, ADAPTERS))
    return classify(first_view, second_view, first.verdict, second.verdict)


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
    originals. Envelopes default-bind like compare_live (C31): a stated
    kid/signature without its bytes raises, bytes present bind."""
    for tag, digest, raw, agr, present in (
            ("first", first_digest, first_bytes, first_agreement, True),
            ("second", second_digest, second_bytes, second_agreement, True),
            ("rerun_a", first_digest, rerun_a_bytes, first_agreement, rerun_a is not None),
            ("rerun_b", second_digest, rerun_b_bytes, second_agreement, rerun_b is not None)):
        if raw is not None and hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"{tag} digest is not the sha256 of the stated bytes")
        if present and raw is None and _has_stated_envelope(agr):
            raise ValueError(f"{tag} envelope without bytes; pass the receipt bytes")
    first_view = _live_view(first_digest, first,
                            _bound_agreement(first_agreement, first_bytes))
    second_view = _live_view(second_digest, second,
                             _bound_agreement(second_agreement, second_bytes))
    rerun_a_view = (None if rerun_a is None
                    else _live_view(first_digest, rerun_a,
                                    _bound_agreement(first_agreement, rerun_a_bytes)))
    rerun_b_view = (None if rerun_b is None
                    else _live_view(second_digest, rerun_b,
                                    _bound_agreement(second_agreement, rerun_b_bytes)))
    return attribute_split(
        first_view, second_view, getattr(first, "verdict"), getattr(second, "verdict"),
        rerun_a_view, rerun_b_view)


def classify_and_attribute(
    store: AssessmentStore,
    doc_sha256: str,
    result: object,
    rerun_a: object | None = None,
    rerun_b: object | None = None,
    agreement: object | None = None,
) -> tuple[Disagreement, SpecCase]:
    """Review a live reassessment against its retained original (488 C27).
    The production caller: the original comes from the store (missing retention raises, not classifies),
    the live result and reruns map to views; attribute_split classifies and runs the spec-bug bar. Pure
    review: the store is never mutated. The live agreement is the caller-derived envelope (488 C39b:
    from bytes, never stored); reruns share it. A ref-less original predates ref retention (same digest,
    same derivability), so the pair compares ref-less and names the gap; modern pairs compare ref to ref."""
    original = store.replay(doc_sha256)
    if original is None:
        raise ValueError("no retained original for this digest; record first")
    original_ref = getattr(original, "agreement", None)
    live_ref = agreement if original_ref is not None else None
    first_view = _retained_view(original)
    second_view = _live_view(doc_sha256, result, live_ref)
    rerun_a_view = None if rerun_a is None else _live_view(doc_sha256, rerun_a, live_ref)
    rerun_b_view = None if rerun_b is None else _live_view(doc_sha256, rerun_b, live_ref)
    disagreement, case = attribute_split(
        first_view, second_view, original.verdict, getattr(result, "verdict"),
        rerun_a_view, rerun_b_view)
    return mark_legacy_gap(disagreement, original_ref, agreement), case
