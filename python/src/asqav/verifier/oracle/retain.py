"""Retain live oracle assessments (488 C16): VerifyResult -> store.

Maps a real VerifyResult onto the retained-assessment shape so two live
runs can be compared and classified. Additive only: verify() itself is
untouched; the caller decides what to retain. The store stays unbounded;
lifecycle belongs to the caller (noted, not solved, here)."""
from __future__ import annotations

from datetime import datetime

from asqav.verifier.oracle.assessment_store import AssessmentEntry, AssessmentStore
from asqav.verifier.oracle.disagreement import (
    Disagreement,
    SpecCase,
    classify,
    mark_legacy_gap,
    spec_case,
)

_CONTEXT_FIELDS = (
    "evaluator",
    "trust_roots",
    "revocation_snapshot",
    "policy_revision",
    "evidence_knowledge_cutoff",
    "evaluation_time",
)


def _freeze(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_freeze(v) for v in value]
    if isinstance(value, dict):
        return {k: _freeze(v) for k, v in value.items()}
    return value


def context_dict(result: object) -> dict:
    """Attribution-only context of a VerifyResult as plain data."""
    ctx = getattr(result, "context", None)
    if ctx is None:
        return {}
    return {name: _freeze(getattr(ctx, name, None)) for name in _CONTEXT_FIELDS}


def axes_dict(result: object) -> dict:
    """Axis name -> PASS/FAIL/SKIPPED token for a VerifyResult."""
    return {axis.axis: axis.result for axis in getattr(result, "axes", [])}


def record_verify_result(
    store: AssessmentStore,
    doc_sha256: str,
    result: object,
    recorded_at: str | None = None,
    agreement: dict | None = None,
) -> AssessmentEntry:
    """Retain one live assessment; returns the stored entry."""
    return store.record(
        doc_sha256,
        context_dict(result),
        axes_dict(result),
        getattr(result, "verdict"),
        getattr(result, "failure_class"),
        recorded_at=recorded_at,
        agreement=agreement,
    )


def reassess_verify_result(
    store: AssessmentStore,
    doc_sha256: str,
    result: object,
    recorded_at: str | None = None,
    agreement: dict | None = None,
    reason: str | None = None,
) -> AssessmentEntry:
    """Retain a later live run of already-retained bytes as a reassessment."""
    return store.reassess(
        doc_sha256,
        context_dict(result),
        axes_dict(result),
        getattr(result, "verdict"),
        getattr(result, "failure_class"),
        recorded_at=recorded_at,
        agreement=agreement,
        reason=reason,
    )


def reassess_and_classify(
    store: AssessmentStore,
    doc_sha256: str,
    result: object,
    recorded_at: str | None = None,
    agreement: dict | None = None,
    reason: str | None = None,
) -> tuple[AssessmentEntry, Disagreement, SpecCase]:
    """Reassess retained bytes, then classify against the original (488 C38).

    The retention-layer classification caller: reassess raises unless
    a retained original exists, so both comparanda are real. Views
    carry the stored envelope refs (488 C39b: original vs new entry),
    exactly like classify_and_attribute without reruns. A ref-less
    original compares ref-less with the gap named, per the rule in
    live_compare.classify_and_attribute. The reason names the driver
    (e.g. key-compromised) so the revision is labelled, never silent."""
    entry = reassess_verify_result(store, doc_sha256, result, recorded_at, agreement,
                                     reason=reason)
    original = store.replay(doc_sha256)
    if original is None:
        raise ValueError("no retained original for this digest; record first")
    verdict = getattr(result, "verdict")
    original_ref = getattr(original, "agreement", None)
    entry_ref = entry.agreement if original_ref is not None else None
    first = {"doc_sha256": original.doc_sha256, "axes": dict(original.axes),
             "context": dict(original.context),
             "agreement": getattr(original, "agreement", None)}
    second = {"doc_sha256": doc_sha256, "axes": axes_dict(result),
              "context": context_dict(result), "agreement": entry_ref}
    disagreement = classify(first, second, original.verdict, verdict)
    return (entry, mark_legacy_gap(disagreement, original_ref, entry.agreement),
            spec_case(first, second, original.verdict, verdict))
