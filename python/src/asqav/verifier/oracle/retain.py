"""Retain live oracle assessments (488 C16): VerifyResult -> store.

Maps a real VerifyResult onto the retained-assessment shape so two live
runs can be compared and classified. Additive only: verify() itself is
untouched; the caller decides what to retain. The store stays unbounded;
lifecycle belongs to the caller (noted, not solved, here)."""
from __future__ import annotations

from datetime import datetime

from asqav.verifier.oracle.assessment_store import AssessmentEntry, AssessmentStore

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
) -> AssessmentEntry:
    """Retain one live assessment; returns the stored entry."""
    return store.record(
        doc_sha256,
        context_dict(result),
        axes_dict(result),
        getattr(result, "verdict"),
        getattr(result, "failure_class"),
        recorded_at=recorded_at,
    )
