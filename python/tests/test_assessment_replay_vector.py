"""Identical-facts replay vector (488 C11): same bytes, two policy eras.

Holds event bytes fixed while varying declared policy revision,
evaluation time and evidence-knowledge cutoff. The original assessment
reproduces from retained inputs; the later reassessment is labelled
separately and never rewrites history."""
from __future__ import annotations

from asqav.verifier.oracle.assessment_store import AssessmentStore

_DOC = "cc" * 32
_T1 = "2029-06-01T00:00:00Z"
_T2 = "2031-01-01T00:00:00Z"
_CTX_T1 = {
    "policy_revision": "2029-01",
    "event_time": _T1,
    "evaluation_time": _T1,
    "evidence_knowledge_cutoff": _T1,
}
_CTX_T2 = {
    "policy_revision": "2031-01",
    "event_time": _T1,
    "evaluation_time": _T2,
    "evidence_knowledge_cutoff": _T2,
}
_AXES_OK = {"expiry": "PASS", "skew": "PASS"}
_AXES_BAD = {"expiry": "FAIL", "skew": "PASS"}


def _two_eras() -> AssessmentStore:
    store = AssessmentStore()
    store.record(_DOC, _CTX_T1, _AXES_OK, "verified", None, recorded_at=_T1)
    store.reassess(_DOC, _CTX_T2, _AXES_BAD, "unverified", "invalid", recorded_at=_T2)
    return store


def test_replay_returns_contemporaneous_inputs_not_current_policy() -> None:
    replayed = _two_eras().replay(_DOC)
    assert replayed is not None and replayed.kind == "original"
    assert replayed.context == _CTX_T1
    assert replayed.axes == _AXES_OK and replayed.verdict == "verified"


def test_later_reassessment_is_labelled_and_linked() -> None:
    store = _two_eras()
    original = store.replay(_DOC)
    assert original is not None
    assert len(store._entries) == 2
    second = store._entries[1]
    assert second.kind == "reassessment" and second.supersedes == original.entry_hash
    assert second.context["policy_revision"] == "2031-01"
    assert store.verify_chain()
