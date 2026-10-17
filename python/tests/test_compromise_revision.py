"""Compromise-driven confidence revision (488 C10): reason-labelled
reassessment, original reproducible, readers served current, never frozen."""
from __future__ import annotations

import copy

import pytest

from asqav.verifier.oracle.assessment_store import AssessmentStore

_ORIG_CTX = {
    "evaluation_time": "2029-06-01T00:00:00Z",
    "evaluator": "c10-probe",
    "revocation_snapshot": "snap-1",
    "policy_revision": "pol-7",
}
_ORIG_AXES = {"signature": "PASS", "key_status": "PASS"}
_NEW_CTX = {
    "evaluation_time": "2031-01-01T00:00:00Z",
    "evaluator": "c10-probe",
    "revocation_snapshot": "snap-2-key-revoked",
    "policy_revision": "pol-9",
}
_NEW_AXES = {"signature": "PASS", "key_status": "FAIL"}


def _store() -> AssessmentStore:
    store = AssessmentStore()
    store.record("cc" * 32, _ORIG_CTX, _ORIG_AXES, "verified", None,
                 recorded_at="2029-06-01T00:00:00Z")
    return store


def test_current_serves_reassessment_never_frozen_original() -> None:
    store = _store()
    assert store.current("cc" * 32) is not None
    assert store.current("cc" * 32).kind == "original"
    snap = copy.deepcopy(store.replay("cc" * 32))
    second = store.reassess("cc" * 32, _NEW_CTX, _NEW_AXES, "unverified", "invalid",
                            recorded_at="2031-01-01T00:00:00Z",
                            reason="key-compromised")
    assert second.kind == "reassessment" and second.reason == "key-compromised"
    assert store.replay("cc" * 32) == snap, "original must never be revised"
    assert store.current("cc" * 32) == second, "readers take current"
    assert store.verify_chain()


def test_reassessment_records_applied_inputs() -> None:
    entry = _store().reassess("cc" * 32, _NEW_CTX, _NEW_AXES, "unverified",
                              "invalid", reason="key-compromised")
    assert entry.context["revocation_snapshot"] == "snap-2-key-revoked"
    assert entry.context["policy_revision"] == "pol-9"
    assert entry.context != _ORIG_CTX, "applied inputs ride the entry"


def test_reason_round_trips_and_old_logs_load() -> None:
    store = _store()
    store.reassess("cc" * 32, _NEW_CTX, _NEW_AXES, "unverified", "invalid",
                   reason="policy-updated")
    restored = AssessmentStore.from_jsonl(store.to_jsonl())
    assert restored.current("cc" * 32) is not None
    assert restored.current("cc" * 32).reason == "policy-updated"
    assert restored.verify_chain()
    legacy = store.to_jsonl().replace(', "reason": "policy-updated"', "")
    assert AssessmentStore.from_jsonl(legacy).verify_chain()


def test_current_misses_unknown_bytes() -> None:
    assert _store().current("dd" * 32) is None


def test_reassessment_without_reason_stays_unlabelled() -> None:
    entry = _store().reassess("cc" * 32, _NEW_CTX, _NEW_AXES, "unverified", "invalid")
    assert entry.reason is None and entry.kind == "reassessment"


def test_reassess_without_original_raises() -> None:
    with pytest.raises(ValueError):
        AssessmentStore().reassess("cc" * 32, _NEW_CTX, _NEW_AXES, "unverified",
                                   "invalid", reason="key-compromised")
