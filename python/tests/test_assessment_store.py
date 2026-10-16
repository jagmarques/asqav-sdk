"""Assessment store (488 C10): retain, replay, reassess, detect tampering."""
from __future__ import annotations

import copy

import pytest

from asqav.verifier.oracle.assessment_store import AssessmentEntry, AssessmentStore

_CTX = {"evaluation_time": "2029-06-01T00:00:00Z", "evaluator": "c10-probe"}
_AXES = {"expiry": "PASS", "skew": "PASS"}


def _store() -> AssessmentStore:
    store = AssessmentStore()
    store.record("aa" * 32, _CTX, _AXES, "verified", None, recorded_at="2029-06-01T00:00:00Z")
    return store


def test_replay_returns_the_retained_original() -> None:
    entry = _store().replay("aa" * 32)
    assert entry is not None and entry.kind == "original"
    assert entry.context == _CTX and entry.axes == _AXES and entry.supersedes is None


def test_replay_misses_unknown_bytes() -> None:
    assert _store().replay("bb" * 32) is None


def test_reassessment_labels_and_links_without_revising() -> None:
    store = _store()
    before = store.replay("aa" * 32)
    assert before is not None
    snap = copy.deepcopy(before)
    second = store.reassess(
        "aa" * 32, {**_CTX, "evaluation_time": "2031-01-01T00:00:00Z"},
        {"expiry": "FAIL", "skew": "PASS"}, "unverified", "invalid",
        recorded_at="2031-01-01T00:00:00Z",
    )
    assert second.kind == "reassessment" and second.supersedes == before.entry_hash
    assert store.replay("aa" * 32) == snap, "original must never be revised"
    assert store.verify_chain()


def test_mutation_breaks_the_chain() -> None:
    store = _store()
    store._entries[0] = AssessmentEntry(**{**store._entries[0].__dict__, "verdict": "x"})
    assert not store.verify_chain()


def test_jsonl_round_trip_preserves_and_rejects() -> None:
    store = _store()
    store.reassess("aa" * 32, _CTX, _AXES, "verified", None)
    restored = AssessmentStore.from_jsonl(store.to_jsonl())
    assert len(restored) == 2 and restored.verify_chain()
    assert restored.replay("aa" * 32) is not None
    with pytest.raises(ValueError):
        AssessmentStore.from_jsonl(store.to_jsonl().replace("verified", "x"))


def test_non_ascii_inputs_hash_utf8_and_round_trip() -> None:
    store = AssessmentStore()
    entry = store.record(
        "ee" * 32,
        {"note": "páy ✓", "evaluation_time": "2029-06-01T00:00:00Z"},
        {"expiry": "PASS"}, "verified", None, recorded_at="2029-06-01T00:00:00Z",
    )
    assert entry.entry_hash == "fc0d78385e892d992c8954103a23d58444c51573c4cb615aa68e8fd29b75676b"
    assert store.verify_chain()
    assert AssessmentStore.from_jsonl(store.to_jsonl()).replay("ee" * 32) == entry
