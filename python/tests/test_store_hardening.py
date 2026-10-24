"""Store hardening (488 C18): duplicate guard, orphan raise, link validation.

Rejected: a second original for the same bytes, a reassessment without a
retained original, and a restored log linking a missing or foreign original,
whose original supersedes, or whose kind is unknown.
"""
from __future__ import annotations

import json

from dataclasses import asdict

import pytest

from asqav.verifier.oracle.assessment_store import AssessmentEntry, AssessmentStore

_CTX = {"evaluation_time": "2029-06-01T00:00:00Z", "evaluator": "c18-probe"}
_AXES = {"expiry": "PASS", "skew": "PASS"}
_SHA = "cc" * 32


def _recorded() -> AssessmentStore:
    store = AssessmentStore()
    store.record(_SHA, _CTX, _AXES, "verified", None)
    return store


def test_duplicate_original_raises_and_keeps_first() -> None:
    store = _recorded()
    with pytest.raises(ValueError):
        store.record(_SHA, _CTX, {"expiry": "FAIL"}, "unverified", "invalid")
    assert len(store) == 1 and store.replay(_SHA) is not None
    assert store.replay(_SHA).axes == _AXES and store.verify_chain()


def test_orphan_reassessment_raises_and_appends_nothing() -> None:
    store = AssessmentStore()
    with pytest.raises(ValueError):
        store.reassess(_SHA, _CTX, _AXES, "verified", None)
    assert len(store) == 0 and store.replay(_SHA) is None


def _relinked(store: AssessmentStore, index: int, **over: object) -> str:
    raw = json.loads(store.to_jsonl().splitlines()[index])
    raw.update(over)
    fixed = AssessmentEntry(**{k: v for k, v in raw.items() if k != "entry_hash"})
    return json.dumps({**raw, "entry_hash": fixed.hashed().entry_hash}, sort_keys=True)


def test_restored_log_rejects_dangling_and_foreign_links() -> None:
    store = _recorded()
    original = store.replay(_SHA)
    assert original is not None
    good = store.reassess(_SHA, _CTX, {"expiry": "FAIL"}, "unverified", "invalid")
    lines = store.to_jsonl().splitlines()
    dangling = [lines[0], _relinked(store, 1, supersedes="ff" * 32)]
    with pytest.raises(ValueError):
        AssessmentStore.from_jsonl("\n".join(dangling))
    foreign = AssessmentEntry(
        doc_sha256="dd" * 32, context=dict(_CTX), axes=dict(_AXES),
        verdict="unverified", failure_class="invalid", kind="reassessment",
        supersedes=original.entry_hash, prev_hash=good.entry_hash,
    ).hashed()
    blob = "\n".join(lines + [json.dumps(asdict(foreign), sort_keys=True)])
    with pytest.raises(ValueError):
        AssessmentStore.from_jsonl(blob)


def test_restored_log_rejects_bad_kind_and_superseding_original() -> None:
    store = _recorded()
    lines = store.to_jsonl().splitlines()
    with pytest.raises(ValueError):
        AssessmentStore.from_jsonl(_relinked(store, 0, kind="snapshot"))
    with pytest.raises(ValueError):
        AssessmentStore.from_jsonl(_relinked(store, 0, supersedes="ff" * 32))
    restored = AssessmentStore.from_jsonl(store.to_jsonl())
    assert len(restored) == 1 and restored.verify_chain()
