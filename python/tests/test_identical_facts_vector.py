"""Identical-facts vector (488 C9): same bytes under another policy
revision or cutoff attribute to inputs, never unexplained."""
from __future__ import annotations

from asqav.verifier.oracle.agreement import COMPARANDA, compare_assessments
from asqav.verifier.oracle.disagreement import attributed_to_trust_time, classify

_CTX = {
    "evaluator": "c9-probe",
    "trust_roots": ["root-a"],
    "revocation_snapshot": "snap-1",
    "evaluation_time": "2029-06-01T00:00:00Z",
    "policy_revision": "pol-7",
    "evidence_knowledge_cutoff": "cut-3",
}
_AXES = {"expiry": "PASS", "skew": "PASS"}


def _view(**over: object) -> dict:
    base = {"doc_sha256": "cc" * 32, "axes": dict(_AXES), "context": dict(_CTX)}
    for key, value in over.items():
        if key == "context":
            base["context"] = {**_CTX, **value}  # type: ignore[dict-item]
        else:
            base[key] = value
    return base


def test_identical_views_match_on_all_nine() -> None:
    assert compare_assessments(_view(), _view()) == {name: True for name in COMPARANDA}


def test_policy_revision_flip_attributes_never_unexplained() -> None:
    other = _view(context={"policy_revision": "pol-9"})
    assert compare_assessments(_view(), other)["policy_revision"] is False
    out = classify(_view(), other, "verified", "unverified")
    assert out.label == "attributed_to_inputs"
    assert out.differing == ("policy_revision",)
    assert not attributed_to_trust_time(out), "a ruleset change is not trust/time drift"


def test_cutoff_flip_attributes_with_same_verdict() -> None:
    other = _view(context={"evidence_knowledge_cutoff": "cut-4"})
    out = classify(_view(), other, "verified", "verified")
    assert out.label == "agreement"
    assert out.differing == ("evidence_knowledge_cutoff",)
    assert attributed_to_trust_time(out), "a cutoff move is a time horizon move"


def test_absent_fields_read_equal_for_old_entries() -> None:
    bare = {"doc_sha256": "cc" * 32, "axes": dict(_AXES), "context": {"evaluator": "c9-probe"}}
    assert compare_assessments(bare, dict(bare))["policy_revision"] is True
    assert compare_assessments(bare, dict(bare))["evidence_knowledge_cutoff"] is True
    out = classify(bare, dict(bare), "verified", "verified")
    assert out.label == "agreement" and out.differing == ()
