"""Event/effective-time vector (488 C9): same bytes under another declared
event or effective time attribute to inputs, never unexplained."""
from __future__ import annotations

from asqav.verifier.oracle.agreement import COMPARANDA, compare_assessments
from asqav.verifier.oracle.assessment_store import AssessmentStore
from asqav.verifier.oracle.disagreement import attributed_to_trust_time, classify

_CTX = {
    "evaluator": "c9t-probe",
    "trust_roots": ["root-a"],
    "revocation_snapshot": "snap-1",
    "evaluation_time": "2029-06-01T00:00:00Z",
    "event_time": "2029-05-01T00:00:00Z",
    "effective_time": "2029-05-02T00:00:00Z",
}
_AXES = {"expiry": "PASS", "skew": "PASS"}


def _view(**over: object) -> dict:
    base = {"doc_sha256": "dd" * 32, "axes": dict(_AXES), "context": dict(_CTX)}
    for key, value in over.items():
        if key == "context":
            base["context"] = {**_CTX, **value}  # type: ignore[dict-item]
        else:
            base[key] = value
    return base


def test_identical_views_match_on_all_comparanda() -> None:
    assert compare_assessments(_view(), _view()) == {name: True for name in COMPARANDA}


def test_event_time_flip_attributes_never_unexplained() -> None:
    other = _view(context={"event_time": "2029-07-01T00:00:00Z"})
    assert compare_assessments(_view(), other)["event_time"] is False
    out = classify(_view(), other, "verified", "unverified")
    assert out.label == "attributed_to_inputs"
    assert out.differing == ("event_time",)
    assert not attributed_to_trust_time(out)


def test_effective_time_flip_attributes_never_unexplained() -> None:
    other = _view(context={"effective_time": "2029-05-03T00:00:00Z"})
    assert compare_assessments(_view(), other)["effective_time"] is False
    out = classify(_view(), other, "verified", "verified")
    assert out.label == "agreement"
    assert out.differing == ("effective_time",)
    assert not attributed_to_trust_time(out)


def test_absent_fields_read_equal_for_old_entries() -> None:
    bare = {"doc_sha256": "dd" * 32, "axes": dict(_AXES), "context": {"evaluator": "c9t-probe"}}
    assert compare_assessments(bare, dict(bare))["event_time"] is True
    assert compare_assessments(bare, dict(bare))["effective_time"] is True


def test_dual_spelled_new_fields_read_camel_first() -> None:
    both = {"event_time": "2029-05-01T00:00:00Z", "eventTime": "2029-05-01T00:00:00Z",
            "effective_time": "2029-05-02T00:00:00Z",
            "effectiveTime": "2029-05-02T00:00:00Z"}
    camel_flip = {**both, "eventTime": "2029-06-01T00:00:00Z"}
    assert compare_assessments(_view(context=both), _view(context=camel_flip))[
        "event_time"] is False
    snake_flip = {**both, "effective_time": "2029-06-02T00:00:00Z"}
    assert compare_assessments(_view(context=both), _view(context=snake_flip))[
        "effective_time"] is True
    out = classify(_view(context=both), _view(context=camel_flip), "verified", "unverified")
    assert out.label == "attributed_to_inputs" and out.differing == ("event_time",)


def test_replay_reproduces_the_retained_contemporaneous_inputs() -> None:
    store = AssessmentStore()
    store.record("dd" * 32, dict(_CTX), dict(_AXES), "verified", None,
                 recorded_at="2029-06-01T00:00:00Z")
    original = store.replay("dd" * 32)
    assert original is not None
    assert original.context["event_time"] == "2029-05-01T00:00:00Z"
    assert original.context["effective_time"] == "2029-05-02T00:00:00Z"
    live = _view(context={"event_time": "2029-07-01T00:00:00Z"})
    out = classify({"doc_sha256": original.doc_sha256, "axes": dict(original.axes),
                    "context": dict(original.context)}, live, "verified", "unverified")
    assert out.label == "attributed_to_inputs" and out.differing == ("event_time",)
