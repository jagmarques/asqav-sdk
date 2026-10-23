"""Classifier comparison parity (488 C20): exact tuples plus absent parity."""
from __future__ import annotations

from asqav.verifier.oracle.agreement import COMPARANDA, compare_assessments
from asqav.verifier.oracle.disagreement import classify


def _view(**over: object) -> dict:
    base = {
        "doc_sha256": "aa" * 32,
        "axes": {"expiry": "PASS"},
        "context": {"evaluator": "c20", "evaluation_time": "T1"},
    }
    return {**base, **over}


def test_single_dimension_differences_are_exact() -> None:
    base = _view()
    only_eval = _view(context={"evaluator": "other", "evaluation_time": "T1"})
    assert classify(base, only_eval, "v", "v").differing == ("evaluator",)
    only_axes = _view(axes={"expiry": "FAIL"})
    assert classify(base, only_axes, "v", "v").differing == ("content_hashes",)
    only_trust = _view(context={
        "evaluator": "c20", "evaluation_time": "T1", "trust_roots": ["root-b"],
    })
    assert classify(base, only_trust, "v", "v").differing == ("trust_roots",)


def test_multi_dimension_difference_follows_comparanda_order() -> None:
    base = _view()
    moved = _view(
        axes={"expiry": "FAIL"},
        context={"evaluator": "other", "evaluation_time": "T2", "trust_roots": ["r"]},
    )
    out = classify(base, moved, "v", "v")
    assert out.label == "agreement"
    assert out.differing == ("content_hashes", "evaluator", "trust_roots", "verification_time")
    assert list(out.differing) == [n for n in COMPARANDA if n in out.differing]


def test_absent_none_and_sibling_null_read_the_same() -> None:
    missing = _view(context={})
    assert compare_assessments(missing, _view(context={"evaluation_time": None}))[
        "verification_time"
    ], "absent and None both read missing"
    nulled = _view(context={"evaluation_time": None, "evaluationTime": "T1"})
    assert not compare_assessments(nulled, missing)["verification_time"]
    assert all(compare_assessments(nulled, _view(context={"evaluationTime": "T1"})).values())


def test_view_without_axes_compares_as_empty() -> None:
    base = _view()
    no_axes = {"doc_sha256": "aa" * 32, "context": base["context"]}
    assert compare_assessments(no_axes, no_axes)["content_hashes"]
    assert not compare_assessments(no_axes, base)["content_hashes"]
