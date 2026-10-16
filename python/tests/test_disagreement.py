"""Disagreement classifier (488 C13): agreement, attribution, unexplained."""
from __future__ import annotations

from asqav.verifier.oracle.agreement import AgreementRef, compare_assessments
from asqav.verifier.oracle.disagreement import attributed_to_trust_time, classify

_CTX = {
    "evaluator": "c13-probe",
    "trust_roots": ["root-a"],
    "revocation_snapshot": "snap-1",
    "evaluation_time": "2029-06-01T00:00:00Z",
}
_AXES = {"expiry": "PASS", "skew": "PASS"}
_AGR = AgreementRef.of_terms("agr-1", "2029-01", {"clause": "pay on delivery"})


def _view(**over: object) -> dict:
    base: dict = {
        "doc_sha256": "ee" * 32,
        "axes": dict(_AXES),
        "context": dict(_CTX),
        "agreement": _AGR,
    }
    for key, value in over.items():
        if key == "context":
            base["context"] = {**_CTX, **value}  # type: ignore[dict-item]
        else:
            base[key] = value
    return base


def test_same_verdict_is_agreement() -> None:
    out = classify(_view(), _view(), "verified", "verified")
    assert out.label == "agreement" and out.differing == ()


def test_time_only_difference_attributes_to_trust_time() -> None:
    later = _view(context={"evaluation_time": "2031-01-01T00:00:00Z"})
    out = classify(_view(), later, "verified", "unverified")
    assert out.label == "attributed_to_inputs"
    assert out.differing == ("verification_time",)
    assert attributed_to_trust_time(out)


def test_terms_difference_attributes_but_not_to_trust_time() -> None:
    other = _view(agreement=AgreementRef.of_terms("agr-1", "2029-02", {"clause": "pay on delivery"}))
    out = classify(_view(), other, "verified", "unverified")
    assert out.label == "attributed_to_inputs"
    assert out.differing == ("agreement_version",)
    assert not attributed_to_trust_time(out)


def test_same_inputs_different_verdict_is_unexplained() -> None:
    out = classify(_view(), _view(), "verified", "unverified")
    assert out.label == "unexplained_needs_investigation" and out.differing == ()


def test_different_bytes_are_a_different_subject() -> None:
    other = _view()
    other["doc_sha256"] = "ff" * 32
    assert classify(_view(), other, "verified", "unverified").label == "different_subject"


def test_same_verdict_with_input_drift_stays_agreement() -> None:
    later = _view(context={"evaluation_time": "2031-01-01T00:00:00Z"})
    out = classify(_view(), later, "verified", "verified")
    assert out.label == "agreement" and out.differing == ("verification_time",)
