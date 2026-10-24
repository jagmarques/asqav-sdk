"""Conflict kinds (488 C7): contradictory terms vs missing evidence."""
from __future__ import annotations

from asqav.verifier.oracle.agreement import AgreementRef
from asqav.verifier.oracle.disagreement import conflict_kind

_CTX = {"evaluator": "c34-probe"}
_AXES = {"expiry": "PASS", "skew": "PASS"}
_TERMS_A = AgreementRef.of_terms("agr-1", "2029-01", {"clause": "pay on delivery"})
_TERMS_B = AgreementRef.of_terms("agr-1", "2029-01", {"clause": "refund in full"})


def _view(**over: object) -> dict:
    base: dict = {
        "doc_sha256": "ee" * 32,
        "axes": dict(_AXES),
        "context": dict(_CTX),
        "agreement": _TERMS_A,
    }
    base.update(over)
    return base


def test_contradictory_signed_terms() -> None:
    other = _view(agreement=_TERMS_B)
    assert conflict_kind(_view(), other) == "contradictory_terms"


def test_missing_evidence_first_side_empty() -> None:
    assert conflict_kind(_view(axes={}), _view()) == "missing_evidence"


def test_missing_evidence_second_side_empty() -> None:
    assert conflict_kind(_view(), _view(axes={})) == "missing_evidence"


def test_both_sides_empty_is_no_call() -> None:
    assert conflict_kind(_view(axes={}), _view(axes={})) is None


def test_different_subject_is_no_call() -> None:
    other = _view()
    other["doc_sha256"] = "ff" * 32
    assert conflict_kind(_view(), other) is None
