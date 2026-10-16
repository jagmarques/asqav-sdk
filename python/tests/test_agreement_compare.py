"""Agreement comparanda (488 C12): all-match, single-dimension flips, none."""
from __future__ import annotations

from asqav.verifier.oracle.agreement import COMPARANDA, AgreementRef, compare_assessments

_CTX = {
    "evaluator": "c12-probe",
    "trust_roots": ["root-a", "root-b"],
    "revocation_snapshot": "snap-1",
    "evaluation_time": "2029-06-01T00:00:00Z",
}
_AXES = {"expiry": "PASS", "skew": "PASS"}
_AGR = AgreementRef.of_terms("agr-1", "2029-01", {"clause": "pay on delivery"})


def _view(**over: object) -> dict:
    base: dict = {
        "doc_sha256": "dd" * 32,
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


def test_identical_views_match_on_all_seven() -> None:
    result = compare_assessments(_view(), _view())
    assert result == {name: True for name in COMPARANDA}


def test_terms_change_flips_only_agreement_version() -> None:
    other = _view(agreement=AgreementRef.of_terms("agr-1", "2029-01", {"clause": "pay upfront"}))
    result = compare_assessments(_view(), other)
    assert result["agreement_version"] is False
    assert all(v for k, v in result.items() if k != "agreement_version")


def test_later_evaluation_time_flips_only_verification_time() -> None:
    other = _view(context={"evaluation_time": "2031-01-01T00:00:00Z"})
    result = compare_assessments(_view(), other)
    assert result["verification_time"] is False
    assert all(v for k, v in result.items() if k != "verification_time")


def test_missing_agreement_is_a_mismatch_not_a_crash() -> None:
    result = compare_assessments(_view(agreement=None), _view())
    assert result["agreement_version"] is False
    assert result["evidence_bytes"] is True


def test_reads_camel_case_contexts_from_the_other_language() -> None:
    snake_view = _view()
    snake_view["context"] = {
        "evaluator": "c12-probe",
        "trustRoots": ["root-b", "root-a"],
        "revocationSnapshot": "snap-1",
        "evaluationTime": "2029-06-01T00:00:00Z",
    }
    assert compare_assessments(_view(), snake_view) == {name: True for name in COMPARANDA}


_NON_ASCII_DIGEST = "3a357c459fc615cbc788440180f136108db0051615defa7f819bf38b8b72f3f7"


def test_non_ascii_terms_hash_matches_the_other_language() -> None:
    ref = AgreementRef.of_terms("agr-1", "2029-01", {"clause": "páy on delivery ✓"})
    assert ref.terms_hash == _NON_ASCII_DIGEST


def test_cross_casing_agreement_dicts_compare_by_value() -> None:
    camel = {"agreementId": "agr-1", "predicateVersion": "2029-01", "termsHash": _AGR.terms_hash}
    same = compare_assessments(_view(), _view(agreement=camel))
    assert same["agreement_version"] is True
    different = dict(camel, termsHash="DIFFERENT")
    assert compare_assessments(_view(), _view(agreement=different))["agreement_version"] is False


def test_both_absent_agreements_match() -> None:
    assert compare_assessments(_view(agreement=None), _view(agreement=None))[
        "agreement_version"
    ] is True
