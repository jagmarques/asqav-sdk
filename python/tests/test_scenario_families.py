"""C7 scenario families (488 C14): success, contradiction, missing, drift.

The four disagreement scenarios run end to end from retained assessments
through the classifier: family 2 has contradictory verdicts on identical
inputs, family 3 drops the agreement comparandum, family 4 varies trust/time."""
from __future__ import annotations

from asqav.verifier.oracle.agreement import AgreementRef
from asqav.verifier.oracle.disagreement import attributed_to_trust_time, classify

_DOC = "ff" * 32
_T1 = "2029-06-01T00:00:00Z"
_T2 = "2031-01-01T00:00:00Z"


def _ctx(**over: object) -> dict:
    base = {
        "evaluator": "c14-probe",
        "trust_roots": ["root-a"],
        "revocation_snapshot": "snap-1",
        "evaluation_time": _T1,
        "policy_revision": "2029-01",
        "evidence_knowledge_cutoff": _T1,
    }
    return {**base, **over}


_AXES_OK = {"expiry": "PASS", "skew": "PASS"}
_AXES_BAD = {"expiry": "FAIL", "skew": "PASS"}
_AGR = AgreementRef.of_terms("agr-1", "2029-01", {"clause": "pay on delivery"})


def _view(**over: object) -> dict:
    base = {"doc_sha256": _DOC, "axes": dict(_AXES_OK), "context": _ctx(), "agreement": _AGR}
    for key, value in over.items():
        if key == "context":
            base["context"] = _ctx(**value)  # type: ignore[arg-type]
        else:
            base[key] = value
    return base


def test_family_1_success_is_agreement() -> None:
    out = classify(_view(), _view(), "verified", "verified")
    assert out.label == "agreement" and out.differing == ()


def test_family_2_contradictory_terms_are_unexplained() -> None:
    out = classify(_view(), _view(), "verified", "unverified")
    assert out.label == "unexplained_needs_investigation"


def test_family_3_missing_agreement_attributes() -> None:
    out = classify(_view(agreement=None), _view(), "verified", "unverified")
    assert out.label == "attributed_to_inputs"
    assert out.differing == ("agreement_version",)
    assert not attributed_to_trust_time(out)


def test_family_4_trust_time_drift_is_legitimate() -> None:
    later = _view(context={"evaluation_time": _T2, "evidence_knowledge_cutoff": _T2})
    out = classify(_view(), later, "verified", "unverified")
    assert out.label == "attributed_to_inputs"
    assert attributed_to_trust_time(out)
