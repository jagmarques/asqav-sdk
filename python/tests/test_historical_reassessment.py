"""Historical reassessment scenarios (624 C4/C7).

(a) prospective policy change and (d) append-with-provenance fail first:
VerificationContext and the retained context drop trust_policy_id and
profile_revision. (b) corrections and (c) key-status pin shipped behavior.
"""
from __future__ import annotations

import copy
import hashlib
import json

from asqav.verifier.oracle.adapter import VerificationContext
from asqav.verifier.oracle.assessment_store import AssessmentStore
from asqav.verifier.oracle.retain import reassess_verify_result, record_verify_result
from asqav.verifier.verify_receipt import check_key_status


class _Axis:
    def __init__(self, axis: str, result: str) -> None:
        self.axis = axis
        self.result = result


class _Result:
    """Minimal VerifyResult shape the retain helpers map."""
    def __init__(self, verdict: str, axes: list, context: object) -> None:
        self.verdict = verdict
        self.failure_class = None
        self.axes = axes
        self.context = context


_AXES = [_Axis("expiry", "PASS"), _Axis("skew", "PASS")]
_RAW_A = json.dumps({"v": 1, "type": "protectmcp:ack"}, sort_keys=True).encode()
_SHA_A = hashlib.sha256(_RAW_A).hexdigest()
_RAW_B = json.dumps({"v": 1, "type": "protectmcp:ack", "n": 2}, sort_keys=True).encode()
_SHA_B = hashlib.sha256(_RAW_B).hexdigest()


def _ctx(policy: str, trust: str, profile: str, at: str) -> VerificationContext:
    return VerificationContext(
        policy_revision=policy, trust_policy_id=trust,
        profile_revision=profile, evaluation_time=at, evaluator="624-probe",
    )


def _res(verdict: str, policy: str, trust: str, profile: str, at: str) -> _Result:
    return _Result(verdict, _AXES, _ctx(policy, trust, profile, at))


_T1 = "2029-01-01T00:00:00Z"
_T2 = "2030-01-01T00:00:00Z"


def test_prospective_policy_change_appends_separate_record() -> None:
    store = AssessmentStore()
    first = record_verify_result(store, _SHA_A, _res("verified", "P1", "T1", "prof-9", _T1))
    snap = copy.deepcopy(first)
    second = reassess_verify_result(
        store, _SHA_A, _res("verified", "P2", "T2", "prof-10", _T2),
        recorded_at=_T2, reason="policy-updated")
    assert store.replay(_SHA_A) == snap, "P1 original must never be rewritten"
    assert store.current(_SHA_A) == second
    got = (first.context["policy_revision"], second.context["policy_revision"])
    assert got == ("P1", "P2")
    got = (first.context["trust_policy_id"], second.context["trust_policy_id"])
    assert got == ("T1", "T2")
    assert second.supersedes == first.entry_hash and store.verify_chain()


def test_alleged_retroactive_correction_never_replaces_original() -> None:
    store = AssessmentStore()
    first = record_verify_result(store, _SHA_A, _res("verified", "P1", "T1", "prof-9", _T1))
    snap = copy.deepcopy(first)
    corrected = record_verify_result(store, _SHA_B, _res("verified", "P1", "T1", "prof-9", _T1))
    assert corrected.kind == "original" and corrected.doc_sha256 == _SHA_B
    assert store.replay(_SHA_A) == snap, "corrected bytes land beside, never over"
    try:
        store.reassess("cc" * 32, {}, {}, "verified", None)
    except ValueError:
        pass
    else:
        raise AssertionError("reassess of unretained bytes must raise")
    assert store.verify_chain()


def test_shipped_key_status_controls_agree() -> None:
    rev = "2026-01-01T00:00:00Z"
    result, _ = check_key_status("revoked", "2026-06-01T00:00:00Z", revoked_at=rev)
    assert result == "FAIL", "later-discovered compromise fails"
    result, _ = check_key_status("revoked", "2026-06-01T00:00:00Z", revoked_at=None)
    assert result == "SKIPPED", "unavailable authority evidence skips"
    result, _ = check_key_status("active", "2026-06-01T00:00:00Z")
    assert result == "PASS"


def test_reassessment_appends_provenance() -> None:
    store = AssessmentStore()
    first = record_verify_result(store, _SHA_A, _res("verified", "P1", "T1", "prof-9", _T1))
    second = reassess_verify_result(
        store, _SHA_A, _res("verified", "P2", "T2", "prof-10", _T2),
        recorded_at=_T2, reason="policy-updated")
    assert second.doc_sha256 == _SHA_A == hashlib.sha256(_RAW_A).hexdigest()
    assert second.context["profile_revision"] == "prof-10"
    assert second.context["trust_policy_id"] == "T2"
    assert second.recorded_at == _T2
    assert second.supersedes == first.entry_hash
    assert first.context["policy_revision"] == "P1"
    restored = AssessmentStore.from_jsonl(store.to_jsonl())
    assert len(restored) == 2 and restored.verify_chain()
    assert restored.current(_SHA_A) == second
