"""Live compare-before-classify over real oracle output (488 C19, C5 core).

Two live VerifyResults of stated bytes run through the same C12 compare
+ C13 classify path the retained views use. Predicate terms bind by id +
version + sha256 of canonical terms bytes from the receipt envelope.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify
from asqav.verifier.oracle.agreement import AgreementRef
from asqav.verifier.oracle.disagreement import attributed_to_trust_time
from asqav.verifier.oracle.live_compare import compare_live

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"
_T1 = "2019-06-01T00:00:00Z"
_T2 = "2026-09-21T00:00:00Z"


def _expiring_doc() -> tuple[dict, str]:
    doc = json.loads((_CORPUS / "asqav-01-genesis-permit" / "receipt.json").read_bytes())
    doc["payload"]["expires_at"] = "2020-01-01T00:00:00Z"
    raw = json.dumps(doc, sort_keys=True).encode()
    return json.loads(raw.decode()), hashlib.sha256(raw).hexdigest()


def _run(doc: dict, evaluation_time: str):
    return verify(doc, ADAPTERS, context=VerificationContext(evaluation_time=evaluation_time))


def test_two_era_live_runs_classify_as_agreement_with_drift() -> None:
    doc, digest = _expiring_doc()
    out = compare_live(digest, _run(doc, _T1), digest, _run(doc, _T2))
    assert out.label == "agreement"
    assert "verification_time" in out.differing and "content_hashes" in out.differing
    assert not attributed_to_trust_time(out), "axis drift is not pure trust/time"


def test_same_era_live_runs_fully_agree() -> None:
    doc, digest = _expiring_doc()
    out = compare_live(digest, _run(doc, _T1), digest, _run(doc, _T1))
    assert out.label == "agreement" and out.differing == ()


def test_different_input_bytes_are_a_different_subject() -> None:
    doc, digest = _expiring_doc()
    other, other_digest = _expiring_doc()
    other["payload"]["expires_at"] = "2021-01-01T00:00:00Z"
    other_digest = hashlib.sha256(json.dumps(other, sort_keys=True).encode()).hexdigest()
    out = compare_live(digest, _run(doc, _T1), other_digest, _run(other, _T1))
    assert out.label == "different_subject"


def test_predicate_terms_bind_by_id_version_and_hash() -> None:
    doc, digest = _expiring_doc()
    terms = AgreementRef.of_terms("pred-1", "2026-01", doc["payload"])
    out = compare_live(digest, _run(doc, _T1), digest, _run(doc, _T1), terms, terms)
    assert out.label == "agreement" and "agreement_version" not in out.differing
    moved = AgreementRef.of_terms("pred-1", "2026-02", doc["payload"])
    out = compare_live(digest, _run(doc, _T1), digest, _run(doc, _T1), terms, moved)
    assert out.label == "agreement" and "agreement_version" in out.differing
