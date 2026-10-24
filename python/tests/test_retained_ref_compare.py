"""Stored-vs-live envelope comparison through classification (488 C39b).

The retained view carries the stored ref, the live view the envelope
re-derived from the reviewed bytes; classify surfaces agreement_version
agreement/split plus the conflict shape on the Disagreement.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from asqav.verifier.oracle import ADAPTERS, verify
from asqav.verifier.oracle.__main__ import main
from asqav.verifier.oracle.agreement import AgreementRef, envelope_of
from asqav.verifier.oracle.assessment_store import AssessmentStore
from asqav.verifier.oracle.disagreement import classify, conflict_kind
from asqav.verifier.oracle.live_compare import classify_and_attribute
from asqav.verifier.oracle.retain import record_verify_result

_FIX = Path(__file__).resolve().parents[2] / "verifier" / "signed-conflict-fixtures"


def _raw(name: str) -> bytes:
    return (_FIX / f"{name}.json").read_bytes()


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _ref(name: str) -> dict:
    return asdict(envelope_of(json.loads(_raw(name)), ADAPTERS))


def _view(doc_sha: str, axes: dict, ref: dict | None) -> dict:
    return {"doc_sha256": doc_sha, "axes": dict(axes), "context": {},
            "agreement": ref}


def test_matching_refs_agree_with_no_conflict() -> None:
    raw = _raw("conflict-allow")
    doc, digest = json.loads(raw), _digest(raw)
    ref = asdict(envelope_of(doc, ADAPTERS))
    store = AssessmentStore()
    record_verify_result(store, digest, verify(doc, ADAPTERS), agreement=ref)
    out, _ = classify_and_attribute(store, digest, verify(doc, ADAPTERS),
                                    agreement=ref)
    assert out.label == "agreement"
    assert "agreement_version" not in out.differing
    assert out.conflict is None


def test_forged_stored_ref_surfaces_as_split_not_agreement() -> None:
    raw = _raw("conflict-allow")
    doc, digest = json.loads(raw), _digest(raw)
    live = asdict(envelope_of(doc, ADAPTERS))
    forged = dict(live, kid="did:key:zForged")
    store = AssessmentStore()
    record_verify_result(store, digest, verify(doc, ADAPTERS), agreement=forged)
    out, _ = classify_and_attribute(store, digest, verify(doc, ADAPTERS),
                                    agreement=live)
    assert "agreement_version" in out.differing
    assert out.conflict is None


def test_contradictory_terms_name_the_conflict() -> None:
    first = _view("dd" * 32, {"expiry": "PASS"},
                  asdict(AgreementRef.of_terms("a", "1", {"c": 1})))
    second = _view("dd" * 32, {"expiry": "PASS"},
                   asdict(AgreementRef.of_terms("a", "1", {"c": 2})))
    assert conflict_kind(first, second) == "contradictory_terms"
    out = classify(first, second, "unverified", "unverified")
    assert out.conflict == "contradictory_terms"
    assert "agreement_version" in out.differing


def test_one_sided_evidence_names_missing_evidence() -> None:
    full = _view("dd" * 32, {"expiry": "PASS"}, None)
    empty = _view("dd" * 32, {}, None)
    assert conflict_kind(full, empty) == "missing_evidence"
    assert conflict_kind(empty, full) == "missing_evidence"
    out = classify(full, empty, "unverified", "unverified")
    assert out.conflict == "missing_evidence"


def test_different_bytes_carry_no_conflict_call() -> None:
    first = _view("dd" * 32, {"expiry": "PASS"}, None)
    second = _view("ee" * 32, {"expiry": "PASS"}, None)
    out = classify(first, second, "unverified", "unverified")
    assert out.label == "different_subject"
    assert out.conflict is None


def test_review_flow_derives_the_live_side_from_bytes(tmp_path: Path, capsys) -> None:
    raw = _raw("conflict-allow")
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(raw)
    store = tmp_path / "store.jsonl"
    assert main(["record", str(store), str(receipt)]) == 0
    capsys.readouterr()
    # Tamper the stored ref (simulated forgery): review must surface
    # the split against the bytes-derived live envelope, not agree.
    lines = [json.loads(line) for line in store.read_text().splitlines()]
    lines[0]["agreement"]["kid"] = "did:key:zForged"
    store.write_text("\n".join(json.dumps(line, sort_keys=True) for line in lines))
    assert main(["review", str(store), str(receipt)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "agreement_version" in out["disagreement"]["differing"]
    assert out["disagreement"]["conflict"] is None


def test_review_flow_agrees_when_refs_match(tmp_path: Path, capsys) -> None:
    raw = _raw("conflict-allow")
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(raw)
    store = tmp_path / "store.jsonl"
    assert main(["record", str(store), str(receipt)]) == 0
    capsys.readouterr()
    assert main(["review", str(store), str(receipt),
                 "--rerun-a", str(receipt), "--rerun-b", str(receipt)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["disagreement"]["differing"] == []
    assert out["disagreement"]["conflict"] is None


def test_legacy_refless_original_never_attributes_a_split() -> None:
    raw = _raw("conflict-allow")
    doc, digest = json.loads(raw), _digest(raw)
    live_ref = asdict(envelope_of(doc, ADAPTERS))
    store = AssessmentStore()
    record_verify_result(store, digest, verify(doc, ADAPTERS))
    flipped = dataclasses.replace(verify(doc, ADAPTERS), verdict="unverified")
    out, _ = classify_and_attribute(store, digest, flipped, agreement=live_ref)
    assert out.label == "unexplained_needs_investigation"
    assert "agreement_version" not in out.differing
    assert out.conflict == "missing_evidence"


def test_legacy_refless_original_agrees_without_ref_drift() -> None:
    raw = _raw("conflict-allow")
    doc, digest = json.loads(raw), _digest(raw)
    live_ref = asdict(envelope_of(doc, ADAPTERS))
    store = AssessmentStore()
    record_verify_result(store, digest, verify(doc, ADAPTERS))
    out, _ = classify_and_attribute(store, digest, verify(doc, ADAPTERS),
                                    agreement=live_ref)
    assert out.label == "agreement"
    assert "agreement_version" not in out.differing
    assert out.conflict == "missing_evidence"


def test_symmetric_absent_refs_compare_as_before() -> None:
    first = _view("dd" * 32, {"expiry": "PASS"}, None)
    second = _view("dd" * 32, {"expiry": "PASS"}, None)
    out = classify(first, second, "unverified", "unverified")
    assert (out.label, out.differing, out.conflict) == ("agreement", (), None)
