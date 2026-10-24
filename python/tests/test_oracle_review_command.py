"""Oracle `review` command: files in, attribution out (488 C28, C2 flow).

main(["review", store, receipt, ...]) loads a retained log, verifies the
receipt plus live reruns and runs classify_and_attribute. Tests address main()
only. The candidate leg is a synthetic prior-version record (like C25).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography", reason="signed fixtures need the cryptography lib")

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify
from asqav.verifier.oracle.__main__ import main
from asqav.verifier.oracle.agreement import envelope_of
from asqav.verifier.oracle.assessment_store import AssessmentStore
from asqav.verifier.oracle.retain import record_verify_result

_FIX = Path(__file__).resolve().parents[2] / "verifier" / "signed-conflict-fixtures"
_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"
_T1 = "2019-06-01T00:00:00Z"
_T2 = "2026-09-21T00:00:00Z"


def _raw(name: str) -> bytes:
    return (_FIX / f"{name}.json").read_bytes()


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _run(doc: dict):
    # Default context, like the CLI verifies: retained and live runs share it.
    return verify(doc, ADAPTERS)


def _write(path: Path, raw: bytes) -> str:
    path.write_bytes(raw)
    return str(path)


def _store_file(path: Path, raw: bytes, result) -> str:
    # Mirror production record(): the retained side carries the
    # bytes-derived envelope ref, so review compares ref to ref.
    try:
        agreement = dataclasses.asdict(envelope_of(json.loads(raw), ADAPTERS))
    except ValueError:
        agreement = None
    store = AssessmentStore()
    record_verify_result(store, _digest(raw), result, agreement=agreement)
    path.write_text(store.to_jsonl())
    return str(path)


def _review_out(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_review_agrees_through_the_flow(tmp_path: Path, capsys) -> None:
    raw = _raw("conflict-allow")
    store = _store_file(tmp_path / "store.jsonl", raw, _run(json.loads(raw)))
    receipt = _write(tmp_path / "receipt.json", raw)
    code = main(["review", store, receipt,
                 "--rerun-a", _write(tmp_path / "a.json", raw),
                 "--rerun-b", _write(tmp_path / "b.json", raw)])
    assert code == 0
    out = _review_out(capsys)
    assert out["digest"] == _digest(raw)
    assert out["disagreement"] == {"label": "agreement", "differing": [],
                                   "conflict": None}
    assert out["spec_case"]["label"] == "not_applicable"


def test_review_missing_original_exits_2(tmp_path: Path, capsys) -> None:
    raw = _raw("conflict-allow")
    store = _write(tmp_path / "store.jsonl", b"")
    code = main(["review", store, _write(tmp_path / "receipt.json", raw)])
    assert code == 2
    assert capsys.readouterr().out == ""


def test_review_prior_record_split_exits_1_with_candidate(tmp_path: Path, capsys) -> None:
    raw = _raw("conflict-allow")
    digest = _digest(raw)
    # SYNTHETIC prior-version record (the only non-live element): a valid
    # chain over a flipped verdict, as an older oracle version would retain
    prior = dataclasses.replace(_run(json.loads(raw)), verdict="unverified")
    store = _store_file(tmp_path / "store.jsonl", raw, prior)
    receipt = _write(tmp_path / "receipt.json", raw)
    code = main(["review", store, receipt,
                 "--rerun-a", _write(tmp_path / "a.json", raw),
                 "--rerun-b", _write(tmp_path / "b.json", raw)])
    assert code == 1
    out = _review_out(capsys)
    assert out["disagreement"]["label"] == "unexplained_needs_investigation"
    assert out["spec_case"]["label"] == "spec_bug_candidate"
    assert out["spec_case"]["pinned"] == [digest]


def test_review_drift_surfaces_time_through_the_flow(tmp_path: Path, capsys) -> None:
    doc = json.loads((_CORPUS / "asqav-01-genesis-permit" / "receipt.json").read_bytes())
    doc["payload"]["expires_at"] = "2020-01-01T00:00:00Z"
    raw = json.dumps(doc, sort_keys=True).encode()

    def era(t: str):
        return verify(json.loads(raw.decode()), ADAPTERS,
                      context=VerificationContext(evaluation_time=t))

    store = _store_file(tmp_path / "store.jsonl", raw, era(_T1))
    receipt = _write(tmp_path / "receipt.json", raw)
    code = main(["review", store, receipt,
                 "--rerun-a", _write(tmp_path / "a.json", raw),
                 "--rerun-b", _write(tmp_path / "b.json", raw)])
    assert code == 0
    out = _review_out(capsys)
    assert out["disagreement"]["label"] == "agreement"
    assert "verification_time" in out["disagreement"]["differing"]


def test_review_rerun_must_reexecute_the_bytes(tmp_path: Path, capsys) -> None:
    allow, deny = _raw("conflict-allow"), _raw("conflict-deny")
    store = _store_file(tmp_path / "store.jsonl", allow, _run(json.loads(allow)))
    code = main(["review", store, _write(tmp_path / "receipt.json", allow),
                 "--rerun-a", _write(tmp_path / "a.json", allow),
                 "--rerun-b", _write(tmp_path / "b.json", deny)])
    assert code == 2
    assert capsys.readouterr().out == ""
