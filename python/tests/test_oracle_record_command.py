"""Oracle `record` command: live receipt in, retained store out (488 C32).

`record` creates the stores the review flow reads (record then review, all
through main()). C35 adds reassess: bytes already retained classify against
the original instead of failing. Tests address main() only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography", reason="signed fixtures need the cryptography lib")

from asqav.verifier.oracle import ADAPTERS, verify
from asqav.verifier.oracle.__main__ import main
from asqav.verifier.oracle.agreement import envelope_of
from asqav.verifier.oracle.assessment_store import AssessmentStore
from asqav.verifier.oracle.retain import record_verify_result

_FIX = Path(__file__).resolve().parents[2] / "verifier" / "signed-conflict-fixtures"


def _raw(name: str) -> bytes:
    return (_FIX / f"{name}.json").read_bytes()


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_record_then_review_agrees_through_the_flow(tmp_path: Path, capsys) -> None:
    raw = _raw("conflict-allow")
    store = str(tmp_path / "store.jsonl")
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(raw)
    assert main(["record", store, str(receipt)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"digest": _digest(raw), "recorded": 1}
    assert main(["review", store, str(receipt),
                 "--rerun-a", str(receipt), "--rerun-b", str(receipt)]) == 0


def test_record_appends_to_the_existing_log(tmp_path: Path, capsys) -> None:
    allow, deny = _raw("conflict-allow"), _raw("conflict-deny")
    store = tmp_path / "store.jsonl"
    first = tmp_path / "first.json"
    first.write_bytes(allow)
    assert main(["record", str(store), str(first)]) == 0
    capsys.readouterr()
    second = tmp_path / "second.json"
    second.write_bytes(deny)
    assert main(["record", str(store), str(second)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["recorded"] == 2
    assert len(AssessmentStore.from_jsonl(store.read_text())) == 2


def test_record_reassesses_and_classifies_a_duplicate_original(tmp_path: Path, capsys) -> None:
    # 488 C35 replaces the duplicate reject (RULES 14): retained bytes
    # reassess through the classification caller instead of failing.
    raw = _raw("conflict-allow")
    store = tmp_path / "store.jsonl"
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(raw)
    assert main(["record", str(store), str(receipt)]) == 0
    capsys.readouterr()
    assert main(["record", str(store), str(receipt)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["digest"] == _digest(raw)
    assert out["digest"] == _digest(raw)
    assert out["disagreement"] == {"label": "agreement", "differing": [],
                                   "conflict": None}
    assert out["spec_case"]["label"] == "not_applicable"
    first, second = (json.loads(line) for line in store.read_text().splitlines())
    assert second["kind"] == "reassessment"
    assert second["supersedes"] == first["entry_hash"]


def test_record_reassess_split_without_reruns_is_insufficient(tmp_path: Path, capsys) -> None:
    # 488 C38: the retain-layer caller classifies with no reruns, so a
    # live split surfaces unexplained but never a candidate from here.
    raw = _raw("conflict-allow")
    digest = _digest(raw)
    prior = dataclasses.replace(verify(json.loads(raw), ADAPTERS), verdict="unverified")
    store = AssessmentStore()
    # Production record() stores the ref; the hand-built prior must
    # carry it too, so the split comes from the verdict flip alone.
    ref = dataclasses.asdict(envelope_of(json.loads(raw), ADAPTERS))
    record_verify_result(store, digest, prior, agreement=ref)
    store_path = tmp_path / "store.jsonl"
    store_path.write_text(store.to_jsonl())
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(raw)
    assert main(["record", str(store_path), str(receipt)]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["disagreement"]["label"] == "unexplained_needs_investigation"
    assert out["spec_case"]["label"] == "evidence_insufficient"


def test_record_rejects_a_broken_receipt(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"a": 1, "a": 2}')
    store = str(tmp_path / "store.jsonl")
    with pytest.raises(SystemExit) as exc:
        main(["record", store, str(bad)])
    assert exc.value.code == 2
    assert not (tmp_path / "store.jsonl").exists()


def test_record_rejects_a_broken_store(tmp_path: Path) -> None:
    raw = _raw("conflict-allow")
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(raw)
    store = tmp_path / "store.jsonl"
    store.write_text("not jsonl\n")
    assert main(["record", str(store), str(receipt)]) == 2


def test_record_reason_labels_the_reassessment(tmp_path: Path, capsys) -> None:
    raw = _raw("conflict-allow")
    store = tmp_path / "store.jsonl"
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(raw)
    assert main(["record", str(store), str(receipt)]) == 0
    capsys.readouterr()
    assert main(["record", str(store), str(receipt),
                 "--reason", "key-compromised"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["disagreement"]["label"] == "agreement"
    first, second = (json.loads(line) for line in store.read_text().splitlines())
    assert second["kind"] == "reassessment"
    assert second["reason"] == "key-compromised"
    assert first["reason"] is None
