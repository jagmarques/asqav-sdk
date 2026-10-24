"""Retained envelope refs ride outside the hash chain (488 C39a).

Recording stores the bytes-derived envelope (kid/signature from
envelope_of) on the entry for later conflict classification; the ref
is excluded from the chain hash so old logs still verify and hashes
never move. Unboundable receipts record with agreement None.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from asqav.verifier.oracle import ADAPTERS, verify
from asqav.verifier.oracle.__main__ import main
from asqav.verifier.oracle.agreement import envelope_of
from asqav.verifier.oracle.assessment_store import AssessmentStore
from asqav.verifier.oracle.retain import reassess_verify_result, record_verify_result

_FIX = Path(__file__).resolve().parents[2] / "verifier" / "signed-conflict-fixtures"


def _raw(name: str) -> bytes:
    return (_FIX / f"{name}.json").read_bytes()


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_record_stores_the_bytes_derived_envelope() -> None:
    raw = _raw("conflict-allow")
    doc = json.loads(raw)
    expected = asdict(envelope_of(doc, ADAPTERS))
    assert expected["kid"] and expected["signature"]
    store = AssessmentStore()
    entry = record_verify_result(store, _digest(raw), verify(doc, ADAPTERS),
                                 agreement=expected)
    assert entry.agreement == expected
    assert store.verify_chain()
    assert AssessmentStore.from_jsonl(store.to_jsonl()).replay(_digest(raw)) == entry


def test_agreement_rides_outside_the_chain_hash() -> None:
    raw = _raw("conflict-allow")
    doc, digest = json.loads(raw), _digest(raw)
    ref = asdict(envelope_of(doc, ADAPTERS))
    plain = record_verify_result(AssessmentStore(), digest, verify(doc, ADAPTERS))
    bound = record_verify_result(AssessmentStore(), digest, verify(doc, ADAPTERS),
                                 agreement=ref)
    assert bound.entry_hash == plain.entry_hash


def test_old_log_without_agreement_still_loads() -> None:
    raw = _raw("conflict-allow")
    doc, digest = json.loads(raw), _digest(raw)
    store = AssessmentStore()
    record_verify_result(store, digest, verify(doc, ADAPTERS),
                         agreement=asdict(envelope_of(doc, ADAPTERS)))
    old_lines = []
    for line in store.to_jsonl().splitlines():
        obj = json.loads(line)
        del obj["agreement"]
        old_lines.append(json.dumps(obj, sort_keys=True))
    restored = AssessmentStore.from_jsonl("\n".join(old_lines))
    assert restored.verify_chain()
    assert restored.replay(digest) is not None
    assert restored.replay(digest).agreement is None


def test_reassess_carries_the_envelope_ref() -> None:
    raw = _raw("conflict-allow")
    doc, digest = json.loads(raw), _digest(raw)
    ref = asdict(envelope_of(doc, ADAPTERS))
    store = AssessmentStore()
    record_verify_result(store, digest, verify(doc, ADAPTERS), agreement=ref)
    second = reassess_verify_result(store, digest, verify(doc, ADAPTERS), agreement=ref)
    assert second.agreement == ref
    assert store.verify_chain()


def test_record_command_derives_and_persists_the_ref(tmp_path: Path, capsys) -> None:
    raw = _raw("conflict-allow")
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(raw)
    store = tmp_path / "store.jsonl"
    assert main(["record", str(store), str(receipt)]) == 0
    capsys.readouterr()
    assert main(["record", str(store), str(receipt)]) == 0
    lines = [json.loads(line) for line in store.read_text().splitlines()]
    assert [entry["kind"] for entry in lines] == ["original", "reassessment"]
    expected = asdict(envelope_of(json.loads(raw), ADAPTERS))
    assert lines[0]["agreement"] == expected
    assert lines[1]["agreement"] == expected


def test_record_command_keeps_none_for_unboundable_bytes(tmp_path: Path, capsys) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"hello": "world"}')
    store = tmp_path / "store.jsonl"
    assert main(["record", str(store), str(receipt)]) == 0
    capsys.readouterr()
    entry = json.loads(store.read_text().splitlines()[0])
    assert entry["agreement"] is None
    assert AssessmentStore.from_jsonl(store.read_text()).verify_chain()
