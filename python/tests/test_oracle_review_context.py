"""Oracle `review --context`: a live trust-split through main() (488 C36).

With --context the live side (receipt plus reruns) verifies under
caller-stated trust roots, evaluator and evaluation time, so C2's trust-split
arm is testable. Tests address main() only, never the library entries.
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


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return str(path)


def _ctx_file(path: Path, obj: object) -> str:
    path.write_text(json.dumps(obj))
    return str(path)


def _store(path: Path, raw: bytes, result) -> str:
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


def _out(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_review_trust_split_attributes_through_the_flow(tmp_path: Path, capsys) -> None:
    raw = (_FIX / "conflict-allow.json").read_bytes()
    doc = json.loads(raw)
    store = _store(tmp_path / "store.jsonl", raw, verify(doc, ADAPTERS))
    ctx = _ctx_file(tmp_path / "ctx.json", {"trust_roots": ["example-root-a"]})
    code = main(["review", store, _write(tmp_path / "receipt.json", raw),
                 "--rerun-a", _write(tmp_path / "a.json", raw),
                 "--rerun-b", _write(tmp_path / "b.json", raw),
                 "--context", ctx])
    assert code == 0
    out = _out(capsys)
    assert out["disagreement"]["label"] == "agreement"
    assert out["disagreement"]["differing"] == ["trust_roots"]
    assert out["spec_case"]["label"] == "not_applicable"


def test_review_matching_context_erases_time_split(tmp_path: Path, capsys) -> None:
    doc = json.loads((_CORPUS / "asqav-01-genesis-permit" / "receipt.json").read_bytes())
    doc["payload"]["expires_at"] = "2020-01-01T00:00:00Z"
    raw = json.dumps(doc, sort_keys=True).encode()
    era = verify(json.loads(raw.decode()), ADAPTERS,
                 context=VerificationContext(evaluation_time=_T1))
    store = _store(tmp_path / "store.jsonl", raw, era)
    ctx = _ctx_file(tmp_path / "ctx.json", {"evaluation_time": _T1})
    code = main(["review", store, _write(tmp_path / "receipt.json", raw),
                 "--rerun-a", _write(tmp_path / "a.json", raw),
                 "--rerun-b", _write(tmp_path / "b.json", raw),
                 "--context", ctx])
    assert code == 0
    out = _out(capsys)
    assert out["disagreement"] == {"label": "agreement", "differing": [],
                                   "conflict": None}


def test_review_bad_context_exits_2(tmp_path: Path, capsys) -> None:
    raw = (_FIX / "conflict-allow.json").read_bytes()
    store = _store(tmp_path / "store.jsonl", raw,
                   verify(json.loads(raw), ADAPTERS))
    receipt = _write(tmp_path / "receipt.json", raw)
    bad = [tmp_path / "missing.json",
           Path(_ctx_file(tmp_path / "list.json", ["not", "an", "object"])),
           Path(_ctx_file(tmp_path / "unknown.json", {"no_such_field": 1})),
           Path(_ctx_file(tmp_path / "roots-int.json", {"trust_roots": 123})),
           Path(_ctx_file(tmp_path / "roots-mixed.json",
                           {"trust_roots": ["root-a", 1]})),
           Path(_ctx_file(tmp_path / "envelope.json",
                           {"originating_envelope": {"kid": "k"}})),
           Path(_ctx_file(tmp_path / "bad-time.json",
                           {"evaluation_time": "not-a-date"})),
           Path(_ctx_file(tmp_path / "evaluator-int.json", {"evaluator": 5}))]
    for ctx_path in bad:
        code = main(["review", store, receipt, "--context", str(ctx_path)])
        assert code == 2
        assert capsys.readouterr().out == ""


def test_review_reruns_inherit_the_live_context(
        tmp_path: Path, capsys, monkeypatch) -> None:
    import asqav.verifier.oracle.__main__ as cli
    seen: list = []
    real_verify = cli.verify

    def spy(doc, adapters, **kwargs):
        seen.append(kwargs.get("context"))
        return real_verify(doc, adapters, **kwargs)

    monkeypatch.setattr(cli, "verify", spy)
    raw = (_FIX / "conflict-allow.json").read_bytes()
    store = _store(tmp_path / "store.jsonl", raw,
                   verify(json.loads(raw), ADAPTERS))
    ctx = _ctx_file(tmp_path / "ctx.json", {"trust_roots": ["example-root-a"]})
    code = main(["review", store, _write(tmp_path / "receipt.json", raw),
                 "--rerun-a", _write(tmp_path / "a.json", raw),
                 "--rerun-b", _write(tmp_path / "b.json", raw),
                 "--context", ctx])
    assert code == 0
    assert len(seen) == 3
    assert all(getattr(c, "trust_roots", None) == ["example-root-a"] for c in seen)


def test_review_default_live_side_carries_no_context(
        tmp_path: Path, capsys, monkeypatch) -> None:
    import asqav.verifier.oracle.__main__ as cli
    seen: list = []
    real_verify = cli.verify

    def spy(doc, adapters, **kwargs):
        seen.append(kwargs.get("context"))
        return real_verify(doc, adapters, **kwargs)

    monkeypatch.setattr(cli, "verify", spy)
    raw = (_FIX / "conflict-allow.json").read_bytes()
    store = _store(tmp_path / "store.jsonl", raw,
                   verify(json.loads(raw), ADAPTERS))
    code = main(["review", store, _write(tmp_path / "receipt.json", raw),
                 "--rerun-a", _write(tmp_path / "a.json", raw),
                 "--rerun-b", _write(tmp_path / "b.json", raw)])
    assert code == 0
    assert seen == [None, None, None]


def test_review_context_states_event_times_through_the_flow(
        tmp_path: Path, capsys) -> None:
    raw = (_FIX / "conflict-allow.json").read_bytes()
    doc = json.loads(raw)
    store = _store(tmp_path / "store.jsonl", raw, verify(doc, ADAPTERS))
    ctx = _ctx_file(tmp_path / "ctx.json", {"event_time": "2029-05-01T00:00:00Z",
                                            "effective_time": "2029-05-02T00:00:00Z"})
    code = main(["review", store, _write(tmp_path / "receipt.json", raw),
                 "--rerun-a", _write(tmp_path / "a.json", raw),
                 "--rerun-b", _write(tmp_path / "b.json", raw),
                 "--context", ctx])
    assert code == 0
    out = _out(capsys)
    assert out["disagreement"]["label"] == "agreement"
    assert out["disagreement"]["differing"] == ["event_time", "effective_time"]
