"""Four-dimension replay with declared per-dim effects (488 C17, C9 core)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify
from asqav.verifier.oracle.assessment_store import AssessmentStore
from asqav.verifier.oracle.disagreement import classify
from asqav.verifier.oracle.retain import reassess_verify_result, record_verify_result

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"
_T1 = "2019-06-01T00:00:00Z"
_T2 = "2026-09-21T00:00:00Z"


def _frozen_doc(issued_at: str | None = None) -> tuple[dict, str]:
    doc = json.loads((_CORPUS / "asqav-01-genesis-permit" / "receipt.json").read_bytes())
    doc["payload"]["expires_at"] = "2020-01-01T00:00:00Z"
    if issued_at is not None:
        doc["payload"]["issued_at"] = issued_at
    raw = json.dumps(doc, sort_keys=True).encode()
    assert "issued_at" in doc["payload"], "event time must ride in the bytes"
    return json.loads(raw.decode()), hashlib.sha256(raw).hexdigest()


def _ctx(**over: object) -> VerificationContext:
    base = {
        "evaluation_time": _T1,
        "trust_roots": ["root-a"],
        "revocation_snapshot": "snap-1",
        "policy_revision": "2029-01",
        "evidence_knowledge_cutoff": _T1,
        "evaluator": "c17-probe",
    }
    return VerificationContext(**{**base, **over})


def test_four_dims_with_declared_effects_and_replay() -> None:
    doc, digest = _frozen_doc()
    store = AssessmentStore()
    baseline = record_verify_result(store, digest, verify(doc, ADAPTERS, context=_ctx()))
    assert baseline.axes["expiry"] == "PASS"

    moved = reassess_verify_result(
        store, digest, verify(doc, ADAPTERS, context=_ctx(evaluation_time=_T2)))
    assert moved.axes["expiry"] == "FAIL", "evaluation_time moves time axes"
    assert moved.kind == "reassessment" and moved.supersedes == baseline.entry_hash

    for label, over in (
        ("policy_revision", {"policy_revision": "2031-01"}),
        ("cutoff", {"evidence_knowledge_cutoff": _T2}),
        ("trust_roots", {"trust_roots": ["root-b"]}),
    ):
        same = verify(doc, ADAPTERS, context=_ctx(**over))
        assert {a.axis: a.result for a in same.axes} == baseline.axes, label
        assert same.verdict == baseline.verdict, label

    event_doc, event_digest = _frozen_doc(issued_at="2018-01-01T00:00:00Z")
    assert event_digest != digest, "event variant is separately keyed"
    event_run = verify(event_doc, ADAPTERS, context=_ctx())
    assert event_run.axis("skew").result == "PASS", "event time moves the skew axis"
    assert baseline.axes["skew"] == "FAIL"
    record_verify_result(store, event_digest, event_run)

    replayed = verify(doc, ADAPTERS, context=VerificationContext(**store.replay(digest).context))
    assert {a.axis: a.result for a in replayed.axes} == baseline.axes
    assert replayed.verdict == baseline.verdict, "re-execution reproduces the baseline"
    out = classify(baseline.__dict__, moved.__dict__, baseline.verdict, moved.verdict)
    assert out.label == "agreement", "verdicts never split here (expiry is non-folding)"
    assert out.differing == ("content_hashes", "verification_time")
