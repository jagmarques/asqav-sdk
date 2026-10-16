"""Four-dimension replay with declared per-dim effects (488 C17, C9 core).

Semantics decision (recorded 2026-10-16, reversible): event time lives in
the receipt bytes (payload issued_at) and is held fixed by holding bytes
fixed. Of the four varied context dims, only evaluation_time moves the
outcome (time axes); policy_revision and evidence_knowledge_cutoff ride
along as attribution passengers; trust_roots likewise. The test declares
each effect empirically against the live oracle, then replays the
baseline from retained inputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify
from asqav.verifier.oracle.assessment_store import AssessmentStore
from asqav.verifier.oracle.disagreement import classify
from asqav.verifier.oracle.retain import record_verify_result

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"
_T1 = "2019-06-01T00:00:00Z"
_T2 = "2026-09-21T00:00:00Z"


def _frozen_doc() -> tuple[dict, str]:
    doc = json.loads((_CORPUS / "asqav-01-genesis-permit" / "receipt.json").read_bytes())
    doc["payload"]["expires_at"] = "2020-01-01T00:00:00Z"
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

    moved = record_verify_result(
        store, digest, verify(doc, ADAPTERS, context=_ctx(evaluation_time=_T2)))
    assert moved.axes["expiry"] == "FAIL", "evaluation_time moves time axes"

    for label, over in (
        ("policy_revision", {"policy_revision": "2031-01"}),
        ("cutoff", {"evidence_knowledge_cutoff": _T2}),
        ("trust_roots", {"trust_roots": ["root-b"]}),
    ):
        same = verify(doc, ADAPTERS, context=_ctx(**over))
        assert {a.axis: a.result for a in same.axes} == baseline.axes, label
        assert same.verdict == baseline.verdict, label

    assert store.replay(digest) == baseline, "baseline reproduces from retained inputs"
    out = classify(baseline.__dict__, moved.__dict__, baseline.verdict, moved.verdict)
    assert out.label == "agreement", "same verdict, honest drift, no false split"
    assert "verification_time" in out.differing
