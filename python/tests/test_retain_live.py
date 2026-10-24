"""Retain live oracle results (488 C16): mapping plus live two-era run.

The mapping test pins the VerifyResult -> retained shape. The live test
runs the real oracle twice over byte-identical bytes at two evaluation
times, retains both, and classifies: trust/time-attributed, replayable.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify
from asqav.verifier.oracle.assessment_store import AssessmentStore
from asqav.verifier.oracle.disagreement import classify
from asqav.verifier.oracle.retain import axes_dict, context_dict
from asqav.verifier.oracle.retain import reassess_verify_result, record_verify_result

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"


def _expiring_doc() -> tuple[dict, str]:
    doc = json.loads((_CORPUS / "asqav-01-genesis-permit" / "receipt.json").read_bytes())
    doc["payload"]["expires_at"] = "2020-01-01T00:00:00Z"
    raw = json.dumps(doc, sort_keys=True).encode()
    return json.loads(raw.decode()), hashlib.sha256(raw).hexdigest()


def test_mapping_keeps_attribution_fields() -> None:
    doc, digest = _expiring_doc()
    res = verify(doc, ADAPTERS, context=VerificationContext(
        evaluation_time="2019-06-01T00:00:00Z", evaluator="c16-probe",
        trust_roots=["root-a"], revocation_snapshot="snap-1",
    ))
    assert res.axis("expiry").result == "PASS"
    ctx = context_dict(res)
    assert ctx["evaluator"] == "c16-probe" and ctx["trust_roots"] == ["root-a"]
    assert ctx["evaluation_time"] == "2019-06-01T00:00:00Z"
    axes = axes_dict(res)
    assert axes["expiry"] == "PASS" and len(axes) == len(res.axes)
    entry = record_verify_result(AssessmentStore(), digest, res)
    assert entry.doc_sha256 == digest and entry.verdict == res.verdict


def test_freeze_normalizes_nested_datetimes() -> None:
    from types import SimpleNamespace

    from asqav.verifier.oracle.retain import context_dict

    ctx = SimpleNamespace(
        evaluator="c16-probe",
        trust_roots=["root-a"],
        revocation_snapshot="snap-1",
        policy_revision="2029-01",
        evidence_knowledge_cutoff={"at": datetime(2019, 6, 1)},
        evaluation_time=datetime(2019, 6, 1, tzinfo=timezone.utc),
    )
    frozen = context_dict(SimpleNamespace(context=ctx, axes=[]))
    assert frozen["evaluation_time"] == "2019-06-01T00:00:00+00:00"
    assert frozen["evidence_knowledge_cutoff"] == {"at": "2019-06-01T00:00:00"}


def test_live_two_era_run_retains_and_replays() -> None:
    """Same bytes at two evaluation times: axis flips, verdict holds.

    Both runs are unverified (signature SKIPPED without a key provider downgrades), but expiry legitimately
    flips PASS->FAIL with time moved. The store replays the first era; the classifier reports agreement with
    recorded drift, never a false split."""
    doc, digest = _expiring_doc()
    store = AssessmentStore()
    first = record_verify_result(store, digest, verify(
        doc, ADAPTERS, context=VerificationContext(evaluation_time="2019-06-01T00:00:00Z")))
    second = reassess_verify_result(store, digest, verify(
        doc, ADAPTERS, context=VerificationContext(evaluation_time="2026-09-21T00:00:00Z")))
    assert first.axes["expiry"] == "PASS" and second.axes["expiry"] == "FAIL"
    assert second.kind == "reassessment" and second.supersedes == first.entry_hash
    assert first.verdict == second.verdict == "unverified"
    assert store.replay(digest) == first
    out = classify(first.__dict__, second.__dict__, first.verdict, second.verdict)
    assert out.label == "agreement"
    assert "verification_time" in out.differing
    assert "content_hashes" in out.differing, "axes moved with time"
