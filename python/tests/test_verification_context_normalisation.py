"""Normalization-context widening (criterion 488, T1)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"


def _expiring_doc() -> tuple[dict, str]:
    """Corpus receipt with an absolute-past expiry; returns (doc, sha256)."""
    doc = json.loads((_CORPUS / "asqav-01-genesis-permit" / "receipt.json").read_bytes())
    doc["payload"]["expires_at"] = "2020-01-01T00:00:00Z"
    raw = json.dumps(doc, sort_keys=True).encode()
    return json.loads(raw.decode()), hashlib.sha256(raw).hexdigest()


def test_context_constructs_with_the_new_fields() -> None:
    ctx = VerificationContext(
        evaluation_time="2019-06-01T00:00:00Z",
        trust_roots=["root-a"],
        revocation_snapshot="snap-1",
        policy_revision="2026-09",
        evidence_knowledge_cutoff="2026-09-01T00:00:00Z",
        evaluator="t1-probe",
    )
    assert ctx.evaluation_time == "2019-06-01T00:00:00Z"
    assert ctx.evaluator == "t1-probe"


def test_evaluation_time_moves_the_expiry_axis() -> None:
    """Same input bytes, different evaluation_time, different axis result."""
    doc, digest = _expiring_doc()
    assert hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest() == digest
    past = verify(doc, ADAPTERS, context=VerificationContext(evaluation_time="2019-06-01T00:00:00Z"))
    now = verify(doc, ADAPTERS, context=VerificationContext(evaluation_time="2026-09-21T00:00:00Z"))
    assert past.axis("expiry").result == "PASS", past.axis("expiry").note
    assert now.axis("expiry").result == "FAIL", now.axis("expiry").note
    assert past.context is not None and past.context.evaluation_time == "2019-06-01T00:00:00Z"


def test_empty_context_preserves_current_outcomes() -> None:
    """No context supplied: wall-clock outcomes, context recorded empty."""
    doc, _digest = _expiring_doc()
    res = verify(doc, ADAPTERS)
    assert res.axis("expiry").result == "FAIL"
    assert res.context is not None and res.context.evaluation_time is None
