"""Default envelope binding in compare_live (488 C30, C5 second half).

A stated kid/signature without bytes raises (the C24 digest rule for the
envelope); bytes present bind the terms to the derived envelope.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography", reason="live verify needs the cryptography lib")

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify
from asqav.verifier.oracle.agreement import AgreementRef, envelope_of
from asqav.verifier.oracle.live_compare import compare_live

_FIX = Path(__file__).resolve().parents[2] / "verifier" / "signed-conflict-fixtures"
_CTX = {"evaluation_time": "2026-10-16T12:00:00Z", "evaluator": "c30-probe"}


def _raw(name: str) -> bytes:
    return (_FIX / f"{name}.json").read_bytes()


def _verify(raw: bytes):
    return verify(json.loads(raw), ADAPTERS, context=VerificationContext(**_CTX))


def test_stated_envelope_without_bytes_raises() -> None:
    raw = _raw("conflict-allow")
    digest = hashlib.sha256(raw).hexdigest()
    res = _verify(raw)
    env = envelope_of(json.loads(raw), ADAPTERS)
    honest = {"kid": env.kid, "signature": env.signature}
    with pytest.raises(ValueError, match="first envelope without bytes"):
        compare_live(digest, res, digest, res, honest, None)
    with pytest.raises(ValueError, match="second envelope without bytes"):
        compare_live(digest, res, digest, res, None, honest)
    with pytest.raises(ValueError, match="first envelope without bytes"):
        compare_live(digest, res, digest, res, {"kid": env.kid}, None)


def test_bound_envelope_matches_and_contradiction_raises() -> None:
    raw = _raw("conflict-allow")
    digest = hashlib.sha256(raw).hexdigest()
    res = _verify(raw)
    env = envelope_of(json.loads(raw), ADAPTERS)
    honest = {"kid": env.kid, "signature": env.signature}
    out = compare_live(digest, res, digest, res, honest, honest,
                       first_bytes=raw, second_bytes=raw)
    assert out.label == "agreement"
    forged = {"kid": "attacker-key", "signature": "00" * 64}
    with pytest.raises(ValueError, match="contradicts the verified envelope"):
        compare_live(digest, res, digest, res, forged, None,
                     first_bytes=raw, second_bytes=raw)


def test_terms_only_without_bytes_still_classifies() -> None:
    raw = _raw("conflict-allow")
    digest = hashlib.sha256(raw).hexdigest()
    res = _verify(raw)
    terms = AgreementRef.of_terms("agree-1", "v1", {"scope": "c30"})
    out = compare_live(digest, res, digest, res, terms, terms)
    assert out.label == "agreement"
