"""Envelope binding in attribute_live_split (488 C31).

Same default-binding as C30, extended to the rerun sides: a stated
kid/signature without its bytes raises; bytes present bind.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography", reason="live verify needs the cryptography lib")

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify
from asqav.verifier.oracle.agreement import envelope_of
from asqav.verifier.oracle.live_compare import attribute_live_split

_FIX = Path(__file__).resolve().parents[2] / "verifier" / "signed-conflict-fixtures"
_CTX = {"evaluation_time": "2026-10-16T12:00:00Z", "evaluator": "c31-probe"}


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
        attribute_live_split(digest, res, digest, res, None, None, honest, None)
    with pytest.raises(ValueError, match="rerun_a envelope without bytes"):
        attribute_live_split(digest, res, digest, res, res, None, honest, None,
                             first_bytes=raw, second_bytes=raw)


def test_bound_rerun_bytes_bind_and_contradiction_raises() -> None:
    raw = _raw("conflict-allow")
    digest = hashlib.sha256(raw).hexdigest()
    res = _verify(raw)
    env = envelope_of(json.loads(raw), ADAPTERS)
    honest = {"kid": env.kid, "signature": env.signature}
    out, _ = attribute_live_split(digest, res, digest, res, res, res, honest, honest,
                                  first_bytes=raw, second_bytes=raw,
                                  rerun_a_bytes=raw, rerun_b_bytes=raw)
    assert out.label == "agreement"
    forged = {"kid": "attacker-key", "signature": "00" * 64}
    with pytest.raises(ValueError, match="contradicts the verified envelope"):
        attribute_live_split(digest, res, digest, res, None, None, forged, None,
                             first_bytes=raw, second_bytes=raw)
