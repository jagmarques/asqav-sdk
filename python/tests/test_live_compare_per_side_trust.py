"""Per-side trust contexts in the strict entry (488 C33, C5 remainder)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography", reason="live verify needs the cryptography lib")

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify
from asqav.verifier.oracle.live_compare import compare_live_strict

_FIX = Path(__file__).resolve().parents[2] / "verifier" / "signed-conflict-fixtures"


def _raw(name: str) -> bytes:
    return (_FIX / f"{name}.json").read_bytes()


def _ctx(**overrides):
    base = {"evaluation_time": "2026-10-16T12:00:00Z", "evaluator": "c33-probe"}
    base.update(overrides)
    return VerificationContext(**base)


def test_shared_context_unchanged() -> None:
    raw = _raw("conflict-allow")
    out = compare_live_strict(raw, raw, context=_ctx())
    assert out.label == "agreement" and out.differing == ()


def test_per_side_trust_roots_surface_in_differing() -> None:
    raw = _raw("conflict-allow")
    out = compare_live_strict(
        raw, raw,
        first_context=_ctx(trust_roots=["root-a"]),
        second_context=_ctx(trust_roots=["root-b"]))
    assert "trust_roots" in out.differing


def test_omitted_side_falls_back_to_shared_context() -> None:
    raw = _raw("conflict-allow")
    out = compare_live_strict(raw, raw, context=_ctx(),
                              first_context=_ctx(trust_roots=["root-a"]))
    assert "trust_roots" in out.differing
