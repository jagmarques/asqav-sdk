"""Strict live compare: the entry verifies the bytes itself (488 C26).

No digest or result parameter exists, so no unbound or foreign-result call
is expressible; classification always runs over re-verified bytes. All
oracle output here is live.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography", reason="live verify needs the cryptography lib")

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify
from asqav.verifier.oracle.live_compare import compare_live, compare_live_strict

_FIX = Path(__file__).resolve().parents[2] / "verifier" / "signed-conflict-fixtures"
_CTX = {"evaluation_time": "2026-10-16T12:00:00Z", "evaluator": "c26-probe"}


def _raw(name: str) -> bytes:
    return (_FIX / f"{name}.json").read_bytes()


def _verify(raw: bytes):
    return verify(json.loads(raw), ADAPTERS, context=VerificationContext(**_CTX))


def test_strict_agrees_on_same_bytes() -> None:
    raw = _raw("conflict-allow")
    out = compare_live_strict(raw, raw, context=VerificationContext(**_CTX))
    assert out.label == "agreement" and out.differing == ()


def test_strict_conflict_stays_split() -> None:
    allow, deny = _raw("conflict-allow"), _raw("conflict-deny")
    out = compare_live_strict(allow, deny, context=VerificationContext(**_CTX))
    assert out.label == "different_subject"


def test_strict_matches_live_pipeline() -> None:
    allow, deny = _raw("conflict-allow"), _raw("conflict-deny")
    ctx = VerificationContext(**_CTX)
    strict_out = compare_live_strict(allow, deny, context=ctx)
    digests = [hashlib.sha256(b).hexdigest() for b in (allow, deny)]
    pipe_out = compare_live(digests[0], _verify(allow), digests[1], _verify(deny),
                            first_bytes=allow, second_bytes=deny)
    assert (strict_out.label, strict_out.differing) == (pipe_out.label, pipe_out.differing)


def test_no_result_or_digest_channel() -> None:
    names = inspect.signature(compare_live_strict).parameters
    assert not any("digest" in name or "result" in name for name in names)
    allow, deny = _raw("conflict-allow"), _raw("conflict-deny")
    with pytest.raises(TypeError):
        compare_live_strict(allow, deny, first_result=_verify(allow))  # type: ignore[call-arg]
