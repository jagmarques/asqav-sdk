"""Strict live compare: the entry verifies the bytes itself (488 C26, C29).

No digest or result parameter exists, so no unbound or foreign-result call
is expressible; classification always runs over re-verified bytes. All
oracle output here is live.

C29 binds the verified envelope: the entry derives kid/signature from the
same bytes it verifies (no envelope parameter exists, so no stated envelope
is expressible except caller terms); a caller-stated envelope contradicting
the bytes raises instead of classifying.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography", reason="live verify needs the cryptography lib")

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify
from asqav.verifier.oracle.agreement import envelope_of
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


def test_strict_derives_envelope_from_the_verified_bytes() -> None:
    # Same bytes agree with the bytes-derived envelope on both sides; the
    # entry takes no envelope parameter, so forgery is unexpressible here.
    raw = _raw("conflict-allow")
    out = compare_live_strict(raw, raw, context=VerificationContext(**_CTX))
    assert out.label == "agreement" and out.differing == ()
    names = inspect.signature(compare_live_strict).parameters
    assert not any("envelope" in name or "kid" in name for name in names)


def test_strict_forged_envelope_raises() -> None:
    # A caller-stated envelope contradicting the bytes raises (C24's digest
    # rule, applied to the envelope); a matching one is redundant but fine.
    raw = _raw("conflict-allow")
    forged = {"kid": "attacker-key", "signature": "00" * 64}
    with pytest.raises(ValueError):
        compare_live_strict(raw, raw, forged, None, context=VerificationContext(**_CTX))
    env = envelope_of(json.loads(raw), ADAPTERS)
    honest = {"kid": env.kid, "signature": env.signature}
    out = compare_live_strict(raw, raw, honest, None, context=VerificationContext(**_CTX))
    assert out.label == "agreement" and out.differing == ()


def test_strict_conflict_label_unchanged_by_envelope_dims() -> None:
    # Evidence short-circuit still governs: differing envelopes never leak
    # past a different-subject split (differing stays empty, label holds).
    allow, deny = _raw("conflict-allow"), _raw("conflict-deny")
    out = compare_live_strict(allow, deny, context=VerificationContext(**_CTX))
    assert out.label == "different_subject" and out.differing == ()
