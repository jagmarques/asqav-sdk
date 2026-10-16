"""Digest-to-bytes binding over live compare (488 C24, C5 core).

A stated digest classifies as-is by default; passing the raw receipt
bytes binds it (mismatch raises instead of classifying). Retention uses
the existing store: both live assessments are recorded first, and the
compare runs over the retained inputs' re-execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography", reason="live verify needs the cryptography lib")

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify
from asqav.verifier.oracle.assessment_store import AssessmentStore
from asqav.verifier.oracle.live_compare import compare_live
from asqav.verifier.oracle.retain import record_verify_result

_FIX = Path(__file__).resolve().parents[2] / "verifier" / "signed-conflict-fixtures"
_CTX = {"evaluation_time": "2026-10-16T12:00:00Z", "evaluator": "c24-probe"}


def _raw(name: str) -> bytes:
    return (_FIX / f"{name}.json").read_bytes()


def _verify(raw: bytes):
    return verify(json.loads(raw), ADAPTERS, context=VerificationContext(**_CTX))


def test_bound_bytes_agree() -> None:
    raw = _raw("conflict-allow")
    digest = hashlib.sha256(raw).hexdigest()
    out = compare_live(digest, _verify(raw), digest, _verify(raw),
                       first_bytes=raw, second_bytes=raw)
    assert out.label == "agreement" and out.differing == ()


def test_bound_conflict_stays_split() -> None:
    allow, deny = _raw("conflict-allow"), _raw("conflict-deny")
    digests = [hashlib.sha256(b).hexdigest() for b in (allow, deny)]
    out = compare_live(digests[0], _verify(allow), digests[1], _verify(deny),
                       first_bytes=allow, second_bytes=deny)
    assert out.label == "different_subject" and out.differing == ()


def test_mismatched_digest_raises_instead_of_classifying() -> None:
    allow, deny = _raw("conflict-allow"), _raw("conflict-deny")
    good = hashlib.sha256(allow).hexdigest()
    with pytest.raises(ValueError, match="first digest"):
        compare_live("00" * 32, _verify(allow), good, _verify(allow),
                     first_bytes=allow, second_bytes=allow)
    with pytest.raises(ValueError, match="second digest"):
        compare_live(good, _verify(allow), good, _verify(deny),
                     first_bytes=allow, second_bytes=deny)


def test_compare_runs_over_retained_inputs() -> None:
    allow, deny = _raw("conflict-allow"), _raw("conflict-deny")
    store = AssessmentStore()
    for raw in (allow, deny):
        record_verify_result(store, hashlib.sha256(raw).hexdigest(), _verify(raw))
    assert len(store) == 2
    replayed = [
        verify(json.loads(raw), ADAPTERS,
               context=VerificationContext(**store.replay(hashlib.sha256(raw).hexdigest()).context))
        for raw in (allow, deny)
    ]
    assert [r.verdict for r in replayed] == ["verified", "verified"]
    digests = [hashlib.sha256(b).hexdigest() for b in (allow, deny)]
    out = compare_live(digests[0], replayed[0], digests[1], replayed[1],
                       first_bytes=allow, second_bytes=deny)
    assert out.label == "different_subject"
