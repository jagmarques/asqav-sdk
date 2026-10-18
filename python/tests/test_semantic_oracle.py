"""Independent verdict semantics for the oracle (criterion 574 D5).

The aggregation in ``oracle.semantic`` is derived from the normative
rules (418/438/426 + chain tolerance), not ported from any verifier.
These tests pin that derivation three ways: the vocabulary matches the
core's (shared surface, separate logic); a hand-derived truth table
covers every rule including the cases a naive port gets wrong (expiry
failure still verifies, chain skip tolerated, keying never rescues a
failure); and real corpus vectors keep their verdicts through the new
fold. A naive "any FAIL is invalid, any SKIP blocks, keyed always wins"
fold fails four of these patterns.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from asqav.verifier.oracle import ADAPTERS, VerificationContext, crypto, verify
from asqav.verifier.oracle import semantic
from asqav.verifier.oracle.core import (
    FAILURE_INVALID,
    FAILURE_UNVERIFIABLE,
    VERDICT_UNVERIFIED,
    VERDICT_VERIFIED,
    VERDICT_VERIFIED_KEYED,
    AxisResult,
)
from asqav.verifier.oracle.runner import _key_provider as _runner_key_provider
from asqav.verifier.oracle.runner import _load as _runner_load

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"


def _ax(axis: str, result: str, failure_class: str | None) -> AxisResult:
    return AxisResult(axis=axis, result=result, note=f"{axis} {result}", failure_class=failure_class)


def _passing(*axes: str) -> list[AxisResult]:
    return [_ax(a, crypto.PASS, None) for a in axes]


_BASE = ("structure", "signature", "chain", "seq")


def test_vocabulary_matches_the_core_surface() -> None:
    assert semantic.VERDICT_VERIFIED == VERDICT_VERIFIED
    assert semantic.VERDICT_VERIFIED_KEYED == VERDICT_VERIFIED_KEYED
    assert semantic.VERDICT_UNVERIFIED == VERDICT_UNVERIFIED
    assert semantic.FAILURE_INVALID == FAILURE_INVALID
    assert semantic.FAILURE_UNVERIFIABLE == FAILURE_UNVERIFIABLE


@pytest.mark.parametrize(
    ("axes", "keyed", "verdict", "failure_class"),
    [
        (_passing(*_BASE), False, "verified", None),
        (_passing(*_BASE), True, "verified_keyed", None),
        # 426: expiry reports but never folds, even keyed.
        (_passing(*_BASE) + [_ax("expiry", crypto.FAIL, FAILURE_INVALID)], False, "verified", None),
        (_passing(*_BASE) + [_ax("expiry", crypto.FAIL, FAILURE_INVALID)], True, "verified_keyed", None),
        # Chain skip is the expected genesis shape, not a gap.
        (_passing("structure", "signature", "seq") + [_ax("chain", crypto.SKIPPED, FAILURE_UNVERIFIABLE)], False, "verified", None),
        # Any other skip blocks.
        (_passing("structure", "chain", "seq") + [_ax("signature", crypto.SKIPPED, FAILURE_UNVERIFIABLE)], False, "unverified", "unverifiable"),
        # A proven binding failure is invalid; anything else is unverifiable.
        (_passing("structure", "chain", "seq") + [_ax("signature", crypto.FAIL, FAILURE_INVALID)], False, "unverified", "invalid"),
        (_passing("structure", "chain", "seq") + [_ax("signature", crypto.FAIL, FAILURE_UNVERIFIABLE)], False, "unverified", "unverifiable"),
        # Invalid dominates: strongest ground wins over weaker ones and skips.
        (
            _passing("structure", "seq")
            + [_ax("signature", crypto.FAIL, FAILURE_INVALID), _ax("chain", crypto.FAIL, FAILURE_UNVERIFIABLE)],
            False,
            "unverified",
            "invalid",
        ),
        (
            _passing("structure", "seq")
            + [_ax("signature", crypto.FAIL, FAILURE_INVALID), _ax("chain", crypto.SKIPPED, FAILURE_UNVERIFIABLE)],
            False,
            "unverified",
            "invalid",
        ),
        # Keying never rescues a failure.
        (_passing("structure", "chain", "seq") + [_ax("signature", crypto.FAIL, FAILURE_INVALID)], True, "unverified", "invalid"),
    ],
)
def test_truth_table(axes: list[AxisResult], keyed: bool, verdict: str, failure_class: str | None) -> None:
    assert semantic.fold_independent(axes, keyed=keyed) == (verdict, failure_class)


def _verify_vec(vec: str) -> tuple[str, str | None]:
    manifest = json.loads((_CORPUS / vec / "expected.json").read_text(encoding="utf-8"))
    fmt = manifest["format"]
    receipt = _runner_load(_CORPUS / vec / "receipt.json")
    try:
        predecessor = _runner_load(_CORPUS / vec / "predecessor.json")
    except FileNotFoundError:
        predecessor = None
    try:
        envelope = _runner_load(_CORPUS / vec / "originating_envelope.json")
    except FileNotFoundError:
        envelope = None
    res = verify(
        receipt,
        ADAPTERS,
        key_provider=_runner_key_provider(_CORPUS / vec, fmt),
        predecessor=predecessor,
        context=VerificationContext(envelope),
    )
    return res.verdict, res.failure_class


@pytest.mark.parametrize(
    ("vec", "verdict", "failure_class"),
    [
        ("asqav-01-genesis-permit", VERDICT_VERIFIED, None),
        ("asqav-55-heartbeat", VERDICT_VERIFIED, None),
        ("asqav-18-seq-gap", VERDICT_UNVERIFIED, FAILURE_INVALID),
    ],
)
def test_real_vectors_keep_their_verdicts(vec: str, verdict: str, failure_class: str | None) -> None:
    assert _verify_vec(vec) == (verdict, failure_class)
