"""Gates for the seq continuity axis - the SDK half of the omission class.

A gap in the counter proves receipts were withheld without needing the withheld
receipts. The no-SKIPPED gate matters most: fold_verdict blocks on a non-chain
SKIPPED, so a counter-less receipt would go from verified to unverified.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from asqav.verifier.oracle import ADAPTERS, crypto, verify

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"

_ED25519_AVAILABLE = crypto.verify_ed25519(b"\x00" * 32, b"m", b"\x00" * 64)[0] != crypto.SKIPPED
requires_ed25519 = pytest.mark.skipif(
    not _ED25519_AVAILABLE, reason="cryptography not installed; Ed25519 verify SKIPs"
)


def _load(vec: str, name: str) -> dict:
    return json.loads((_CORPUS / vec / name).read_text())


def _provider(vec: str) -> dict:
    return _load(vec, "jwks.json")


def _with_seq(doc: dict, seq: object) -> dict:
    # Injection breaks the signature, so assertions read the seq axis directly
    # unless the verdict itself is under test on an untouched receipt.
    out = json.loads(json.dumps(doc))
    out["payload"]["seq"] = seq
    return out


@requires_ed25519
def test_a_receipt_carrying_no_seq_still_verifies() -> None:
    """A real receipt with no counter must stay verified, not blocked by a SKIPPED."""
    doc = _load("asqav-01-genesis-permit", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("asqav-01-genesis-permit"))
    assert res.verdict == "verified"
    assert res.failure_class is None
    seq = res.axis("seq")
    assert seq is not None, "the seq axis must always be emitted, never dropped"
    assert seq.result == crypto.PASS
    assert "not part of a counted series" in seq.note


def test_the_seq_axis_never_reports_skipped_across_the_whole_corpus() -> None:
    """Corpus-wide, so a future branch that forgets the no-SKIPPED rule is caught."""
    skipped = []
    for vec_dir in sorted(p for p in _CORPUS.iterdir() if p.is_dir()):
        receipt = vec_dir / "receipt.json"
        if not receipt.exists():
            continue
        doc = json.loads(receipt.read_text())
        pred_path = vec_dir / "predecessor.json"
        pred = json.loads(pred_path.read_text()) if pred_path.exists() else None
        res = verify(doc, ADAPTERS, predecessor=pred)
        axis = res.axis("seq")
        if axis is not None and axis.result == crypto.SKIPPED:
            skipped.append((vec_dir.name, axis.note))
    assert not skipped, f"seq axis reported a blocking SKIPPED: {skipped}"


def test_a_contiguous_counter_passes() -> None:
    pred = _with_seq(_load("asqav-03-chain-link", "predecessor.json"), 7)
    doc = _with_seq(_load("asqav-03-chain-link", "receipt.json"), 8)
    axis = verify(doc, ADAPTERS, predecessor=pred).axis("seq")
    assert axis.result == crypto.PASS
    assert "seq 8 follows predecessor 7" in axis.note


def test_a_gap_fails_and_counts_the_withheld_receipts() -> None:
    """The omission signal: 7 -> 11 means three receipts were withheld."""
    pred = _with_seq(_load("asqav-03-chain-link", "predecessor.json"), 7)
    doc = _with_seq(_load("asqav-03-chain-link", "receipt.json"), 11)
    res = verify(doc, ADAPTERS, predecessor=pred)
    axis = res.axis("seq")
    assert axis.result == crypto.FAIL
    assert "3 receipt(s) withheld between 7 and 11" in axis.note
    # A proven omission is a binding failure, not an incomplete recompute.
    assert axis.failure_class == "invalid"
    assert res.verdict == "unverified"


def test_a_repeated_or_rewound_counter_fails() -> None:
    for seq in (7, 6, 1):
        pred = _with_seq(_load("asqav-03-chain-link", "predecessor.json"), 7)
        doc = _with_seq(_load("asqav-03-chain-link", "receipt.json"), seq)
        axis = verify(doc, ADAPTERS, predecessor=pred).axis("seq")
        assert axis.result == crypto.FAIL, f"seq {seq} after 7 must fail"
        assert "not monotonic" in axis.note


@pytest.mark.parametrize("bad", [True, False, "8", 0, -1, 1.5, [8], {"n": 8}, None])
def test_a_malformed_counter_is_refused(bad: object) -> None:
    """None is legal absence; True matters because bool is an int subclass."""
    doc = _with_seq(_load("asqav-01-genesis-permit", "receipt.json"), bad)
    axis = verify(doc, ADAPTERS).axis("seq")
    if bad is None:
        assert axis.result == crypto.PASS
    else:
        assert axis.result == crypto.FAIL, f"{bad!r} must not pass as a counter"
        assert "malformed seq" in axis.note


def test_a_counter_with_no_predecessor_supplied_passes_with_a_note() -> None:
    doc = _with_seq(_load("asqav-01-genesis-permit", "receipt.json"), 4)
    axis = verify(doc, ADAPTERS).axis("seq")
    assert axis.result == crypto.PASS
    assert "no predecessor supplied" in axis.note


def test_a_predecessor_carrying_no_counter_passes_with_a_note() -> None:
    """The migration case: the predecessor was minted before the counter shipped."""
    pred = _load("asqav-03-chain-link", "predecessor.json")
    assert "seq" not in pred["payload"], "fixture must genuinely lack a counter"
    doc = _with_seq(_load("asqav-03-chain-link", "receipt.json"), 2)
    axis = verify(doc, ADAPTERS, predecessor=pred).axis("seq")
    assert axis.result == crypto.PASS
    assert "predecessor carries no seq" in axis.note


def test_a_malformed_predecessor_counter_fails() -> None:
    pred = _with_seq(_load("asqav-03-chain-link", "predecessor.json"), "seven")
    doc = _with_seq(_load("asqav-03-chain-link", "receipt.json"), 8)
    axis = verify(doc, ADAPTERS, predecessor=pred).axis("seq")
    assert axis.result == crypto.FAIL
    assert "malformed predecessor seq" in axis.note


def test_counters_are_never_compared_across_formats() -> None:
    """A foreign receipt's counter is not this series; comparing would fake a gap."""
    pred = _load("acta-01-genesis", "receipt.json")
    pred["seq"] = 99
    doc = _with_seq(_load("asqav-01-genesis-permit", "receipt.json"), 2)
    axis = verify(doc, ADAPTERS, predecessor=pred).axis("seq")
    assert axis.result == crypto.PASS
    assert "different receipt format" in axis.note


def test_a_hash_mode_receipt_cannot_claim_a_counter() -> None:
    """Hash mode signs flat fields only, so a pasted seq is an unsigned claim."""
    doc = _load("asqav-05-hash-mode-prod", "receipt.json")
    clean = verify(doc, ADAPTERS, key_provider=_provider("asqav-05-hash-mode-prod"))
    assert clean.axis("structure").result == crypto.PASS
    assert clean.axis("seq").result == crypto.PASS

    forged = json.loads(json.dumps(doc))
    forged["seq"] = 5
    res = verify(forged, ADAPTERS, key_provider=_provider("asqav-05-hash-mode-prod"))
    assert res.axis("structure").result == crypto.FAIL
    assert "signature does not cover" in res.axis("structure").note
    # The counter itself is never read off an unsigned field set.
    assert res.axis("seq").result == crypto.PASS
    assert "not part of a counted series" in res.axis("seq").note
    assert res.verdict == "unverified"
