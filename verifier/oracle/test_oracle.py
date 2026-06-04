"""Oracle multi-format verification tests.

Each adapter must PASS a valid receipt and reject a tampered one (signature flip
and chain break). The conformance corpus runs end-to-end and every vector must
match its manifest outcome. Ed25519 is verified via the cryptography library; if
that library is absent the signature axis SKIPs and the signature-bearing
assertions are skipped rather than failing for the wrong reason.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_VERIFIER = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _VERIFIER)

from oracle import ADAPTERS, crypto, verify  # noqa: E402
from oracle.adapters.aerf import AerfAdapter  # noqa: E402
from oracle.adapters.asqav_native import AsqavNativeAdapter  # noqa: E402
from oracle.runner import run_corpus  # noqa: E402

_CORPUS = Path(_VERIFIER) / "conformance-vectors"

_ED25519_AVAILABLE = crypto.verify_ed25519(b"\x00" * 32, b"m", b"\x00" * 64)[0] != crypto.SKIPPED

requires_ed25519 = pytest.mark.skipif(
    not _ED25519_AVAILABLE, reason="cryptography not installed; Ed25519 verify SKIPs"
)


def _load(vec: str, name: str) -> dict:
    return json.loads((_CORPUS / vec / name).read_text())


def _provider(vec: str, fmt: str):
    fname = "jwks.json" if fmt == "asqav-native" else "keys.json"
    return _load(vec, fname)


def test_adapters_registered() -> None:
    names = [a.name for a in ADAPTERS]
    assert names == ["asqav-native", "aerf", "acta"]


def test_detection_picks_the_right_adapter() -> None:
    aerf = _load("aerf-01-genesis", "receipt.json")
    native = _load("asqav-01-genesis-permit", "receipt.json")
    assert AerfAdapter().detect(aerf) is True
    assert AerfAdapter().detect(native) is False
    assert AsqavNativeAdapter().detect(native) is True
    assert AsqavNativeAdapter().detect(aerf) is False


@requires_ed25519
def test_aerf_valid_receipt_passes() -> None:
    doc = _load("aerf-01-genesis", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("aerf-01-genesis", "aerf"))
    assert res.fmt == "aerf"
    assert res.verdict == "PASS"
    assert res.axis("signature").result == crypto.PASS


@requires_ed25519
def test_aerf_chain_link_passes() -> None:
    doc = _load("aerf-02-chain-link", "receipt.json")
    pred = _load("aerf-02-chain-link", "predecessor.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("aerf-02-chain-link", "aerf"), predecessor=pred)
    assert res.verdict == "PASS"
    assert res.axis("chain").result == crypto.PASS


@requires_ed25519
def test_aerf_tampered_evidence_fails_signature() -> None:
    doc = _load("aerf-03-tamper-evidence", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("aerf-03-tamper-evidence", "aerf"))
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


@requires_ed25519
def test_aerf_tampered_chain_fails_chain() -> None:
    doc = _load("aerf-04-tamper-chain", "receipt.json")
    pred = _load("aerf-04-tamper-chain", "predecessor.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("aerf-04-tamper-chain", "aerf"), predecessor=pred)
    assert res.verdict == "FAIL"
    assert res.axis("chain").result == crypto.FAIL


@requires_ed25519
def test_asqav_native_valid_receipt_passes() -> None:
    doc = _load("asqav-01-genesis-permit", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("asqav-01-genesis-permit", "asqav-native"))
    assert res.fmt == "asqav-native"
    assert res.verdict == "PASS"
    assert res.axis("signature").result == crypto.PASS


@requires_ed25519
def test_asqav_native_chain_link_passes() -> None:
    doc = _load("asqav-03-chain-link", "receipt.json")
    pred = _load("asqav-03-chain-link", "predecessor.json")
    res = verify(
        doc, ADAPTERS, key_provider=_provider("asqav-03-chain-link", "asqav-native"), predecessor=pred
    )
    assert res.verdict == "PASS"
    assert res.axis("chain").result == crypto.PASS


@requires_ed25519
def test_asqav_native_tampered_signature_fails() -> None:
    doc = _load("asqav-04-tamper-sig", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("asqav-04-tamper-sig", "asqav-native"))
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


def test_asqav_native_chain_break_fails_without_crypto() -> None:
    """A broken chain link FAILs the chain axis independent of the signature dep."""
    doc = _load("asqav-01-genesis-permit", "receipt.json")
    successor = json.loads(json.dumps(doc))
    successor["payload"]["previousReceiptHash"] = "f" * 64
    pred = _load("asqav-01-genesis-permit", "receipt.json")
    res = verify(successor, ADAPTERS, predecessor=pred)
    assert res.axis("chain").result == crypto.FAIL


def test_unknown_format_fails_closed() -> None:
    res = verify({"hello": "world"}, ADAPTERS)
    assert res.fmt == "unknown"
    assert res.verdict == "FAIL"


def test_signature_skips_downgrade_to_incomplete() -> None:
    """No key provider means the signature cannot be checked; never a PASS."""
    doc = _load("aerf-01-genesis", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider={})
    assert res.axis("signature").result == crypto.SKIPPED
    assert res.verdict == "INCOMPLETE"


@requires_ed25519
def test_corpus_runs_and_every_vector_matches() -> None:
    results = run_corpus(_CORPUS)
    assert len(results) == 12
    failures = [r for r in results if not r.ok]
    assert not failures, f"corpus mismatches: {[(r.dir, r.actual_verdict) for r in failures]}"


# --- ACTA adapter + cross-format-confusion gate (rule 4.9 clause 3) ---

from oracle.adapters.acta import ActaAdapter  # noqa: E402


def _acta_keys(vec: str):
    return _load(vec, "acta-keys.json")


@requires_ed25519
def test_acta_valid_genesis_passes() -> None:
    doc = _load("acta-01-genesis", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_acta_keys("acta-01-genesis"))
    assert res.fmt == "acta"
    assert res.verdict == "PASS"
    assert res.axis("signature").result == crypto.PASS


@requires_ed25519
def test_acta_chain_link_passes() -> None:
    doc = _load("acta-02-chain-link", "receipt.json")
    pred = _load("acta-02-chain-link", "predecessor.json")
    res = verify(doc, ADAPTERS, key_provider=_acta_keys("acta-02-chain-link"), predecessor=pred)
    assert res.verdict == "PASS"
    assert res.axis("chain").result == crypto.PASS


@requires_ed25519
def test_acta_tampered_signature_fails() -> None:
    doc = _load("acta-03-tamper-sig", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_acta_keys("acta-03-tamper-sig"))
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


@requires_ed25519
def test_acta_commitment_mode_fails_not_false_passes() -> None:
    """A commitment-mode receipt (signs SHA-256(JCS)) must FAIL the baseline verifier."""
    doc = _load("acta-05-commitment-mode-unsupported", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_acta_keys("acta-05-commitment-mode-unsupported"))
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


def test_acta_and_asqav_native_are_mutually_exclusive() -> None:
    """Asqav-native profiles ACTA, so detection must never let both claim one receipt."""
    acta = _load("acta-01-genesis", "receipt.json")
    native = _load("asqav-01-genesis-permit", "receipt.json")
    assert ActaAdapter().detect(acta) is True
    assert AsqavNativeAdapter().detect(acta) is False
    assert AsqavNativeAdapter().detect(native) is True
    assert ActaAdapter().detect(native) is False


@requires_ed25519
def test_cross_format_confusion_asqav_with_hex_sig_does_not_verify_as_native() -> None:
    """An Asqav-shaped receipt with a hex sig routes to ACTA and FAILs - never a false native PASS."""
    native = _load("asqav-01-genesis-permit", "receipt.json")
    forged = json.loads(json.dumps(native))
    forged.pop("anchors", None)
    forged["signature"]["sig"] = "ab" * 64  # 128-char lowercase hex (ACTA-shaped)
    res = verify(forged, ADAPTERS, key_provider=_acta_keys("acta-01-genesis"))
    assert res.fmt == "acta"
    assert res.verdict != "PASS"
    # The unmodified native receipt still routes to asqav-native.
    assert verify(native, ADAPTERS, key_provider=_provider("asqav-01-genesis-permit", "asqav-native")).fmt == "asqav-native"


@requires_ed25519
def test_acta_cross_format_predecessor_fails_chain() -> None:
    """An ACTA successor with an asqav-native predecessor is not a valid chain link."""
    succ = _load("acta-02-chain-link", "receipt.json")
    foreign_pred = _load("asqav-01-genesis-permit", "receipt.json")
    res = verify(succ, ADAPTERS, key_provider=_acta_keys("acta-02-chain-link"), predecessor=foreign_pred)
    assert res.axis("chain").result == crypto.FAIL


@requires_ed25519
def test_acta_genesis_spoof_breaks_signature() -> None:
    """Deleting previousReceiptHash to fake genesis breaks the signature (it is signed)."""
    succ = _load("acta-02-chain-link", "receipt.json")
    spoof = json.loads(json.dumps(succ))
    del spoof["payload"]["previousReceiptHash"]
    res = verify(spoof, ADAPTERS, key_provider=_acta_keys("acta-02-chain-link"))
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL
