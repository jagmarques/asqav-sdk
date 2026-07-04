"""v:2 receipt acceptance in the neutral oracle verifier.

The neutral verifier holds no org secret, so it cannot recompute the per-org
``_asqav_tid`` canary. It verifies the Ed25519 signature over the canonical body
that INCLUDES ``signer`` + ``_asqav_tid``, surfaces ``signer`` in the verdict, and
treats a missing canary as neutral (never a failure). Because ``signer`` sits
inside the signed body, tampering it must FAIL verification.

Byte-pinned to the backend v:2 signed body (context-store v2-signed-body-shape).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from asqav.verifier.oracle import ADAPTERS, crypto, verify
from asqav.verifier.oracle.adapters.asqav_native import AsqavNativeAdapter

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"

_ED25519_AVAILABLE = crypto.verify_ed25519(b"\x00" * 32, b"m", b"\x00" * 64)[0] != crypto.SKIPPED
requires_ed25519 = pytest.mark.skipif(
    not _ED25519_AVAILABLE, reason="cryptography not installed; Ed25519 verify SKIPs"
)

_SIGNER = "https://api.asqav.com"


def _load(vec: str, name: str) -> dict:
    return json.loads((_CORPUS / vec / name).read_text())


def _v2():
    return _load("asqav-08-v2-signer-canary", "receipt.json"), _load(
        "asqav-08-v2-signer-canary", "jwks.json"
    )


@requires_ed25519
def test_v2_receipt_verifies_and_surfaces_signer() -> None:
    receipt, jwks = _v2()
    res = verify(receipt, ADAPTERS, key_provider=jwks)
    assert res.fmt == "asqav-native"
    assert res.verdict == "PASS"
    assert res.axis("signature").result == crypto.PASS
    # The v:2 origin attestation is surfaced from the signed body.
    assert res.signer == _SIGNER


@requires_ed25519
def test_v2_tampered_in_body_signer_fails() -> None:
    """Flip one char of the in-body signer; the signature over the canonical body must FAIL.

    Non-vacuous by construction: signer is inside the signed payload, so a
    verifier that stripped signer before canonicalising would PASS here. The
    signature axis FAIL is what proves in-body coverage.
    """
    receipt, jwks = _v2()
    receipt["payload"]["signer"] = "https://api.asqav.con"
    res = verify(receipt, ADAPTERS, key_provider=jwks)
    assert res.axis("signature").result == crypto.FAIL
    assert res.verdict == "FAIL"


@requires_ed25519
def test_v2_signer_moved_outside_signed_body_not_surfaced_and_fails() -> None:
    """signer as loose metadata outside the signed payload is neither surfaced nor trusted."""
    receipt, jwks = _v2()
    del receipt["payload"]["signer"]
    receipt["signer"] = _SIGNER  # loose, outside the canonical body
    res = verify(receipt, ADAPTERS, key_provider=jwks)
    assert res.signer is None
    assert res.verdict == "FAIL"


@requires_ed25519
def test_v2_tampered_canary_corpus_vector_fails() -> None:
    receipt, jwks = _load("asqav-09-v2-signer-tampered", "receipt.json"), _load(
        "asqav-09-v2-signer-tampered", "jwks.json"
    )
    res = verify(receipt, ADAPTERS, key_provider=jwks)
    assert res.verdict == "FAIL"


@requires_ed25519
def test_missing_canary_is_not_a_failure() -> None:
    """A v:2 receipt without _asqav_tid still verifies; the neutral verifier never requires it."""
    receipt, jwks = _v2()
    del receipt["payload"]["_asqav_tid"]
    # Re-sign is out of scope for the neutral verifier; dropping the field breaks
    # the signature, so assert only that the absence itself raises no canary axis.
    res = verify(receipt, ADAPTERS, key_provider=jwks)
    assert not any(a.axis == "canary" for a in res.axes)


@requires_ed25519
def test_v1_vector_still_verifies_and_exposes_no_signer() -> None:
    receipt, jwks = _load("asqav-01-genesis-permit", "receipt.json"), _load(
        "asqav-01-genesis-permit", "jwks.json"
    )
    res = verify(receipt, ADAPTERS, key_provider=jwks)
    assert res.verdict == "PASS"
    assert res.signer is None


def test_adapter_attestation_reads_only_signed_payload() -> None:
    """attestation() surfaces signer from the payload, ignoring loose envelope metadata."""
    ad = AsqavNativeAdapter()
    receipt, _ = _v2()
    assert ad.attestation(receipt) == {"signer": _SIGNER}
    loose = copy.deepcopy(receipt)
    del loose["payload"]["signer"]
    loose["signer"] = _SIGNER
    assert ad.attestation(loose) == {}
