"""A reused buffer must never read an earlier verification result (criterion 742)."""

from __future__ import annotations

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat  # noqa: E402

from asqav.verifier.oracle import crypto  # noqa: E402


def _signed(message: bytes) -> tuple[bytes, bytes]:
    key = Ed25519PrivateKey.generate()
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw), key.sign(message)


def test_a_message_rewritten_in_place_is_verified_again() -> None:
    pk, sig = _signed(b"receipt A")
    msg = bytearray(b"receipt A")
    assert crypto.verify_signature("Ed25519", pk, msg, sig)[0] == "PASS"
    msg[:] = b"receipt B"
    assert crypto.verify_signature("Ed25519", pk, msg, sig)[0] == "FAIL"
    assert crypto.verify_signature("Ed25519", pk, b"receipt B", sig)[0] == "FAIL"


def test_a_forged_signature_after_a_genuine_one_is_refused() -> None:
    pk, sig = _signed(b"receipt A")
    assert crypto.verify_signature("Ed25519", pk, bytearray(b"receipt A"), sig)[0] == "PASS"
    forged = bytearray(sig)
    forged[0] ^= 0x01
    assert crypto.verify_signature("Ed25519", pk, bytearray(b"receipt A"), bytes(forged))[0] == "FAIL"


def test_the_same_input_still_gives_the_same_answer() -> None:
    pk, sig = _signed(b"receipt A")
    first = crypto.verify_signature("Ed25519", pk, b"receipt A", sig)
    assert crypto.verify_signature("Ed25519", pk, b"receipt A", sig) == first == ("PASS", "signature valid")
