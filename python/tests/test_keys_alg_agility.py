"""Algorithm agility on the SDK offline signing path."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav.keys import (
    ALGORITHM_ED25519,
    ALGORITHM_ES256,
    ALGORITHM_ML_DSA_65,
    SUPPORTED_ALGORITHMS,
    LocalKeypair,
    _check_algorithm,
    generate_local_keypair,
)

# === Constants and namespace ===


def test_supported_algorithms_set() -> None:
    assert SUPPORTED_ALGORITHMS == frozenset(
        {ALGORITHM_ED25519, ALGORITHM_ES256}
    )


def test_constants_match_spec_strings() -> None:
    """Identifier strings match the cloud's SignatureRecord.algorithm column."""
    assert ALGORITHM_ML_DSA_65 == "ml-dsa-65"
    assert ALGORITHM_ED25519 == "ed25519"
    assert ALGORITHM_ES256 == "es256"


def test_re_exports_at_package_root() -> None:
    assert asqav.ALGORITHM_ML_DSA_65 == ALGORITHM_ML_DSA_65
    assert asqav.ALGORITHM_ED25519 == ALGORITHM_ED25519
    assert asqav.ALGORITHM_ES256 == ALGORITHM_ES256
    assert asqav.SUPPORTED_ALGORITHMS == SUPPORTED_ALGORITHMS
    assert asqav.LocalKeypair is LocalKeypair
    assert asqav.generate_local_keypair is generate_local_keypair


# === _check_algorithm ===


def test_check_algorithm_normalises_case() -> None:
    assert _check_algorithm("ED25519") == "ed25519"
    assert _check_algorithm("ES256") == "es256"


def test_check_algorithm_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported_algorithm"):
        _check_algorithm("rsa")


def test_check_algorithm_rejects_ml_dsa_65() -> None:
    """ml-dsa-65 keypair generation is server-side only."""
    with pytest.raises(ValueError, match="unsupported_algorithm"):
        _check_algorithm("ml-dsa-65")


# === Ed25519 ===


def test_ed25519_keypair_generates_pem() -> None:
    pytest.importorskip("cryptography")
    kp = generate_local_keypair("ed25519")
    assert kp.algorithm == "ed25519"
    assert kp.private_key_pem.startswith(b"-----BEGIN PRIVATE KEY-----")
    assert kp.public_key_pem.startswith(b"-----BEGIN PUBLIC KEY-----")


def test_ed25519_public_key_round_trips() -> None:
    """The PEM-encoded public key parses back to a valid Ed25519 key."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    kp = generate_local_keypair("ed25519")
    pub = serialization.load_pem_public_key(kp.public_key_pem)
    assert isinstance(pub, ed25519.Ed25519PublicKey)


def test_ed25519_signs_and_verifies() -> None:
    """End-to-end: generate, sign with private, verify with public."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    kp = generate_local_keypair("ed25519")
    priv = serialization.load_pem_private_key(
        kp.private_key_pem, password=None
    )
    assert isinstance(priv, ed25519.Ed25519PrivateKey)
    msg = b"compliance receipt envelope bytes"
    sig = priv.sign(msg)

    pub = serialization.load_pem_public_key(kp.public_key_pem)
    pub.verify(sig, msg)  # raises on failure


# === ES256 ===


def test_es256_keypair_generates_pem() -> None:
    pytest.importorskip("cryptography")
    kp = generate_local_keypair("es256")
    assert kp.algorithm == "es256"
    assert kp.private_key_pem.startswith(b"-----BEGIN PRIVATE KEY-----")
    assert kp.public_key_pem.startswith(b"-----BEGIN PUBLIC KEY-----")


def test_es256_curve_is_p256() -> None:
    """ES256 means ECDSA over P-256 per RFC 7518."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    kp = generate_local_keypair("es256")
    pub = serialization.load_pem_public_key(kp.public_key_pem)
    assert isinstance(pub, ec.EllipticCurvePublicKey)
    assert isinstance(pub.curve, ec.SECP256R1)


def test_es256_signs_and_verifies() -> None:
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    kp = generate_local_keypair("es256")
    priv = serialization.load_pem_private_key(
        kp.private_key_pem, password=None
    )
    assert isinstance(priv, ec.EllipticCurvePrivateKey)
    msg = b"compliance receipt envelope bytes"
    sig = priv.sign(msg, ec.ECDSA(hashes.SHA256()))

    pub = serialization.load_pem_public_key(kp.public_key_pem)
    pub.verify(sig, msg, ec.ECDSA(hashes.SHA256()))


# === ML-DSA-65 is server-side only ===


def test_ml_dsa_65_local_generation_rejected() -> None:
    """Local ML-DSA-65 generation is not supported; the cloud KMS mints
    these keys. SDK rejects the request rather than shipping a stub."""
    with pytest.raises(ValueError, match="unsupported_algorithm"):
        generate_local_keypair("ml-dsa-65")  # type: ignore[arg-type]


def test_default_algorithm_is_ed25519() -> None:
    """`generate_local_keypair()` with no args defaults to ed25519."""
    pytest.importorskip("cryptography")
    kp = generate_local_keypair()
    assert kp.algorithm == "ed25519"


# === Validation ===


def test_unsupported_algorithm_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unsupported_algorithm"):
        generate_local_keypair("rsa")  # type: ignore[arg-type]


def test_local_keypair_carries_only_pem_bytes() -> None:
    """LocalKeypair is a plain dataclass; no extra state survives."""
    pytest.importorskip("cryptography")
    kp = generate_local_keypair("ed25519")
    assert isinstance(kp.private_key_pem, bytes)
    assert isinstance(kp.public_key_pem, bytes)
    assert kp.algorithm in SUPPORTED_ALGORITHMS
