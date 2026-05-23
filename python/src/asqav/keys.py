"""Algorithm-agility helpers for offline keypair generation (Ed25519, ES256).

ML-DSA-65 is cloud-KMS only via ``Agent.create(algorithm='ml-dsa-65')``.
``cryptography`` is a deferred optional import so cloud-only callers never see
a hard dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Profile-recognised algorithm identifiers; mirrors cloud's SignatureRecord.algorithm column.
ALGORITHM_ML_DSA_65 = "ml-dsa-65"
ALGORITHM_ED25519 = "ed25519"
ALGORITHM_ES256 = "es256"

#: Algorithms `generate_local_keypair` can mint locally (ML-DSA-65 is cloud-KMS only).
SUPPORTED_ALGORITHMS: frozenset[str] = frozenset(
    {ALGORITHM_ED25519, ALGORITHM_ES256}
)

Algorithm = Literal["ed25519", "es256"]


@dataclass
class LocalKeypair:
    """A locally-generated keypair for offline / air-gapped flows.

    PEM-encoded (PKCS#8 private, SubjectPublicKeyInfo public). The SDK does
    not retain the private key after return.
    """

    algorithm: str
    private_key_pem: bytes
    public_key_pem: bytes


def _check_algorithm(algorithm: str) -> str:
    """Normalise + validate the requested algorithm string."""
    norm = algorithm.lower()
    if norm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            "unsupported_algorithm: "
            f"{algorithm!r} is not in {sorted(SUPPORTED_ALGORITHMS)}"
        )
    return norm


def generate_local_keypair(algorithm: Algorithm = "ed25519") -> LocalKeypair:
    """Generate a local Ed25519 or ES256 keypair for offline flows.

    Raises ``ValueError`` for unsupported algorithms and ``ImportError`` when
    ``cryptography`` is missing.
    """
    alg = _check_algorithm(algorithm)
    if alg == ALGORITHM_ED25519:
        return _generate_ed25519()
    return _generate_es256()


# === Per-algorithm implementations ===


def _generate_ed25519() -> LocalKeypair:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as e:
        raise ImportError(
            "ed25519 keypair generation requires the `cryptography` "
            "package. Install with `pip install cryptography`."
        ) from e

    private_key = ed25519.Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return LocalKeypair(
        algorithm=ALGORITHM_ED25519,
        private_key_pem=private_pem,
        public_key_pem=public_pem,
    )


def _generate_es256() -> LocalKeypair:
    """Generate an ECDSA P-256 keypair (alg=ES256 in JOSE)."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec
    except ImportError as e:
        raise ImportError(
            "es256 keypair generation requires the `cryptography` "
            "package. Install with `pip install cryptography`."
        ) from e

    private_key = ec.generate_private_key(ec.SECP256R1())
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return LocalKeypair(
        algorithm=ALGORITHM_ES256,
        private_key_pem=private_pem,
        public_key_pem=public_pem,
    )
