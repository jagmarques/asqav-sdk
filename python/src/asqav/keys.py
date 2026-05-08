"""Algorithm-agility helpers for the IETF Compliance Receipts profile.

The Asqav cloud signs receipts server-side; client-side keypair
generation is only useful for offline scenarios (LocalQueue, air-gapped
demos, conformance vectors). This module exposes a tiny stable API so
callers can opt into Ed25519 or ES256 (in addition to the default
ML-DSA-65) without depending on `cryptography` / `pynacl` directly.

The module is import-safe even when `cryptography` is not installed:
the import is deferred to call sites so legacy users (cloud-only,
ML-DSA-65) never see a hard dependency. Callers that ask for Ed25519
or ES256 without the optional dep get a clear ImportError pointing
them at the install command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Public identifiers the Compliance Receipts profile recognises. Mirrors
# the cloud's SignatureRecord.algorithm column under §10.8 so cloud and
# SDK never disagree on what a receipt advertises.
ALGORITHM_ML_DSA_65 = "ml-dsa-65"
ALGORITHM_ED25519 = "ed25519"
ALGORITHM_ES256 = "es256"

SUPPORTED_ALGORITHMS: frozenset[str] = frozenset(
    {ALGORITHM_ML_DSA_65, ALGORITHM_ED25519, ALGORITHM_ES256}
)

Algorithm = Literal["ml-dsa-65", "ed25519", "es256"]


@dataclass
class LocalKeypair:
    """A locally-generated keypair for offline / air-gapped flows.

    `private_key_pem` is PKCS#8 (Ed25519, ES256) or a raw 32-byte secret
    for ML-DSA-65. `public_key_pem` is SubjectPublicKeyInfo (Ed25519,
    ES256) or raw 32-byte public for ML-DSA-65. The SDK intentionally
    does not retain the private key after the dataclass is returned;
    storage is the caller's problem.
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


def generate_local_keypair(algorithm: Algorithm = "ml-dsa-65") -> LocalKeypair:
    """Generate a keypair locally for the requested algorithm.

    For cloud-signed receipts the cloud generates the keypair; this
    helper is for offline scenarios (LocalQueue, conformance vector
    fixtures, air-gapped operator tooling).

    Args:
        algorithm: One of `ml-dsa-65`, `ed25519`, `es256`. Default is
            `ml-dsa-65` to match the cloud's default.

    Returns:
        A :class:`LocalKeypair` with PEM-encoded keys (or raw bytes for
        ML-DSA-65; PEM is not standardised for that algorithm yet).

    Raises:
        ValueError: If `algorithm` is not in `SUPPORTED_ALGORITHMS`.
        ImportError: If the optional `cryptography` package is needed
            but not installed.
    """
    alg = _check_algorithm(algorithm)
    if alg == ALGORITHM_ED25519:
        return _generate_ed25519()
    if alg == ALGORITHM_ES256:
        return _generate_es256()
    return _generate_ml_dsa_65()


# ---------------------------------------------------------------------------
# Per-algorithm implementations.
# ---------------------------------------------------------------------------


def _generate_ed25519() -> LocalKeypair:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as e:
        raise ImportError(
            "ed25519 keypair generation requires the `cryptography` "
            "package. Install with `pip install asqav[crypto]` or "
            "`pip install cryptography`."
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
    """Generate an ECDSA P-256 keypair (alg=ES256 in JOSE / RFC 7518)."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec
    except ImportError as e:
        raise ImportError(
            "es256 keypair generation requires the `cryptography` "
            "package. Install with `pip install asqav[crypto]` or "
            "`pip install cryptography`."
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


def _generate_ml_dsa_65() -> LocalKeypair:
    """Stub for ML-DSA-65; the cloud is authoritative for this algorithm.

    A pure-Python or Rust ML-DSA library is not yet in our optional deps.
    Calling this raises NotImplementedError with a pointer to the cloud
    `Agent.create()` flow which the SDK already supports.
    """
    raise NotImplementedError(
        "ml-dsa-65 keypair generation runs server-side; call "
        "`asqav.Agent.create(name, algorithm='ml-dsa-65')` to have the "
        "cloud generate the keypair."
    )
