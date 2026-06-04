"""Signature-verify dispatch shared across format adapters.

One entry point, ``verify_signature(alg, pk, msg, sig)``, returns a 3-state
``(result, note)`` where result is ``PASS`` / ``FAIL`` / ``SKIPPED`` - the same
contract the standalone ``verify_receipt`` uses, so the oracle never prints a
PASS for a signature it could not actually check.

Algorithm wiring:
  - ``ML-DSA-65`` reuses ``verify_receipt.verify_signature`` (the dilithium-py
    path); absent the optional dep it SKIPs, never FAILs.
  - ``Ed25519`` uses the ``cryptography`` library when present; absent it SKIPs
    with a clear install hint, mirroring the dilithium-py treatment.

No crypto is hand-rolled here; every algorithm delegates to a vetted library.
"""
from __future__ import annotations

import os
import sys

# The standalone verifier lives one directory up and is import-safe (stdlib +
# argparse only at import time); reuse its ML-DSA path instead of duplicating it.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import verify_receipt as _vr  # noqa: E402

#: Verdict strings shared with verify_receipt so the two surfaces report alike.
PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"


def verify_ml_dsa_65(pk: bytes, msg: bytes, sig: bytes) -> tuple[str, str]:
    """ML-DSA-65 verify via the standalone tool's dilithium-py path."""
    return _vr.verify_signature(pk, msg, sig, "ML-DSA-65")


def verify_ed25519(pk: bytes, msg: bytes, sig: bytes) -> tuple[str, str]:
    """Ed25519 verify via the ``cryptography`` library; SKIP when it is absent.

    The public key is the raw 32-byte form (RFC 8032); callers that hold an SPKI
    or JWK pass the decoded raw bytes.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return SKIPPED, "run 'pip install cryptography' for the Ed25519 check"
    try:
        key = Ed25519PublicKey.from_public_bytes(pk)
    except Exception as exc:  # malformed key bytes
        return FAIL, f"bad Ed25519 public key: {exc}"
    try:
        key.verify(sig, msg)
        return PASS, "signature valid"
    except InvalidSignature:
        return FAIL, "signature mismatch"
    except Exception as exc:  # malformed signature bytes
        return FAIL, f"verify error: {exc}"


def verify_es256(pk: bytes, msg: bytes, sig: bytes) -> tuple[str, str]:
    """ES256 (ECDSA P-256 over SHA-256) verify via ``cryptography``; SKIP when absent.

    The public key is the 65-byte uncompressed point (0x04 || X || Y). The signature
    is the 64-byte raw ``r || s`` form that WebCrypto and JOSE emit; it is converted
    to DER for the library, which does not accept the raw form directly.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric.ec import (
            ECDSA,
            SECP256R1,
            EllipticCurvePublicKey,
        )
        from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
    except ImportError:
        return SKIPPED, "run 'pip install cryptography' for the ES256 check"
    try:
        key = EllipticCurvePublicKey.from_encoded_point(SECP256R1(), pk)
    except Exception as exc:  # malformed point bytes
        return FAIL, f"bad P-256 public key: {exc}"
    if len(sig) != 64:
        return FAIL, f"ES256 signature must be 64-byte raw r||s, got {len(sig)}"
    der = encode_dss_signature(int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big"))
    try:
        key.verify(der, msg, ECDSA(hashes.SHA256()))
        return PASS, "signature valid"
    except InvalidSignature:
        return FAIL, "signature mismatch"
    except Exception as exc:  # malformed signature bytes
        return FAIL, f"verify error: {exc}"


#: alg token (upper-cased) -> verify callable; the dispatch table the oracle walks.
_DISPATCH = {
    "ML-DSA-65": verify_ml_dsa_65,
    "ED25519": verify_ed25519,
    "EDDSA": verify_ed25519,
    "ES256": verify_es256,
}


def verify_signature(alg: str, pk: bytes, msg: bytes, sig: bytes) -> tuple[str, str]:
    """Dispatch to the algorithm's verifier; SKIP an unsupported alg."""
    fn = _DISPATCH.get((alg or "").upper())
    if fn is None:
        return SKIPPED, f"unsupported alg {alg!r} (oracle checks {sorted(_DISPATCH)})"
    return fn(pk, msg, sig)
