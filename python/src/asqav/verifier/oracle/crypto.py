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

# The standalone verifier is a package sibling and is import-safe (stdlib +
# argparse only at import time); reuse its ML-DSA path instead of duplicating it.
from asqav.verifier import verify_receipt as _vr

#: Axis-result tokens shared with verify_receipt so both surfaces report alike.
#: Criterion 418: FAIL is gone - every failure names its class, INVALID or
#: UNVERIFIABLE, and SKIPPED survives only for axes that do not apply.
PASS = "PASS"
INVALID = "INVALID"
UNVERIFIABLE = "UNVERIFIABLE"
SKIPPED = "SKIPPED"

#: Raw Ed25519 wire lengths (RFC 8032); a short input is malformed, never checked.
ED25519_PK_LEN = 32
ED25519_SIG_LEN = 64

#: Raw ES256 wire lengths: the 65-byte uncompressed point and 64-byte r||s.
ES256_PK_LEN = 65
ES256_SIG_LEN = 64


    # ML-DSA-65 verify via the standalone tool's dilithium-py path.
def verify_ml_dsa_65(pk: bytes, msg: bytes, sig: bytes) -> tuple[str, str, str]:
    return _vr.verify_signature(pk, msg, sig, "ML-DSA-65")


def verify_ed25519(pk: bytes, msg: bytes, sig: bytes) -> tuple[str, str, str]:
    """Ed25519 verify via the ``cryptography`` library.

    Returns ``(result, note, reason_code)``: a mismatch is INVALID, an input the
    check cannot run on (missing library, malformed key or signature bytes) is
    UNVERIFIABLE, never a collapsed generic failure (criterion 418). The public
    key is the raw 32-byte form (RFC 8032); callers holding an SPKI or JWK pass
    the decoded raw bytes.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return (
            UNVERIFIABLE,
            "run 'pip install cryptography' for the Ed25519 check",
            "crypto_dependency_missing",
        )
    if len(pk) != ED25519_PK_LEN:
        return (
            UNVERIFIABLE,
            f"bad Ed25519 public key: expected {ED25519_PK_LEN} bytes, got {len(pk)}",
            "key_malformed",
        )
    if len(sig) != ED25519_SIG_LEN:
        return (
            UNVERIFIABLE,
            f"bad Ed25519 signature: expected {ED25519_SIG_LEN} bytes, got {len(sig)}",
            "signature_malformed",
        )
    try:
        key = Ed25519PublicKey.from_public_bytes(pk)
    except Exception as exc:  # malformed key bytes
        return UNVERIFIABLE, f"bad Ed25519 public key: {exc}", "key_malformed"
    try:
        key.verify(sig, msg)
        return PASS, "signature valid", "none"
    except InvalidSignature:
        return INVALID, "signature mismatch", "signature_mismatch"
    except Exception as exc:  # malformed signature bytes
        return UNVERIFIABLE, f"verify error: {exc}", "signature_malformed"


def verify_es256(pk: bytes, msg: bytes, sig: bytes) -> tuple[str, str, str]:
    """ES256 (ECDSA P-256 over SHA-256) verify via ``cryptography``.

    Returns ``(result, note, reason_code)`` under the criterion 418 classes.
    The public key is the 65-byte uncompressed point (0x04 || X || Y). The
    signature is the 64-byte raw ``r || s`` form that WebCrypto and JOSE emit;
    it is converted to DER for the library, which does not accept the raw form.
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
        return (
            UNVERIFIABLE,
            "run 'pip install cryptography' for the ES256 check",
            "crypto_dependency_missing",
        )
    if len(pk) != ES256_PK_LEN or (pk and pk[0] != 0x04):
        return (
            UNVERIFIABLE,
            f"bad P-256 public key: expected {ES256_PK_LEN}-byte uncompressed point, "
            f"got {len(pk)}",
            "key_malformed",
        )
    try:
        key = EllipticCurvePublicKey.from_encoded_point(SECP256R1(), pk)
    except Exception as exc:  # malformed point bytes
        return UNVERIFIABLE, f"bad P-256 public key: {exc}", "key_malformed"
    if len(sig) != ES256_SIG_LEN:
        return (
            UNVERIFIABLE,
            f"ES256 signature must be {ES256_SIG_LEN}-byte raw r||s, got {len(sig)}",
            "signature_malformed",
        )
    der = encode_dss_signature(int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big"))
    try:
        key.verify(der, msg, ECDSA(hashes.SHA256()))
        return PASS, "signature valid", "none"
    except InvalidSignature:
        return INVALID, "signature mismatch", "signature_mismatch"
    except Exception as exc:  # malformed signature bytes
        return UNVERIFIABLE, f"verify error: {exc}", "signature_malformed"


#: alg token (upper-cased) -> verify callable; the dispatch table the oracle walks.
_DISPATCH = {
    "ML-DSA-65": verify_ml_dsa_65,
    "ED25519": verify_ed25519,
    "EDDSA": verify_ed25519,
    "ES256": verify_es256,
}


    # Dispatch to the algorithm's verifier; an unsupported alg cannot be recomputed.
def verify_signature(alg: object, pk: bytes, msg: bytes, sig: bytes) -> tuple[str, str, str]:
    fn = _DISPATCH.get((alg if isinstance(alg, str) else "").upper())
    if fn is None:
        return (
            UNVERIFIABLE,
            f"unsupported alg {alg!r} (oracle checks {sorted(_DISPATCH)})",
            "algorithm_unsupported",
        )
    return fn(pk, msg, sig)
