# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""RFC 3161 tokens signed under the bare rsaEncryption OID verify with pinned material.

RFC 5652 permits 1.2.840.113549.1.1.1 as the SignerInfo signatureAlgorithm, the
hash taken from the SignerInfo digestAlgorithm, and real TSAs emit exactly that
shape: the corpus's production token (asqav-24-anchor-block-hash-prod) is signed
by tsa.izenpe.com under rsaEncryption with sha256. The dispatch used to know
only the shaNNNWithRSAEncryption identifiers, so no pinned material could ever
verify this token.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from asqav.verifier import verify_receipt as v
from asqav.verifier.verify_receipt import envelope_minus_anchors_jcs

_CORPUS = (
    Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"
)
_RSA_ENCRYPTION_OID = "1.2.840.113549.1.1.1"


def _receipt() -> dict:
    return json.loads((_CORPUS / "asqav-24-anchor-block-hash-prod" / "receipt.json").read_text())


def _token() -> bytes:
    return base64.b64decode(_receipt()["anchors"][0]["value"])


def _embedded_cert_ders() -> list[bytes]:
    """DER of each certificate embedded in the token (the TSA's own chain)."""
    return v._parse_time_stamp_resp(_token())["certs"]


def _envelope() -> dict:
    r = _receipt()
    return {"payload": r["payload"], "signature": r["signature"], "anchors": r["anchors"]}


def _bound_and_env_jcs() -> tuple[bytes, bytes]:
    env_jcs = envelope_minus_anchors_jcs(_envelope())
    return hashlib.sha256(env_jcs).digest(), env_jcs


def test_token_shape_is_the_bare_rsaencryption_oid() -> None:
    """The fixture really is the shape under test, or the tests below pass vacuously."""
    info = v._parse_time_stamp_resp(_token())
    assert info["sig_alg"] == _RSA_ENCRYPTION_OID
    assert info["digest_alg"] == "2.16.840.1.101.3.4.2.1"  # sha256
    assert info["signed_attrs"] is not None
    assert _embedded_cert_ders(), "the token embeds no signer certificates"


def test_real_token_verifies_with_its_embedded_signer_certs() -> None:
    """The production token verifies once its own signer certs are pinned."""
    bound, env_jcs = _bound_and_env_jcs()
    outcome, detail, when = v._check_rfc3161_anchor(
        _token(), bound, _embedded_cert_ders(), env_jcs
    )
    assert outcome == "verified", detail
    assert when is not None

    ev = v.evaluate_anchors(_envelope(), trusted_tsa_keys=_embedded_cert_ders())
    assert "- rfc3161: verified" in ev.note
    # The OTS entry still cannot complete offline, so the axis as a whole SKIPs.
    assert ev.result == "SKIPPED"


def test_wrong_rsa_key_is_invalid_not_unverifiable() -> None:
    """A usable key whose signature does not verify is a proven failure."""
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
    )

    other = rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()
    other_der = other.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
    bound, env_jcs = _bound_and_env_jcs()
    outcome, detail, _ = v._check_rfc3161_anchor(
        _token(), bound, [other_der], env_jcs
    )
    assert outcome == "invalid"
    assert "does not verify" in detail


def test_flipped_cms_signature_byte_is_invalid() -> None:
    """One flipped byte in the CMS signature breaks it, correct key or not."""
    token = _token()
    sig = v._parse_time_stamp_resp(token)["signature"]
    off = token.find(sig)
    assert off != -1 and token.count(sig) == 1
    tampered = token[:off] + bytes([token[off] ^ 0x01]) + token[off + 1 :]

    bound, env_jcs = _bound_and_env_jcs()
    outcome, detail, _ = v._check_rfc3161_anchor(
        tampered, bound, _embedded_cert_ders(), env_jcs
    )
    assert outcome == "invalid"
    assert "does not verify" in detail


def test_unknown_digest_oid_with_rsaencryption_is_unverifiable() -> None:
    """A digest OID the table does not know: unverifiable, never a crash or a pass."""
    state = v._verify_tsa_signature(
        _RSA_ENCRYPTION_OID,
        b"\x00",
        b"\x00",
        _embedded_cert_ders(),
        digest_alg="1.2.840.113549.99.99.99",
    )
    assert state == "unverifiable"
