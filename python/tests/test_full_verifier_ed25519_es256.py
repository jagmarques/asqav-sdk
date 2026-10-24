# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""The full verifier verifies Ed25519 and ES256 signatures, not only ML-DSA-65.

The profile's mandatory-to-implement algorithm is Ed25519, and the platform's
self-hosted signer emits Ed25519 and ES256; the single-file verifier used to
SKIP every non-ML-DSA-65 algorithm, so no Ed25519 vector's signature axis ever
ran through it. The dispatch now verifies both via the optional ``cryptography``
dependency, with the oracle's key and signature forms (raw 32-byte Ed25519 key;
65-byte uncompressed P-256 point and 64-byte raw r||s signature).
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

from asqav.verifier import verify_receipt as v

_VECTORS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"


def _vector(name: str):
    d = _VECTORS / name
    return json.loads((d / "receipt.json").read_text()), json.loads((d / "jwks.json").read_text())


def _predecessor_payload(name: str):
    p = _VECTORS / name / "predecessor.json"
    if not p.exists():
        return None
    doc = json.loads(p.read_text())
    return doc["payload"] if isinstance(doc.get("payload"), dict) else doc


def _axis(result: dict, name: str) -> dict:
    return next(a for a in result["axes"] if a["name"] == name)


def _es256_material(payload: dict):
    """A P-256 key, the canonical payload signed, and the JWKS entry for it."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric.ec import (
        ECDSA,
        SECP256R1,
        generate_private_key,
    )
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    sk = generate_private_key(SECP256R1())
    point = sk.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    msg = v.canonical_json(payload)
    r, s = decode_dss_signature(sk.sign(msg, ECDSA(hashes.SHA256())))
    raw_sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    jwks_entry = {
        "kid": "k1",
        "issuer_id": payload["issuer_id"],
        "alg": "ES256",
        "status": "active",
        "public_key": base64.b64encode(point).decode(),
    }
    return point, msg, raw_sig, jwks_entry


def test_ed25519_vector_signature_passes_and_the_fold_stays_fail_closed() -> None:
    """asqav-17: the signature axis PASSes; the verdict stays unverified on the
    anchors SKIP (the anchor-less receipt cannot complete that axis offline)."""
    receipt, jwks = _vector("asqav-17-seq-contiguous")
    result = v.run_structured(
        receipt, jwks, predecessor_payload=_predecessor_payload("asqav-17-seq-contiguous")
    )
    signature = _axis(result, "signature")
    assert signature["result"] == "PASS", signature
    anchors = _axis(result, "anchors")
    assert anchors["result"] == "SKIPPED"
    assert anchors["note"] == "no anchors on this receipt"
    assert result["verdict"] == "unverified"
    assert result["failure_class"] == "unverifiable"


def test_tampered_ed25519_vector_signature_fails() -> None:
    """asqav-04 (decision flipped after signing): FAIL, and the fold says invalid."""
    receipt, jwks = _vector("asqav-04-tamper-sig")
    result = v.run_structured(receipt, jwks)
    signature = _axis(result, "signature")
    assert signature["result"] == "FAIL", signature
    assert signature["note"] == "signature mismatch"
    assert result["verdict"] == "unverified"
    assert result["failure_class"] == "invalid"


def test_es256_verifies_directly_and_through_run_structured() -> None:
    """A P-256 signature verifies; a flipped byte and a 63-byte form fail."""
    pytest.importorskip("cryptography")
    receipt, _jwks = _vector("asqav-01-genesis-permit")
    payload = receipt["payload"]
    point, msg, raw_sig, jwks_entry = _es256_material(payload)

    assert v.verify_signature(point, msg, raw_sig, "ES256") == ("PASS", "signature valid")
    bad = bytearray(raw_sig)
    bad[10] ^= 0x01
    assert v.verify_signature(point, msg, bytes(bad), "ES256")[0] == "FAIL"
    short = v.verify_signature(point, msg, raw_sig[:63], "ES256")
    assert short[0] == "FAIL" and "64-byte" in short[1] and "63" in short[1]

    envelope = {
        "payload": payload,
        "signature": {"alg": "ES256", "kid": "k1", "sig": base64.b64encode(raw_sig).decode()},
        "anchors": [],
    }
    result = v.run_structured(envelope, {"keys": [jwks_entry]})
    signature = _axis(result, "signature")
    assert signature["result"] == "PASS", signature


def test_ml_dsa_65_path_unchanged() -> None:
    """asqav-06's signature still verifies through the dilithium-py path."""
    pytest.importorskip("dilithium_py.ml_dsa")
    receipt, jwks = _vector("asqav-06-mldsa65-payload-prod")
    result = v.run_structured(receipt, jwks)
    signature = _axis(result, "signature")
    assert signature["result"] == "PASS", signature


def test_unknown_alg_still_skips_naming_it() -> None:
    result, note = v.verify_signature(b"\x00", b"\x00", b"\x00", "Ed448")
    assert result == "SKIPPED"
    assert "Ed448" in note


def test_missing_cryptography_skips_instead_of_crashing(monkeypatch) -> None:
    """No `cryptography` installed: Ed25519/ES256 report SKIPPED with the install hint."""
    # Every cached submodule must read as absent: a None parent alone lets
    # already-cached children through the from-import.
    for name in list(sys.modules):
        if name == "cryptography" or name.startswith("cryptography."):
            monkeypatch.setitem(sys.modules, name, None)
    for alg in ("Ed25519", "ES256"):
        result, note = v.verify_signature(b"\x00" * 32, b"\x00", b"\x00" * 64, alg)
        assert result == "SKIPPED", (alg, note)
        assert "pip install cryptography" in note
