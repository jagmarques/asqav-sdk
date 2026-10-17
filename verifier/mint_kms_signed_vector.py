"""Mint the asqav-54 KMS-signed conformance vector (criterion 569 C9).

Test-KMS mint, deterministic: dilithium-py key_derive over a published seed
phrase (NEVER production), signed through the kms.sign(ref, message) call
shape. Re-running byte-reproduces conformance-vectors/asqav-54-kms-signed/.
Ran with dilithium-py==1.5.1.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python" / "src"))

from dilithium_py.ml_dsa import ML_DSA_65

from asqav.verifier.oracle.canonical import asqav_jcs
from asqav.verifier.verify_receipt import envelope_minus_anchors_jcs

OUT = HERE / "conformance-vectors" / "asqav-54-kms-signed"
SEED = b"asqav test-KMS vector asqav-54 (never production)"
ISSUED = "2026-10-17T18:00:00Z"
EXPIRES = "2027-10-17T18:00:00Z"
OID_MLDSA65 = bytes.fromhex("0609608648016503040312")


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b32ish(digest: bytes, n: int) -> str:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    num, out = int.from_bytes(digest, "big"), ""
    while len(out) < n:
        out += alphabet[num % 62]
        num //= 62
    return out[:n]


def _der_len(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(b)]) + b


def _tlv(tag: int, content: bytes) -> bytes:
    return bytes([tag]) + _der_len(len(content)) + content


def _read_len(der: bytes, pos: int) -> tuple[int, int]:
    first = der[pos]
    pos += 1
    if first < 0x80:
        return first, pos
    n = first & 0x7F
    return int.from_bytes(der[pos : pos + n], "big"), pos + n


def _unwrap_spki(der: bytes) -> bytes:
    # Same walk as asqav_cloud.core.kms._unwrap_spki_to_raw_key.
    assert der[0] == 0x30
    _, pos = _read_len(der, 1)
    assert der[pos] == 0x30
    alen, apos = _read_len(der, pos + 1)
    pos = apos + alen
    assert der[pos] == 0x03
    blen, bpos = _read_len(der, pos + 1)
    assert der[bpos] == 0
    return der[bpos + 1 : bpos + blen]


def main() -> None:
    cal = json.loads(
        (HERE / "conformance-vectors" / "asqav-06-mldsa65-payload-prod"
         / "receipt.json").read_text()
    )
    ctx_jcs = asqav_jcs(cal["payload"]["context"])
    assert hashlib.sha256(ctx_jcs).hexdigest() == (
        "b0e3686703eb8c50ee5575ddacfe2277169c24bec8a59d118dca612b08cd061d"
    )
    assert len(ctx_jcs) == 63

    pk, sk = ML_DSA_65.key_derive(hashlib.sha256(SEED).digest())
    assert len(pk) == 1952
    issuer = str(uuid.uuid5(uuid.NAMESPACE_URL, "asqav-54-kms-test-issuer"))
    agent = "agt_" + _b32ish(hashlib.sha256(b"asqav-54-kms-test-agent").digest(), 22)
    action = "act_" + _b32ish(hashlib.sha256(b"asqav-54-kms-test-action").digest(), 22)
    nonce = hashlib.sha256(b"asqav-54-kms-test-nonce").hexdigest()[:24]
    context = {"purpose": "kms-signed-test-vector", "tag": "mldsa65-kms54"}
    ctx_raw = asqav_jcs(context)
    # Generator-owned vectors bind action_ref to the payload digest by
    # construction (test_action_ref_form.py keeps that invariant).
    pd_hash = hashlib.sha256(ctx_raw).hexdigest()
    payload = {
        "v": 1,
        "action_id": action,
        "agent_id": agent,
        "org_id": issuer,
        "policy_digest": "sha256:" + hashlib.sha256(b"asqav-54-kms-policy").hexdigest(),
        "type": "protectmcp:decision",
        "issued_at": ISSUED,
        "issuer_id": issuer,
        "action_ref": "sha256:" + pd_hash,
        "payload_digest": {
            "hash": pd_hash,
            "size": len(ctx_raw),
        },
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
        "mode": "payload",
        "action_type": "test:conformance",
        "context": context,
        "timestamp": ISSUED,
        "tool_name": "test:conformance",
        "expires_at": EXPIRES,
        "nonce": nonce,
        "controls_evaluated": {
            "emergency_halt": {"checked": True, "halted": False},
            "result": "allowed",
            "policy": {"evaluated": True, "matched_count": 1},
            "content_scan": {"ran": True, "blocking": False},
        },
    }
    message = asqav_jcs(payload)
    sig = ML_DSA_65.sign(sk, message, deterministic=True)
    assert ML_DSA_65.verify(pk, message, sig)
    sig_b64u = _b64u(sig)
    assert len(sig_b64u) == 4412 and "+" not in sig_b64u
    assert "/" not in sig_b64u and "=" not in sig_b64u
    assert len(base64.urlsafe_b64decode(sig_b64u + "==")) == 3309

    spki = _tlv(0x30, _tlv(0x30, OID_MLDSA65) + _tlv(0x03, b"\x00" + pk))
    assert _unwrap_spki(spki) == pk
    kid = _b64u(hashlib.sha256(spki).digest())
    jwk = {
        "kid": kid,
        "agent_id": agent,
        "issuer_id": issuer,
        "alg": "ML-DSA-65",
        "kty": "AKP",
        "pub": _b64u(pk),
        "public_key": base64.b64encode(pk).decode("ascii"),
        "status": "active",
    }
    receipt = {
        "payload": payload,
        "signature": {"alg": "ML-DSA-65", "kid": issuer, "sig": sig_b64u},
        "anchors": [],
    }
    anchor_hex = hashlib.sha256(envelope_minus_anchors_jcs(receipt)).hexdigest()
    expected = {
        "format": "asqav-native",
        "outcome": "verified",
        "reason_code": "",
        "notes": (
            "Payload-mode receipt signed by a test KMS (criterion 569 C9): "
            "key derived from the published seed in mint_kms_signed_vector.py "
            "(deterministic dilithium-py key_derive + sign, NEVER production), "
            "held by a BaseKMSClient-conformant double. kms_spki_der_b64.txt "
            "is the KMS-issued SPKI the AKP JWK pub unwraps from; jwks kid is "
            "base64url(sha256(SPKI)); signature.kid names the issuer. Signed "
            "2026-10-17, expires 2027-10-17 so expiry stays PASS."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (OUT / "jwks.json").write_text(json.dumps({"keys": [jwk]}, indent=2) + "\n")
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    (OUT / "kms_spki_der_b64.txt").write_text(
        base64.b64encode(spki).decode("ascii") + "\n"
    )
    print("anchor sha256:", anchor_hex)


if __name__ == "__main__":
    main()
