# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Generate the keyed-digest-unverifiable conformance vector.

asqav-42-keyed-digest-unverifiable: the payload-mode receipt carries BOTH
context and payload_digest, but hash_algo selects the keyed construction
(hmac-sha256), so the digest is an HMAC under holder key material no
verifier holds. Recomputing a plain SHA-256 would manufacture a mismatch;
the payload_digest axis reports SKIPPED (unverifiable), never FAIL.

The HMAC is computed with stdlib hmac over the JCS canonicalisation of
the carried context, never through the verifier: an expectation must not
come from the function under test. Minted as asqav-42 because asqav-40/41
were taken by the task ahead of this one (next free id, never reused).

Usage: python verifier/conformance-vectors/gen_keyed_digest_vector.py
Re-freeze the corpus lock afterwards: python verifier/freeze_corpus_lock.py
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

_HERE = Path(__file__).resolve().parent

#: Nothing-up-my-sleeve seed phrases; keys are SHA-256 of these ASCII bytes.
SEED_PHRASE = b"asqav conformance corpus v1 keyed-digest signing seed"
HMAC_SEED = b"asqav conformance corpus v1 keyed-digest holder-key seed"
KID = "asqav-keyed-digest-vec-key"
ISSUER = "Asqav Ltd"

CONTEXT = {"probe": "criterion-627"}


def _jcs(obj: object) -> bytes:
    """Canonical JSON bytes: sorted keys, tight separators, UTF-8."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(SEED_PHRASE).digest())


def _holder_key() -> bytes:
    """Holder key material the verifier never sees (hence unverifiable)."""
    return hashlib.sha256(HMAC_SEED).digest()


def _keyed_digest(context: dict) -> dict:
    """payload_digest as HMAC-SHA-256 over the JCS context bytes."""
    encoded = _jcs(context)
    return {
        "hash": hmac.new(_holder_key(), encoded, hashlib.sha256).hexdigest(),
        "size": len(encoded),
    }


def _payload(digest: dict) -> dict:
    """A genesis payload-mode receipt carrying context + keyed digest."""
    return {
        "type": "protectmcp:decision",
        "v": 1,
        "issued_at": "2026-09-01T12:00:00+00:00",
        "issuer_id": ISSUER,
        "agent_id": "agt_kd_001",
        "action_ref": f"sha256:{digest['hash']}",
        "mode": "payload",
        "hash_algo": "hmac-sha256",
        "context": CONTEXT,
        "payload_digest": digest,
        "policy_digest": f"sha256:{hashlib.sha256(b'').hexdigest()}",
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
        "tool_name": "demo.action",
    }


def _sign(payload: dict, sk: Ed25519PrivateKey) -> dict:
    """The three-key envelope: signature is over the canonical payload bytes."""
    return {
        "payload": payload,
        "signature": {
            "alg": "Ed25519",
            "kid": KID,
            "sig": base64.b64encode(sk.sign(_jcs(payload))).decode(),
        },
        "anchors": [],
    }


def _jwks() -> dict:
    pub = _signing_key().public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return {
        "keys": [
            {
                "kid": KID,
                "issuer_id": ISSUER,
                "alg": "Ed25519",
                "status": "active",
                "public_key": base64.b64encode(pub).decode(),
            }
        ]
    }


def main() -> int:
    sk = _signing_key()
    digest = _keyed_digest(CONTEXT)
    out = _HERE / "asqav-42-keyed-digest-unverifiable"
    out.mkdir(parents=True, exist_ok=True)
    (out / "receipt.json").write_text(
        json.dumps(_sign(_payload(digest), sk), indent=2) + "\n", encoding="utf-8"
    )
    (out / "jwks.json").write_text(json.dumps(_jwks(), indent=2) + "\n", encoding="utf-8")
    (out / "expected.json").write_text(
        json.dumps(
            {
                "format": "asqav-native",
                "outcome": "unverified",
                "failure_class": "unverifiable",
                "reason_code": "payload_digest_unchecked",
                "notes": (
                    "Payload-mode genesis carrying BOTH context and payload_digest, "
                    "but hash_algo selects the keyed construction: the digest is "
                    "an HMAC under holder key material no verifier holds, so the "
                    "payload_digest axis reports SKIPPED (unverifiable) rather "
                    "than manufacturing a mismatch FAIL. The HMAC is computed "
                    "with stdlib hmac over JCS bytes, never through the verifier."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("wrote asqav-42-keyed-digest-unverifiable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
