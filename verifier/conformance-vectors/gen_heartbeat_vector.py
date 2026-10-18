# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Generate the heartbeat-shape conformance vector.

Covers the heartbeat axis the corpus otherwise lacks: a hash-mode receipt
with type protectmcp:lifecycle, decision observation, the declared interval in
metadata.heartbeat_description, no action_type member, and a consumed sequence
number. The signed policy_digest commits to the liveness_heartbeat artefact;
the artefact object itself is not a signed member.

  asqav-55-heartbeat   hash-mode heartbeat, verifies

The signing key is derived from a published phrase so anyone can re-derive it;
nothing here depends on a secret the corpus does not ship. Ed25519 keeps the
vector reproducible without a KMS or ML-DSA provider; the hash-mode shape is
what the vector pins, not the algorithm.

Usage: python verifier/conformance-vectors/gen_heartbeat_vector.py
Re-freeze the corpus lock afterwards: python verifier/freeze_corpus_lock.py
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

_HERE = Path(__file__).resolve().parent

#: Nothing-up-my-sleeve seed phrase; the key is SHA-256 of these ASCII bytes
SEED_PHRASE = b"asqav conformance corpus v1 heartbeat signing seed"
KID = "asqav-heartbeat-vec-key"
ISSUER = "Asqav Ltd"
DIR = _HERE / "asqav-55-heartbeat"

#: The fingerprinted input the hash-mode descriptor commits to, documented
#: here so a verifier author can reproduce it byte for byte
FINGERPRINT_INPUT = b"heartbeat:liveness:3600"


def _jcs(obj: object) -> bytes:
    """Canonical JSON bytes, matching the oracle's asqav_jcs."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(SEED_PHRASE).digest())


def _payload() -> dict:
    digest_input = FINGERPRINT_INPUT
    digest_hex = hashlib.sha256(digest_input).hexdigest()
    artefact = {"interval_seconds": 3600, "kind": "liveness_heartbeat"}
    return {
        "v": 1,
        "action_id": "act_hb_001",
        "agent_id": "agt_hb_001",
        "org_id": "org_hb_001",
        "policy_digest": "sha256:" + hashlib.sha256(_jcs(artefact)).hexdigest(),
        "type": "protectmcp:lifecycle",
        "issued_at": "2026-10-18T00:00:00+00:00",
        "issuer_id": ISSUER,
        "action_ref": "sha256:" + digest_hex,
        "payload_digest": {"hash": digest_hex, "size": len(digest_input)},
        "previousReceiptHash": "0" * 64,
        "decision": "observation",
        "mode": "hash",
        "hash": "sha256:" + digest_hex,
        "hash_algo": "sha256",
        "metadata": {
            "heartbeat_description": {"idle_seconds": 3599, "interval_seconds": 3600}
        },
        "server_timestamp": "2026-10-18T00:00:00Z",
        "seq": 1,
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


def main() -> None:
    sk = _signing_key()
    payload = _payload()
    receipt = {
        "payload": payload,
        "signature": {
            "alg": "Ed25519",
            "kid": KID,
            "sig": base64.b64encode(sk.sign(_jcs(payload))).decode(),
        },
        "anchors": [],
    }
    DIR.mkdir(exist_ok=True)
    (DIR / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (DIR / "jwks.json").write_text(json.dumps(_jwks(), indent=2) + "\n")
    (DIR / "expected.json").write_text(
        json.dumps(
            {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "Hash-mode heartbeat (criterion 574 C474): type "
                    "protectmcp:lifecycle, decision observation, interval in "
                    "metadata.heartbeat_description, no action_type member, "
                    "seq 1 on a genesis chain. Signed policy_digest commits "
                    "to the liveness_heartbeat artefact."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {DIR}")


if __name__ == "__main__":
    main()
