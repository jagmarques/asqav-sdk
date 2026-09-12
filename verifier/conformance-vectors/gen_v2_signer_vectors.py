# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Generate the v2-signer conformance vectors.

asqav-08 and asqav-09 predate the published-seed discipline (they were
committed directly, and their signing seed was never published), so they are
re-minted here under a published phrase, following the precedent
gen_oracle_vectors.py documents for exactly this situation: anyone can
re-derive the key and reproduce every signature byte.

  asqav-08-v2-signer-canary   valid v:2 receipt; signer and _asqav_tid sit
                             inside the signed body, signer surfaced
  asqav-09-v2-signer-tampered 08's payload with the signer flipped after
                             signing; the signature never matches it

asqav-09 shares asqav-08's context and digest by construction: it is 08's
payload with one member flipped, so the digest still recomputes and the
signature axis is what FAILs — the designed failure.

Usage: python verifier/conformance-vectors/gen_v2_signer_vectors.py
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

#: Nothing-up-my-sleeve seed phrase; the key is SHA-256 of these ASCII bytes.
#: The v2 key this replaces predates the published-seed discipline and its
#: seed is not recoverable; the directory entries below are the re-mint.
SEED_PHRASE = b"asqav conformance corpus v1 oracle v2 signing seed"

KID = "asqav-oracle-v2-key"
ISSUER = "Asqav Ltd"

#: The fixed policy digest the v2 demo receipts commit to.
POLICY_DIGEST = "sha256:9b71d224bd62f3785d96d46ad3ea3d73319bfbc2890caadae2dff72519673ca7"

#: The v2 canary's transaction id, carried inside the signed body.
CANARY_TID = "4d9a220e3f0b7c35c932d1e85bd4e081fb702c0a47caa046fb18b88bc9246a7f"

#: Honest signer and the flipped value the tampered twin carries.
HONEST_SIGNER = "https://api.asqav.com"
FLIPPED_SIGNER = "https://attacker.example.com"


def _jcs(obj: object) -> bytes:
    """Canonical JSON bytes, matching the oracle's asqav_jcs."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest_of(context: dict) -> dict:
    """payload_digest over `context`, computed independently of the verifier."""
    encoded = _jcs(context)
    return {"hash": hashlib.sha256(encoded).hexdigest(), "size": len(encoded)}


def _signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(SEED_PHRASE).digest())


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


def _payload(signer: str) -> dict:
    """The v2 demo payload; every member preserved from the original mint.

    context names the canary the digest commits to; action_ref is the one wire
    form (-09 §5.1.5): the prefixed rendering of payload_digest.hash.
    """
    context = {"subject": "v2-signer-canary", "signer": HONEST_SIGNER, "tid": CANARY_TID}
    digest = _digest_of(context)
    return {
        "type": "protectmcp:decision",
        "v": 2,
        "issued_at": "2026-05-04T12:00:00+00:00",
        "issuer_id": ISSUER,
        "agent_id": "agt_demo_001",
        "action_id": "act_v2_demo_0001",
        "action_ref": f"sha256:{digest['hash']}",
        "context": context,
        "payload_digest": digest,
        "policy_digest": POLICY_DIGEST,
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
        "tool_name": "demo.action",
        "mode": "hash",
        "hash": "sha256:" + "e" * 64,
        "hash_algo": "sha256",
        "org_id": "org_demo_canary",
        "server_timestamp": "2026-05-04T12:00:00+00:00",
        "signer": signer,
        "_asqav_tid": CANARY_TID,
    }


def _jwks() -> dict:
    pub = (
        _signing_key()
        .public_key()
        .public_bytes(Encoding.Raw, PublicFormat.Raw)
    )
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


def _write(name: str, files: dict[str, object]) -> None:
    out = _HERE / name
    out.mkdir(parents=True, exist_ok=True)
    for fname, obj in files.items():
        (out / fname).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {name}")


def main() -> int:
    sk = _signing_key()

    # asqav-08: the valid v:2 canary. Expected text is the vector's own,
    # verbatim: the re-mint changes no outcome and no notes sentence.
    env08 = _sign(_payload(HONEST_SIGNER), sk)
    _write(
        "asqav-08-v2-signer-canary",
        {
            "receipt.json": env08,
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "Valid v:2 receipt; signer and _asqav_tid sit inside the "
                    "signed body, Ed25519 over the canonical payload verifies, "
                    "signer is surfaced in the verdict."
                ),
            },
        },
    )

    # asqav-09: 08's payload with the signer flipped after signing. Stays
    # tampered by construction: nothing re-signs the carried payload.
    env09 = {
        "payload": _payload(FLIPPED_SIGNER),
        "signature": env08["signature"],
        "anchors": [],
    }
    _write(
        "asqav-09-v2-signer-tampered",
        {
            "receipt.json": env09,
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "unverified",
                "reason_code": "issuer_signature",
                "notes": (
                    "signer flipped after signing; because signer is inside the "
                    "signed body the Ed25519 signature over the canonical payload "
                    "no longer matches and the verdict FAILs."
                ),
                "failure_class": "invalid",
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
