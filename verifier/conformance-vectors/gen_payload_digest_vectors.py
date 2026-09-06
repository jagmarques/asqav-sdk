# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Generate the payload_digest-over-context conformance vectors.

-09 §10.2: a receipt carrying BOTH `context` and `payload_digest` is checkable
from its own bytes - `hash` is SHA-256 over the JCS canonicalisation of the
carried context, `size` that canonical form's byte length. Every other
digest-carrying vector omits the context, so the recompute path had no fixture.

Two vectors, minted as asqav-25/26 because asqav-23/24 were minted by the
anchor-entry task ahead of this one (next free id at mint time, never reused):

  asqav-25-payload-digest-rederives  honest digest beside its context; verifies
  asqav-26-payload-digest-mismatch   the same receipt signed over a digest of a
                                     DIFFERENT context; the signature PASSes and
                                     the payload_digest axis is what FAILs

The digest is computed with stdlib json.dumps in JCS shape, never by importing
the verifier's canonical_json: an expectation must not come from the function
under test.

Usage: python verifier/conformance-vectors/gen_payload_digest_vectors.py
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
SEED_PHRASE = b"asqav conformance corpus v1 payload-digest signing seed"
KID = "asqav-payload-digest-vec-key"
ISSUER = "Asqav Ltd"
_ZERO_DIGEST = hashlib.sha256(b"").hexdigest()

#: The two production context shapes measured 2026-09-03 (task T-005): ASCII,
#: no raw-content keys. The honest vector carries the first; the lying digest
#: commits to the second.
HONEST_CONTEXT = {"probe": "criterion-462"}
OTHER_CONTEXT = {"n": 3}


def _jcs(obj: object) -> bytes:
    """Canonical JSON bytes: sorted keys, tight separators, UTF-8."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


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


def _digest_of(context: dict) -> dict:
    """payload_digest over `context`, computed independently of the verifier."""
    encoded = _jcs(context)
    return {"hash": hashlib.sha256(encoded).hexdigest(), "size": len(encoded)}


def _payload(digest: dict) -> dict:
    """A genesis payload-mode receipt carrying BOTH context and payload_digest."""
    return {
        "type": "protectmcp:decision",
        "v": 1,
        "issued_at": "2026-09-01T12:00:00+00:00",
        "issuer_id": ISSUER,
        "agent_id": "agt_pd_001",
        "action_ref": f"sha256:{digest['hash']}",
        "mode": "payload",
        "context": HONEST_CONTEXT,
        "payload_digest": digest,
        "policy_digest": f"sha256:{_ZERO_DIGEST}",
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
        "tool_name": "demo.action",
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


def _write(name: str, files: dict[str, object]) -> None:
    out = _HERE / name
    out.mkdir(parents=True, exist_ok=True)
    for fname, obj in files.items():
        (out / fname).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {name}")


def main() -> int:
    sk = _signing_key()
    honest = _digest_of(HONEST_CONTEXT)
    assert honest["size"] == 25, honest  # the measured production shape

    _write(
        "asqav-25-payload-digest-rederives",
        {
            "receipt.json": _sign(_payload(honest), sk),
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "Payload-mode genesis carrying BOTH context and payload_digest, the "
                    "one pair no other vector has: hash is SHA-256 over the JCS of the "
                    "carried context and size its byte length (25), so the payload_digest "
                    "axis recomputes and PASSes rather than passing on absence. The digest "
                    "is computed with stdlib json.dumps in JCS shape, never through the "
                    "verifier's canonical_json."
                ),
            },
        },
    )

    lying = dict(honest)
    lying["hash"] = _digest_of(OTHER_CONTEXT)["hash"]
    _write(
        "asqav-26-payload-digest-mismatch",
        {
            "receipt.json": _sign(_payload(lying), sk),
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "unverified",
                "failure_class": "invalid",
                "reason_code": "payload_digest_mismatch",
                "notes": (
                    "The asqav-25 receipt with only payload_digest.hash changed: it "
                    "commits to a DIFFERENT context ({'n': 3}) than the one carried, and "
                    "the receipt is re-signed over the lying payload, so the signature "
                    "axis PASSes and the digest is the only lie. The payload_digest axis "
                    "recomputes from the carried context and FAILs - a proven internal "
                    "inconsistency, so the class is invalid, not unverifiable."
                ),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
