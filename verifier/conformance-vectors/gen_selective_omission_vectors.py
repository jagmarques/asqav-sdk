# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Generate the selective-omission conformance vectors.

Three vectors covering what chain integrity does and does not show:

  asqav-14-omitted-action-chain   an Action produced NO receipt; the successor
                                  still verifies, because the chain links only
                                  the receipts that exist
  asqav-15-unsigned-gap           a signer outage carried as evidence in the
                                  next receipt the issuer managed to sign
  asqav-16-chain-emission-blocked a predecessor-lookup timeout carried as a
                                  lifecycle receipt that links into the chain

The signing key is derived from a published phrase so anyone can re-derive it;
nothing here depends on a secret the corpus does not ship.

Usage: python verifier/conformance-vectors/gen_selective_omission_vectors.py
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
SEED_PHRASE = b"asqav conformance corpus v1 selective-omission signing seed"
KID = "asqav-omission-vec-key"
ISSUER = "Asqav Ltd"
_ZERO_DIGEST = hashlib.sha256(b"").hexdigest()

#: The one wire form (-09 §5.1.5): the prefixed rendering of payload_digest.hash.
ACTION_REF = f"sha256:{_ZERO_DIGEST}"


def _jcs(obj: object) -> bytes:
    """Canonical JSON bytes, matching the oracle's asqav_jcs."""
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


def _chain_hash(payload: dict) -> str:
    """The successor's previousReceiptHash: SHA-256 of the predecessor payload."""
    return hashlib.sha256(_jcs(payload)).hexdigest()


def _payload(previous: str, **extra) -> dict:
    payload = {
        "type": "protectmcp:decision",
        "v": 1,
        "issued_at": "2026-08-30T12:00:00+00:00",
        "issuer_id": ISSUER,
        "agent_id": "agt_omission_001",
        "action_ref": ACTION_REF,
        "payload_digest": {"hash": _ZERO_DIGEST, "size": 0},
        "policy_digest": f"sha256:{_ZERO_DIGEST}",
        "previousReceiptHash": previous,
        "decision": "allow",
        "tool_name": "demo.action",
    }
    payload.update(extra)
    return payload


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
    genesis_prev = "0" * 64

    # Action 1 is receipted, Action 2 happens with the signer never reached, and
    # Action 3 is receipted linking straight back to Action 1's receipt
    first = _payload(genesis_prev)
    third = _payload(_chain_hash(first))
    _write(
        "asqav-14-omitted-action-chain",
        {
            "predecessor.json": _sign(first, sk),
            "receipt.json": _sign(third, sk),
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "An Action between the two receipts produced no receipt at all. "
                    "The chain still verifies: links cover the receipts that exist "
                    "and are silent about an Action that never reached the signer. "
                    "Completeness is a deployment property, not a chain property."
                ),
            },
        },
    )

    # The signer was unavailable for two Actions; the next receipt that signs
    # carries the tally so the gap is evidenced rather than silent
    gap_prev = _payload(genesis_prev)
    gap = _payload(
        _chain_hash(gap_prev),
        unsigned_gap={
            "count": 2,
            "from": "2026-08-30T11:58:00+00:00",
            "to": "2026-08-30T11:59:30+00:00",
        },
    )
    _write(
        "asqav-15-unsigned-gap",
        {
            "predecessor.json": _sign(gap_prev, sk),
            "receipt.json": _sign(gap, sk),
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "Server-built unsigned_gap evidences a signer outage that "
                    "preceded this receipt. A verifier MUST NOT read it as an "
                    "assertion that the unsigned Actions were policy-evaluated."
                ),
            },
        },
    )

    # A predecessor-lookup timeout blocked emission; the lifecycle receipt
    # naming it is itself a Compliance Receipt and links into the chain
    blocked_prev = _payload(genesis_prev)
    blocked = _payload(
        _chain_hash(blocked_prev),
        type="protectmcp:lifecycle",
        decision="deny",
        reason="chain_emission_blocked",
    )
    _write(
        "asqav-16-chain-emission-blocked",
        {
            "predecessor.json": _sign(blocked_prev, sk),
            "receipt.json": _sign(blocked, sk),
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "A bounded predecessor-lookup timeout blocked emission. The "
                    "lifecycle receipt naming it links into the chain once emission "
                    "resumes, so the blocked interval is detectable rather than silent."
                ),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
