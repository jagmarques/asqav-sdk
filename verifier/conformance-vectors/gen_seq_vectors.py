# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Generate the seq continuity conformance vectors.

`seq` is a server-built per-agent counter bound into the signed payload, so a
gap in the series proves receipts were withheld WITHOUT needing the withheld
receipts. That is the one thing hash linkage cannot show on its own: a chain
spanning an omission verifies perfectly (see asqav-14-omitted-action-chain),
because a link only relates the receipts that exist.

Six vectors: the four seq outcomes, plus asqav-17's two anchors-spelling twins,
which this generator owns so an action_ref-class change cannot strand them.

  asqav-17-seq-contiguous     seq N then N+1; verifies
  asqav-18-seq-gap            seq N then N+4; three receipts withheld, invalid
  asqav-19-seq-non-monotonic  a counter that goes backwards, invalid
  asqav-20-seq-absent         neither receipt carries one; still verifies, because
                              absence has to stay legal or every receipt minted
                              before the counter shipped would regress
  asqav-27-anchors-absent     asqav-17's receipt with the anchors member removed
  asqav-28-anchors-null-malformed  the same receipt with anchors present as null

The signing key is derived from a published phrase so anyone can re-derive it;
nothing here depends on a secret the corpus does not ship.

Usage: python verifier/conformance-vectors/gen_seq_vectors.py
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
SEED_PHRASE = b"asqav conformance corpus v1 seq-continuity signing seed"
KID = "asqav-seq-vec-key"
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
        "agent_id": "agt_seq_001",
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


def _pair(sk, first_extra: dict, second_extra: dict):
    """A predecessor and the successor that links to it, both properly signed.

    Both are signed over their real bytes, so a vector never depends on a broken
    signature to reach its outcome - the seq axis has to be what decides it.
    """
    first = _payload("0" * 64, **first_extra)
    second = _payload(_chain_hash(first), **second_extra)
    return _sign(first, sk), _sign(second, sk)


def main() -> int:
    sk = _signing_key()

    pred, rec = _pair(sk, {"seq": 7}, {"seq": 8})
    _write(
        "asqav-17-seq-contiguous",
        {
            "predecessor.json": pred,
            "receipt.json": rec,
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "seq 8 follows predecessor 7 with nothing missing between them. "
                    "The baseline the gap vector is measured against."
                ),
            },
        },
    )

    # 17's anchors twins: one payload build, one signature; only the envelope's
    # anchors spelling differs. Expected texts are the twins' own, verbatim.
    rec_absent = {k: v for k, v in rec.items() if k != "anchors"}
    _write(
        "asqav-27-anchors-absent",
        {
            "predecessor.json": pred,
            "receipt.json": rec_absent,
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "asqav-17's receipt with the anchors member absent entirely: "
                    "absent and an empty array are the two conformant spellings of "
                    "no anchors, so the outcome and every axis result must equal "
                    "asqav-17's. The signature covers the payload only, so removing "
                    "the envelope-level member keeps it valid."
                ),
            },
        },
    )

    rec_null = dict(rec)
    rec_null["anchors"] = None
    _write(
        "asqav-28-anchors-null-malformed",
        {
            "predecessor.json": pred,
            "receipt.json": rec_null,
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "unverified",
                "failure_class": "unverifiable",
                "reason_code": "malformed_member",
                "notes": (
                    "asqav-17's receipt with the anchors member present as a JSON "
                    "null: a third spelling of no anchors found in the wild, and a "
                    "malformed one - the structure axis FAILs, naming the member and "
                    "the shape, and the verdict reads unverified/unverifiable rather "
                    "than reading null as absent. Nothing cryptographic was "
                    "disproved, so the class is unverifiable, not invalid (same "
                    "family as duplicate_member: the receipt's own bytes are "
                    "malformed)."
                ),
            },
        },
    )

    pred, rec = _pair(sk, {"seq": 7}, {"seq": 11})
    _write(
        "asqav-18-seq-gap",
        {
            "predecessor.json": pred,
            "receipt.json": rec,
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "unverified",
                "failure_class": "invalid",
                "reason_code": "seq_gap",
                "notes": (
                    "seq jumps 7 -> 11, so three receipts were withheld. Both receipts "
                    "are correctly signed and the chain link rederives, so the hash "
                    "chain alone reports nothing wrong; the counter is what makes the "
                    "omission evidence. A proven omission is a binding failure, not an "
                    "incomplete recompute, so the class is invalid rather than "
                    "unverifiable."
                ),
            },
        },
    )

    pred, rec = _pair(sk, {"seq": 9}, {"seq": 4})
    _write(
        "asqav-19-seq-non-monotonic",
        {
            "predecessor.json": pred,
            "receipt.json": rec,
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "unverified",
                "failure_class": "invalid",
                "reason_code": "seq_not_monotonic",
                "notes": (
                    "The counter goes backwards, 9 then 4. A replayed or reordered "
                    "receipt reaches this before a gap does, and it is refused for the "
                    "same reason: the series cannot decrease under a server-built "
                    "monotonic counter."
                ),
            },
        },
    )

    pred, rec = _pair(sk, {}, {})
    _write(
        "asqav-20-seq-absent",
        {
            "predecessor.json": pred,
            "receipt.json": rec,
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "Neither receipt carries a counter, which is every receipt minted "
                    "before the member shipped. Absence MUST stay legal: the axis "
                    "passes with a note rather than skipping, because a blocking skip "
                    "would regress all of them from verified to unverified."
                ),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
