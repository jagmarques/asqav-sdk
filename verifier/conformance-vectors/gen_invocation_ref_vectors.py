# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Generate the invocation_ref visibility conformance vectors.

`invocation_ref` carries a hook event's `tool_use_id` into the signed payload
so a pre-action receipt and its post-action receipt bind to ONE invocation. A
duplicate emission (two pre-action receipts, one ref) is representable and
verifies: the member exists so a reader can SEE duplicates, never to reject
them. No de-duplication, uniqueness check, or rejection is keyed on it.

  asqav-35-invocation-ref-binds-pre-post  pre + post sharing one ref; verifies
  asqav-36-invocation-ref-duplicate-emission  two pres sharing one ref; verifies

The signing key is derived from a published phrase so anyone can re-derive it;
nothing here depends on a secret the corpus does not ship.

Usage: python verifier/conformance-vectors/gen_invocation_ref_vectors.py
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
SEED_PHRASE = b"asqav conformance corpus v1 invocation-ref visibility seed"
KID = "asqav-invocation-ref-vec-key"
ISSUER = "Asqav Ltd"
_ZERO_DIGEST = hashlib.sha256(b"").hexdigest()

#: The shared invocation both vectors revolve around (fixed: minting is deterministic).
INVOCATION_REF = "toolu_vec_shared"


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


def _digest_of(context: dict) -> dict:
    """payload_digest over `context`, computed independently of the verifier."""
    encoded = _jcs(context)
    return {"hash": hashlib.sha256(encoded).hexdigest(), "size": len(encoded)}


def _payload(
    previous: str, hook_event: str, receipt_type: str, decision: str, context: dict
) -> dict:
    """An invocation payload carrying its own real context.

    action_ref is the one wire form (-09 §5.1.5): the prefixed rendering of
    payload_digest.hash, which proves the -10 §10.2 recomputation.
    """
    # Real posttool emits observation, pretool emits decision. The decision
    # member stays present on both: required, and "observation" claims nothing.
    digest = _digest_of(context)
    return {
        "type": receipt_type,
        "v": 1,
        "issued_at": "2026-09-07T12:00:00+00:00",
        "issuer_id": ISSUER,
        "agent_id": "agt_invocation_001",
        "action_ref": f"sha256:{digest['hash']}",
        "context": context,
        "payload_digest": digest,
        "policy_digest": f"sha256:{_ZERO_DIGEST}",
        "previousReceiptHash": previous,
        "decision": decision,
        "mode": "payload",
        "tool_name": "demo.action",
        "hook_event": hook_event,
        "invocation_ref": INVOCATION_REF,
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

    pre = _payload(
        "0" * 64,
        "PreToolUse",
        "protectmcp:decision",
        "allow",
        {
            "subject": "invocation-pre-post",
            "hook_event": "PreToolUse",
            "invocation_ref": INVOCATION_REF,
        },
    )
    post = _payload(
        _chain_hash(pre),
        "PostToolUse",
        "protectmcp:observation",
        "observation",
        {
            "subject": "invocation-pre-post",
            "hook_event": "PostToolUse",
            "invocation_ref": INVOCATION_REF,
        },
    )
    _write(
        "asqav-35-invocation-ref-binds-pre-post",
        {
            "predecessor.json": _sign(pre, sk),
            "receipt.json": _sign(post, sk),
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "A pre-action receipt and its post-action receipt carry the "
                    "same invocation_ref, so a reader binds them to one tool "
                    "invocation without trusting timestamps. Both receipts are "
                    "correctly signed and the chain link rederives. The shared "
                    "member is pure correlation: it changes no verdict."
                ),
            },
        },
    )

    first = _payload(
        "0" * 64,
        "PreToolUse",
        "protectmcp:decision",
        "allow",
        {"subject": "duplicate-emission", "hook_event": "PreToolUse", "emission": 1},
    )
    second = _payload(
        _chain_hash(first),
        "PreToolUse",
        "protectmcp:decision",
        "allow",
        {"subject": "duplicate-emission", "hook_event": "PreToolUse", "emission": 2},
    )
    _write(
        "asqav-36-invocation-ref-duplicate-emission",
        {
            "predecessor.json": _sign(first, sk),
            "receipt.json": _sign(second, sk),
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "Two pre-action receipts carry the same invocation_ref: the "
                    "duplicate-emission shape a harness produces when it fires "
                    "the hook twice for one invocation. Both receipts verify, "
                    "because invocation_ref exists so a reader can SEE the "
                    "duplicate, never to reject it. A verifier that refuses "
                    "this pair is wrong, not strict."
                ),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
