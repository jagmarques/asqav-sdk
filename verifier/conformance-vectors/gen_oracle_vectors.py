# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Generate the original oracle-family asqav-native vectors.

Nine directories predate the generator convention (they were committed
directly when the oracle landed, and their signing seeds were never
published), so the wire-version retrofit re-mints them here under published
phrases: anyone can re-derive the keys and reproduce every signature byte,
which is the discipline the other families already keep.

  asqav-01-genesis-permit         valid genesis, decision allow
  asqav-02-genesis-deny           valid genesis, decision deny
  asqav-03-chain-link             01's successor; previousReceiptHash rederives
  asqav-04-tamper-sig             01's payload with decision flipped after
                                  signing; the signature never matches it
  asqav-07-revoked-key            a valid signature from a revoked directory key
  asqav-11-dup-member-toplevel    'payload' appears twice; terminal parse
                                  failure before any hashing (criterion 419)
  asqav-13-dup-member-nested      payload_digest.hash appears twice; same class
  asqav-12-time-edge-expiry       ML-DSA-65 over the published corpus seed;
                                  extreme UTC offset, lapsed expires_at
  asqav-23-anchor-status-pending  01's receipt plus one informational anchor
                                  carrying status pending (unsigned member)

asqav-11 and asqav-13 are emitted as TEXT: json.dumps cannot write a
duplicated member, and a parse-edit-serialise round trip would destroy
exactly what they test. Both are asserted to still carry the duplicate on
their raw bytes at write time.

Usage: python verifier/conformance-vectors/gen_oracle_vectors.py
Re-freeze the corpus lock afterwards: python verifier/freeze_corpus_lock.py
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from dilithium_py.ml_dsa import ML_DSA_65

_HERE = Path(__file__).resolve().parent

#: Nothing-up-my-sleeve seed phrases; each key is SHA-256 of the ASCII bytes.
#: The keys these replace predate the published-seed discipline and their
#: seeds are not recoverable; the directory entries below are the re-mint.
SEED_PHRASE = b"asqav conformance corpus v1 oracle signing seed"
REVOKED_SEED_PHRASE = b"asqav conformance corpus v1 revoked-key signing seed"

#: The corpus's published ML-DSA-65 seed phrase; mirrors freeze_corpus_lock.py.
MLDSA_SEED_PHRASE = b"asqav conformance corpus v1 ML-DSA-65 signing seed"

KID = "asqav-oracle-vec-key"
REVOKED_KID = "asqav-revoked-vec-key"
MLDSA_KID = "asqav-corpus-mldsa-time-edge-key"
ISSUER = "Asqav Ltd"
_ZERO_DIGEST = hashlib.sha256(b"").hexdigest()

#: The one wire form (-09 §5.1.5): the prefixed rendering of payload_digest.hash.
ACTION_REF = f"sha256:{_ZERO_DIGEST}"

#: The fixed policy digest the demo receipts commit to.
POLICY_DIGEST = "sha256:9b71d224bd62f3785d96d46ad3ea3d73319bfbc2890caadae2dff72519673ca7"


def _jcs(obj: object) -> bytes:
    """Canonical JSON bytes, matching the oracle's asqav_jcs."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _ed_key(phrase: bytes) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(phrase).digest())


def _ed_pub_b64(sk: Ed25519PrivateKey) -> str:
    return base64.b64encode(
        sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode()


def _sign_ed(payload: dict, sk: Ed25519PrivateKey, kid: str = KID) -> dict:
    """The three-key envelope: the signature covers the canonical payload bytes."""
    return {
        "payload": payload,
        "signature": {
            "alg": "Ed25519",
            "kid": kid,
            "sig": base64.b64encode(sk.sign(_jcs(payload))).decode(),
        },
        "anchors": [],
    }


def _chain_hash(payload: dict) -> str:
    """The successor's previousReceiptHash: SHA-256 of the predecessor payload."""
    return hashlib.sha256(_jcs(payload)).hexdigest()


def _demo_payload(**extra) -> dict:
    """The shared demo payload shape; v: 1 is the wire version (-09 §5.1.1)."""
    payload = {
        "type": "protectmcp:decision",
        "v": 1,
        "issued_at": "2026-05-04T12:00:00+00:00",
        "issuer_id": ISSUER,
        "agent_id": "agt_demo_001",
        "action_ref": ACTION_REF,
        "payload_digest": {"hash": _ZERO_DIGEST, "size": 0},
        "policy_digest": POLICY_DIGEST,
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
        "mode": "payload",
        "tool_name": "demo.action",
    }
    payload.update(extra)
    return payload


def _jwks(kid: str, issuer_id: str, pub_b64: str, status: str = "active") -> dict:
    return {
        "keys": [
            {
                "kid": kid,
                "issuer_id": issuer_id,
                "alg": "Ed25519",
                "status": status,
                "public_key": pub_b64,
            }
        ]
    }


def _write(name: str, files: dict[str, object], indent: int = 2) -> None:
    out = _HERE / name
    out.mkdir(parents=True, exist_ok=True)
    for fname, obj in files.items():
        (out / fname).write_text(json.dumps(obj, indent=indent) + "\n", encoding="utf-8")
        print(f"wrote {name}/{fname}")


def _member_text(name: str, obj: object, level: int) -> str:
    """One pretty-printed object member at the given depth, as text."""
    pad = "  " * level
    lines = json.dumps(obj, indent=2).splitlines()
    body = lines[0] + "".join("\n" + pad + line for line in lines[1:])
    return f'{pad}"{name}": {body}'


def _write_dup_toplevel(name: str, payload_a: dict, payload_b: dict, envelope: dict) -> None:
    """asqav-11: the 'payload' member twice, which no JSON serialiser emits."""
    text = (
        "{\n"
        + ",\n".join(
            [
                _member_text("payload", payload_a, 1),
                _member_text("payload", payload_b, 1),
                _member_text("signature", envelope["signature"], 1),
                _member_text("anchors", envelope["anchors"], 1),
            ]
        )
        + "\n}\n"
    )
    assert text.count('"payload":') == 2, "the duplicate member is the vector"
    out = _HERE / name / "receipt.json"
    out.write_text(text, encoding="utf-8")
    raw = out.read_bytes()
    assert raw.count(b'"payload":') == 2, "duplicate member lost on the raw bytes"
    print(f"wrote {name}/receipt.json")


def _write_dup_nested(name: str, envelope: dict) -> None:
    """asqav-13: payload_digest.hash twice, spliced into the serialised text."""
    text = json.dumps(envelope, indent=2) + "\n"
    anchor = f'      "size": 0\n    }},'
    dup = f'      "size": 0,\n      "hash": "{'f' * 64}"\n    }},'
    assert text.count(anchor) == 1, "payload_digest block not found exactly once"
    text = text.replace(anchor, dup)
    out = _HERE / name / "receipt.json"
    out.write_text(text, encoding="utf-8")
    raw = out.read_bytes()
    assert raw.count(b'"hash":') == 2, "duplicate member lost on the raw bytes"
    print(f"wrote {name}/receipt.json")


def main() -> int:
    sk = _ed_key(SEED_PHRASE)
    revoked_sk = _ed_key(REVOKED_SEED_PHRASE)
    ed_jwks = _jwks(KID, ISSUER, _ed_pub_b64(sk))
    revoked_jwks = _jwks(REVOKED_KID, REVOKED_KID, _ed_pub_b64(revoked_sk), status="revoked")

    # asqav-01: the genesis permit every tamper/dup/anchor twin is measured against.
    p01 = _demo_payload()
    env01 = _sign_ed(p01, sk)
    _write(
        "asqav-01-genesis-permit",
        {
            "receipt.json": env01,
            "jwks.json": ed_jwks,
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "Valid genesis permit; Ed25519 signature over canonical payload "
                    "verifies; all-zero previousReceiptHash seed."
                ),
            },
        },
    )

    # asqav-02: the same genesis shape, decision deny.
    _write(
        "asqav-02-genesis-deny",
        {
            "receipt.json": _sign_ed(_demo_payload(decision="deny"), sk),
            "jwks.json": ed_jwks,
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": "Valid genesis deny decision; signature verifies.",
            },
        },
    )

    # asqav-03: 01's successor; the link is over 01's canonical payload bytes.
    p03 = _demo_payload(previousReceiptHash=_chain_hash(p01), tool_name="demo.action.2")
    _write(
        "asqav-03-chain-link",
        {
            "predecessor.json": env01,
            "receipt.json": _sign_ed(p03, sk),
            "jwks.json": ed_jwks,
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "Successor receipt; signature verifies and previousReceiptHash "
                    "rederives from the predecessor canonical payload."
                ),
            },
        },
    )

    # asqav-04: 01's signature carried beside a payload whose decision flipped
    # after signing. Signed over allow; carries deny. Stays tampered by
    # construction: nothing re-signs the carried payload.
    env04 = {
        "payload": _demo_payload(decision="deny"),
        "signature": env01["signature"],
        "anchors": [],
    }
    _write(
        "asqav-04-tamper-sig",
        {
            "receipt.json": env04,
            "jwks.json": ed_jwks,
            "expected.json": {
                "format": "asqav-native",
                "outcome": "unverified",
                "reason_code": "issuer_signature",
                "notes": (
                    "decision flipped from allow to deny after signing; signature "
                    "no longer matches the canonical payload."
                ),
                "failure_class": "invalid",
            },
        },
    )

    # asqav-07: a valid signature whose directory key is revoked.
    p07 = _demo_payload(
        issued_at="2026-06-01T12:00:00+00:00",
        issuer_id=REVOKED_KID,
        agent_id="agt_revoked_001",
    )
    _write(
        "asqav-07-revoked-key",
        {
            "receipt.json": _sign_ed(p07, revoked_sk, kid=REVOKED_KID),
            "jwks.json": revoked_jwks,
            "expected.json": {
                "format": "asqav-native",
                "outcome": "unverified",
                "reason_code": "key_revoked",
                "notes": (
                    "Valid Ed25519 signature but signing key status is 'revoked'; "
                    "key_status axis FAILs the verdict."
                ),
                "failure_class": "invalid",
            },
        },
    )

    # asqav-11: the valid genesis receipt with the top-level 'payload' member
    # twice (allow, then deny); strict ingest rejects it before any hashing.
    _write_dup_toplevel(
        "asqav-11-dup-member-toplevel", p01, _demo_payload(decision="deny"), env01
    )
    _write(
        "asqav-11-dup-member-toplevel",
        {
            "jwks.json": ed_jwks,
            "expected.json": {
                "format": "asqav-native",
                "outcome": "unverified",
                "failure_class": "unverifiable",
                "reason_code": "duplicate_member",
                "notes": (
                    "Top-level 'payload' member appears twice; strict ingest rejects "
                    "it at parse time, before any hashing or signature check, so it "
                    "never verifies."
                ),
            },
        },
    )

    # asqav-13: the valid genesis receipt with payload_digest.hash duplicated
    # two levels down; same terminal parse failure.
    _write_dup_nested("asqav-13-dup-member-nested", env01)
    _write(
        "asqav-13-dup-member-nested",
        {
            "jwks.json": ed_jwks,
            "expected.json": {
                "format": "asqav-native",
                "outcome": "unverified",
                "failure_class": "unverifiable",
                "reason_code": "duplicate_member",
                "notes": (
                    "The payload_digest.hash member is duplicated two levels down; "
                    "strict ingest rejects duplicate members at any depth at parse "
                    "time, before any hashing or signature check, so it never verifies."
                ),
            },
        },
    )

    # asqav-12: deterministic ML-DSA-65 time edge. The signature reproduces
    # byte for byte from the published corpus seed (FIPS 204 pure variant).
    mldsa_pk, mldsa_sk = ML_DSA_65.key_derive(hashlib.sha256(MLDSA_SEED_PHRASE).digest())
    p12 = {
        "type": "protectmcp:decision",
        "v": 1,
        "issued_at": "2026-05-19T00:10:00+14:00",
        "issuer_id": MLDSA_KID,
        "agent_id": "agt_time_edge_422",
        "action_ref": "sha256:" + "ab" * 32,
        "payload_digest": {"hash": "cd" * 32, "size": 256},
        "policy_digest": "sha256:" + "ef" * 32,
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
        "mode": "payload",
        "expires_at": "2026-05-20T00:10:00Z",
    }
    env12 = {
        "payload": p12,
        "signature": {
            "alg": "ML-DSA-65",
            "kid": MLDSA_KID,
            "sig": base64.b64encode(
                ML_DSA_65.sign(mldsa_sk, _jcs(p12), deterministic=True)
            ).decode(),
        },
        "anchors": [{"type": "rfc3161", "value": "dGVzdC1hbmNob3I="}],
    }
    _write(
        "asqav-12-time-edge-expiry",
        {
            "receipt.json": env12,
            "jwks.json": {
                "keys": [
                    {
                        "kid": MLDSA_KID,
                        "issuer_id": MLDSA_KID,
                        "alg": "ML-DSA-65",
                        "status": "active",
                        "public_key": base64.b64encode(mldsa_pk).decode(),
                    }
                ]
            },
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "Deterministic ML-DSA-65 vector minted from the published corpus "
                    "signing seed (criterion 420; dilithium-py deterministic=True, "
                    "FIPS 204 pure variant), so the signature bytes reproduce exactly. "
                    "Time-edge conformance (criterion 422): issued_at "
                    "2026-05-19T00:10:00+14:00 is an extreme positive UTC offset around "
                    "midnight (UTC 2026-05-18T10:10:00, a past instant the skew axis "
                    "accepts); the signed expires_at 2026-05-20T00:10:00Z has lapsed, "
                    "so the expiry axis FAILs alone while the verdict stays verified "
                    "(criterion 426)."
                ),
            },
        },
    )

    # asqav-23: 01's receipt bytes plus one informational anchor entry whose
    # optional status member reads pending. Anchors sit outside the signed
    # bytes, so the signature is 01's own. This directory keeps its one-space
    # indent from the anchor-entry minting.
    env23 = {
        "payload": p01,
        "signature": env01["signature"],
        "anchors": [
            {"type": "opentimestamps", "value": "dGVzdC1hbmNob3I=", "status": "pending"}
        ],
    }
    _write(
        "asqav-23-anchor-status-pending",
        {
            "receipt.json": env23,
            "jwks.json": ed_jwks,
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "asqav-01's valid genesis receipt with one anchor entry carrying "
                    "the optional informational `status` member (draft anchors[] "
                    "schema): status pending declares a not-yet-anchored proof, so the "
                    "verifier must not count it as a trusted anchor. The signature is "
                    "untouched (anchors sit outside the signed bytes), so the oracle "
                    "verdict stays verified; under the full verifier the anchors axis "
                    "reports SKIPPED (unverifiable), never a PASS on presence."
                ),
            },
        },
        indent=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
