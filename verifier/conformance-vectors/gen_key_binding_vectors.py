# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Generate the key_thumbprint (key binding) conformance vectors.

`key_thumbprint` is a server-built RFC 7638 JWK Thumbprint over the signing key,
bound INSIDE the signed payload. That placement is the whole point: a key swapped
under the same kid still produces a receipt whose signature verifies against the
key the directory now publishes, so the signature axis alone reports nothing
wrong. The bound digest is what stops rederiving.

Two vectors, the smallest pair that makes the axis mean something:

  asqav-21-key-thumbprint-binds   the digest rederives from the resolved key
  asqav-22-key-substituted        it does not; the receipt names a different key

The second is the attack; the first is the control that keeps it honest. Without
a passing case, an axis hard-wired to FAIL would satisfy the mismatch vector.

The digest is computed here from the published construction rather than by
calling the verifier's own helper, so a regression in that helper shows up as a
corpus failure instead of being baked into the expected value.

Both receipts are signed over their real bytes with a real ML-DSA-65 key, so the
signature axis PASSes in both and the key-binding axis is what decides the
outcome. The signer is the corpus's published ML-DSA seed; the second key exists
only to supply a digest and never signs anything.

Usage: python verifier/conformance-vectors/gen_key_binding_vectors.py
Re-freeze the corpus lock afterwards: python verifier/freeze_corpus_lock.py
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from dilithium_py.ml_dsa import ML_DSA_65

_HERE = Path(__file__).resolve().parent

#: The corpus's published ML-DSA-65 seed phrase; mirrors freeze_corpus_lock.py
MLDSA_SEED_PHRASE = b"asqav conformance corpus v1 ML-DSA-65 signing seed"

#: A second nothing-up-my-sleeve key. It never signs; it only supplies the digest
#: the substituted-key receipt names, standing in for the key that was replaced.
OTHER_SEED_PHRASE = b"asqav conformance corpus v1 ML-DSA-65 substituted key"

ALG = "ML-DSA-65"
KID = "asqav-key-binding-vec-key"
ISSUER = "Asqav Ltd"
_ZERO_DIGEST = hashlib.sha256(b"").hexdigest()

#: The one wire form (-09 §5.1.5): the prefixed rendering of payload_digest.hash.
ACTION_REF = f"sha256:{_ZERO_DIGEST}"


def _jcs(obj: object) -> bytes:
    """Canonical JSON bytes, matching the oracle's asqav_jcs."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _keypair(phrase: bytes):
    return ML_DSA_65.key_derive(hashlib.sha256(phrase).digest())


def _thumbprint(public_key: bytes) -> str:
    """RFC 7638 thumbprint over the AKP JWK, written from the published rule.

    `pub` is base64url with padding stripped. That is the one trap worth pinning:
    the directory carries the same bytes as standard base64 under `public_key`,
    and thumbprinting that alphabet yields a digest nobody else reproduces.
    """
    pub = base64.urlsafe_b64encode(public_key).decode("ascii").rstrip("=")
    jwk = {"alg": ALG, "kty": "AKP", "pub": pub}
    canonical = json.dumps(jwk, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _payload(previous: str, thumbprint: str) -> dict:
    return {
        "type": "protectmcp:decision",
        "v": 1,
        "issued_at": "2026-08-30T12:00:00+00:00",
        "issuer_id": ISSUER,
        "agent_id": "agt_keybind_001",
        "action_ref": ACTION_REF,
        "payload_digest": {"hash": _ZERO_DIGEST, "size": 0},
        "policy_digest": f"sha256:{_ZERO_DIGEST}",
        "previousReceiptHash": previous,
        "decision": "allow",
        "mode": "payload",
        "tool_name": "demo.action",
        "key_thumbprint": thumbprint,
    }


def _sign(payload: dict, sk) -> dict:
    """The three-key envelope; the signature covers the canonical payload bytes."""
    return {
        "payload": payload,
        "signature": {
            "alg": ALG,
            "kid": KID,
            "sig": base64.b64encode(
                ML_DSA_65.sign(sk, _jcs(payload), deterministic=True)
            ).decode(),
        },
        "anchors": [],
    }


def _jwks(public_key: bytes) -> dict:
    """The directory the verifier resolves the signing key from."""
    return {
        "keys": [
            {
                "kid": KID,
                "issuer_id": ISSUER,
                "alg": ALG,
                "status": "active",
                "public_key": base64.b64encode(public_key).decode(),
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
    signer_pk, signer_sk = _keypair(MLDSA_SEED_PHRASE)
    other_pk, _ = _keypair(OTHER_SEED_PHRASE)

    signer_tp = _thumbprint(signer_pk)
    other_tp = _thumbprint(other_pk)
    assert signer_tp != other_tp, "the two keys must not share a digest"

    _write(
        "asqav-21-key-thumbprint-binds",
        {
            "receipt.json": _sign(_payload("0" * 64, signer_tp), signer_sk),
            "jwks.json": _jwks(signer_pk),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "The bound key_thumbprint rederives from the resolved signing key. "
                    "The control the substitution vector is measured against: without "
                    "it an axis wired to FAIL would satisfy that vector too."
                ),
            },
        },
    )

    _write(
        "asqav-22-key-substituted",
        {
            "receipt.json": _sign(_payload("0" * 64, other_tp), signer_sk),
            "jwks.json": _jwks(signer_pk),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "unverified",
                "failure_class": "invalid",
                "reason_code": "key_substituted",
                "notes": (
                    "The receipt is correctly signed and the signature verifies against "
                    "the key the directory publishes, so the signature axis PASSes and "
                    "reports nothing wrong. The bound digest names a different key, "
                    "which is what a swap under the same kid looks like from outside. "
                    "A key that cannot be the one the issuer committed to is a proven "
                    "binding break, so the class is invalid rather than unverifiable."
                ),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
