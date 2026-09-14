# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Generate the Article-50 lifecycle-disclosure conformance vector.

-09's Article 50 binding names a `protectmcp:lifecycle` disclosure receipt as
one of the two declaration shapes. The corpus carried no `eu_ai_act_articles`
member in any vector (measured 2026-09-14 against the pinned base: zero files
under verifier/conformance-vectors/, with `protectmcp` matching 70 as the
instrument control), so nothing showed what an Article 50 receipt looks like.

One vector, minted as asqav-29 because the id holes at mint time are exactly
{29, 30} and asqav-30 is reserved by T-040 (ids are never reused):

  asqav-29-article-50-lifecycle-disclosure
      a lifecycle receipt whose signed payload declares
      eu_ai_act_articles ["Article-50"] with
      framework_mappings_self_declared true; verifies

The digest is computed with stdlib json.dumps in JCS shape, never by importing
the verifier's canonical_json: an expectation must not come from the function
under test.

Usage: python verifier/conformance-vectors/gen_article_50_vectors.py
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
SEED_PHRASE = b"asqav conformance corpus v1 article-50 signing seed"
KID = "asqav-article-50-vec-key"
ISSUER = "Asqav Ltd"
_ZERO_DIGEST = hashlib.sha256(b"").hexdigest()

#: The disclosure context the action_ref commits to, matching the T-018
#: derivation gen_payload_digest_vectors.py uses: sha256 of the canonical context.
CONTEXT = {"subject": "article-50-disclosure"}


def _jcs(obj: object) -> bytes:
    """Canonical JSON bytes: sorted keys, tight separators, UTF-8."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(SEED_PHRASE).digest())


def _digest_of(context: dict) -> dict:
    """payload_digest over `context`, computed independently of the verifier."""
    encoded = _jcs(context)
    return {"hash": hashlib.sha256(encoded).hexdigest(), "size": len(encoded)}


def _payload(digest: dict) -> dict:
    """A genesis lifecycle receipt declaring Article-50 on its signed payload."""
    return {
        "type": "protectmcp:lifecycle",
        "v": 1,
        "issued_at": "2026-09-05T12:00:00+00:00",
        "issuer_id": ISSUER,
        "agent_id": "agt_a50_001",
        "action_ref": f"sha256:{digest['hash']}",
        "context": CONTEXT,
        "payload_digest": digest,
        "policy_digest": f"sha256:{_ZERO_DIGEST}",
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
        "mode": "payload",
        "eu_ai_act_articles": ["Article-50"],
        "framework_mappings_self_declared": True,
    }


def _sign(payload: dict, sk: Ed25519PrivateKey) -> dict:
    """The two-key envelope: the anchors member is ABSENT (the T-016 spelling)."""
    return {
        "payload": payload,
        "signature": {
            "alg": "Ed25519",
            "kid": KID,
            "sig": base64.b64encode(sk.sign(_jcs(payload))).decode(),
        },
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
    digest = _digest_of(CONTEXT)
    assert digest["size"] == 35, digest  # the disclosure context's canonical length

    _write(
        "asqav-29-article-50-lifecycle-disclosure",
        {
            "receipt.json": _sign(_payload(digest), sk),
            "jwks.json": _jwks(),
            "expected.json": {
                "format": "asqav-native",
                "outcome": "verified",
                "reason_code": "",
                "notes": (
                    "The Article 50 duty shape named in -09's binding: a "
                    "protectmcp:lifecycle disclosure receipt whose signed payload "
                    "declares eu_ai_act_articles ['Article-50'] with "
                    "framework_mappings_self_declared true. The anchors member is "
                    "absent, the conformant spelling T-016 pins, and the receipt "
                    "verifies like any anchor-less valid receipt (asqav-17's "
                    "outcome family). The verifier does not check framework "
                    "claims: both engine halves declare framework_mapping_claims "
                    "NOT-CHECKED, so the declaration rides the signed bytes "
                    "without changing the outcome."
                ),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
