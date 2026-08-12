"""A multi-agent org's receipts resolve the actual signer, not the first sibling.

Criterion 232. A cloud hash-mode receipt sets signature.kid to the ORG id and signs
with the agent's own key, while the directory publishes every key the org owns under
that same issuer_id/org_id. A kid-shaped match therefore answers for the whole sibling
set and list position picks one, so a receipt signed by any sibling but the first used
to verify against the wrong key and read as a false FAIL. These gates pin the agent
bind (agent_id plus the org_id the receipt signs) that resolves the correct signer.

ANTI-VACUOUS: the signer here is the LAST of three siblings; against the pre-change
kid-first resolution the signature axis checks the FIRST sibling and FAILs, so the
PASS assertions only hold once the org-bound agent match resolves the real signer.
"""

from __future__ import annotations

import base64

import pytest

from asqav.verifier import verify_receipt as v
from asqav.verifier.oracle import crypto
from asqav.verifier.oracle.adapters.asqav_native import AsqavNativeAdapter
from asqav.verifier.oracle.canonical import asqav_jcs
from asqav.verifier.oracle.core import verify

_ED25519_AVAILABLE = crypto.verify_ed25519(b"\x00" * 32, b"m", b"\x00" * 64)[0] != crypto.SKIPPED

requires_ed25519 = pytest.mark.skipif(
    not _ED25519_AVAILABLE, reason="cryptography not installed; Ed25519 verify SKIPs"
)

ORG = "org_multikey_shared"


    # A hash-mode signing payload naming its agent and org inside the signed bytes.
def _flat(agent_id: str) -> dict:
    return {
        "v": 1,
        "mode": "hash",
        "hash": "c" * 64,
        "hash_algo": "sha256",
        "metadata": {},
        "server_timestamp": "2026-01-01T00:00:00Z",
        "action_id": "act_1",
        "agent_id": agent_id,
        "org_id": ORG,
        "policy_digest": "d" * 64,
        "policy_decision": "allow",
    }


    # Build the hash-mode receipt envelope signed by sk over the canonical flat bytes.
def _sign(flat: dict, sk) -> dict:
    sig = base64.b64encode(sk.sign(asqav_jcs(flat))).decode()
    doc = dict(flat)
    doc["payload"] = None
    doc["algorithm"] = "Ed25519"
    doc["key_id"] = ORG
    doc["signature_b64"] = sig
    return doc


    # Three sibling keys sharing issuer_id/org_id; agent_id uniquely names each.
def _sibling_jwks(signer_index: int, public_keys: list[bytes]) -> dict:
    keys = []
    for i, pk in enumerate(public_keys):
        keys.append(
            {
                "kid": f"crypto_kid_{i}",
                "agent_id": f"agt_{i}",
                "issuer_id": ORG,
                "org_id": ORG,
                "alg": "Ed25519",
                "public_key": base64.b64encode(pk).decode(),
                "status": "active",
            }
        )
    return {"keys": keys}


    # A receipt signed by the last of three siblings verifies to PASS, not a false FAIL.
@requires_ed25519
def test_last_sibling_receipt_resolves_the_actual_signer() -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sks = [Ed25519PrivateKey.generate() for _ in range(3)]
    pks = [sk.public_key().public_bytes_raw() for sk in sks]
    flat = _flat("agt_2")
    receipt = _sign(flat, sks[2])
    jwks = _sibling_jwks(2, pks)

    result = verify(receipt, [AsqavNativeAdapter()], key_provider=jwks)
    assert result.axis("signature").result == "PASS", result.axes
    assert result.verdict == "verified", result.axes
    assert result.failure_class is None, result.axes


    # Whichever sibling signs, the agent bind resolves that sibling's key, never another.
@requires_ed25519
def test_each_sibling_resolves_to_its_own_key() -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sks = [Ed25519PrivateKey.generate() for _ in range(3)]
    pks = [sk.public_key().public_bytes_raw() for sk in sks]
    jwks = _sibling_jwks(0, pks)
    ad = AsqavNativeAdapter()
    for i, sk in enumerate(sks):
        receipt = _sign(_flat(f"agt_{i}"), sk)
        pk, note = ad.resolve_key(receipt, jwks)
        assert pk == pks[i], (i, note)
        assert verify(receipt, [ad], key_provider=jwks).verdict == "verified"


    # A signature from a key the directory does not publish fails, never a false PASS.
@requires_ed25519
def test_forged_signature_against_the_org_still_fails() -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sks = [Ed25519PrivateKey.generate() for _ in range(3)]
    pks = [sk.public_key().public_bytes_raw() for sk in sks]
    attacker = Ed25519PrivateKey.generate()
    receipt = _sign(_flat("agt_2"), attacker)
    jwks = _sibling_jwks(2, pks)

    result = verify(receipt, [AsqavNativeAdapter()], key_provider=jwks)
    assert result.verdict == "unverified", result.axes
    assert result.failure_class == "invalid", result.axes


    # agent_id is attacker-controlled: a key from another org never resolves the claim.
@requires_ed25519
def test_cross_org_agent_bind_does_not_match() -> None:
    entry = v._match_key_by_agent(
        {
            "keys": [
                {
                    "kid": "foreign",
                    "agent_id": "agt_2",
                    "issuer_id": "some-other-org",
                    "org_id": "some-other-org",
                    "public_key": "QUFBQQ==",
                    "status": "active",
                }
            ]
        },
        "agt_2",
        None,
        ORG,
    )
    assert entry is None, "a foreign-org key must not satisfy the org binding"
