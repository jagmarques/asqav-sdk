"""Anti-vacuous tests for offline attestation verification (criterion 281).

Prove the SDK re-derives the cloud's statementHash byte for byte from the
original claim, that a tampered claim or receipt fails, and that a good
receipt yields the distinct "verified_keyed" outcome (commitment not
re-derivable) rather than a plain verified verdict.

The golden statementHash below was captured from the cloud route's merged
canonicalization (src/asqav_cloud/api/routes/attestations.py), not a live call.
"""

from __future__ import annotations

import base64
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.attestation import (
    FAILURE_INVALID,
    FAILURE_UNVERIFIABLE,
    VERDICT_UNVERIFIED,
    VERDICT_VERIFIED_KEYED,
    compute_statement_hash,
    merkle_inclusion_proof,
    merkle_leaf_hash,
    merkle_root,
    reconstruct_signed_message,
    verify_attestation_offline,
)

# Fixture captured from the cloud's merged canonicalization (no live call).
_CLAIM = {
    "type": "vendor.risk_assessment/1",
    "subject": "agt-swarm-9f2c",
    "assertions": {"score": "risk-0.93"},
}
_COMMITMENT = "ab" * 32
_KEY_ID = "caller-key-1"
_TIMESTAMP = "2026-07-26T12:00:00Z"
_GOLDEN_HASH = "sha256:ffdf04a90b96ede30294d6f4622bf8946fb03e4f2500a201a4e05030886199a0"

_ACTION_ID = "act_test"
_AGENT_ID = "agt_test"
_ORG_ID = "org_test"


    # Return (sign(message) -> bytes, verify(message, sig) -> bool) for a fresh key.
def _ed25519_signer():
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric import ed25519

    private = ed25519.Ed25519PrivateKey.generate()
    public = private.public_key()

    def verify(message: bytes, signature: bytes) -> bool:
        try:
            public.verify(signature, message)
            return True
        except InvalidSignature:
            return False

    return private.sign, verify


def _signed_message(claim: dict = _CLAIM) -> bytes:
    statement_hash = compute_statement_hash(claim, _COMMITMENT, _KEY_ID, _TIMESTAMP)
    return reconstruct_signed_message(
        statement_hash=statement_hash,
        commitment=_COMMITMENT,
        commitment_alg="HMAC-SHA256",
        key_id=_KEY_ID,
        claim_type=claim["type"],
        server_timestamp=_TIMESTAMP,
        action_id=_ACTION_ID,
        agent_id=_AGENT_ID,
        org_id=_ORG_ID,
    )


    # Build a small Merkle tree and return (root, inclusionProof dict) for index.
def _tree_head_and_proof(index: int = 2, size: int = 4):
    leaves = [merkle_leaf_hash(f"entry-{i}".encode()) for i in range(size)]
    proof = merkle_inclusion_proof(leaves, index)
    return merkle_root(leaves), {
        "leafIndex": index,
        "treeSize": size,
        "leafHash": leaves[index].hex(),
        "auditPath": [h.hex() for h in proof],
    }


def _good_receipt():
    """A receipt whose hash, signature, and inclusion proof all verify.

    Returns (receipt, message, verify, tree_head) sharing one consistent key
    so each axis can be tampered in isolation.
    """
    sign, verify = _ed25519_signer()
    message = _signed_message()
    tree_head, proof = _tree_head_and_proof()
    receipt = {
        "statementHash": _GOLDEN_HASH,
        "commitment": _COMMITMENT,
        "keyId": _KEY_ID,
        "timestamp": _TIMESTAMP,
        "signature": base64.b64encode(sign(message)).decode(),
        "inclusionProof": proof,
        "logIndex": proof["leafIndex"],
        "log_status": "included",
    }
    return receipt, message, verify, tree_head


    # (b) The SDK re-derives the exact statementHash the backend returned.
def test_compute_statement_hash_matches_cloud_fixture() -> None:
    got = compute_statement_hash(_CLAIM, _COMMITMENT, _KEY_ID, _TIMESTAMP)
    assert got == _GOLDEN_HASH


    # The signed message carries the exact 11 fields the cloud signs.
def test_reconstruct_signed_message_mirrors_cloud_shape() -> None:
    message = json.loads(_signed_message())
    assert message.keys() == {
        "v", "mode", "statement_hash", "commitment", "commitment_alg",
        "key_id", "claim_type", "server_timestamp", "action_id",
        "agent_id", "org_id",
    }
    assert message["v"] == 1
    assert message["mode"] == "attestation"
    assert message["statement_hash"] == _GOLDEN_HASH


    # (c) A tampered claim re-derives a different hash and fails verification.
def test_tampered_claim_changes_hash_and_does_not_pass() -> None:
    tampered = dict(_CLAIM)
    tampered["subject"] = "agt-EVIL"
    assert compute_statement_hash(tampered, _COMMITMENT, _KEY_ID, _TIMESTAMP) != _GOLDEN_HASH

    receipt, message, verify, tree_head = _good_receipt()
    result = verify_attestation_offline(
        tampered,
        receipt,
        signed_message=message,
        verify_signature=verify,
        signed_tree_head=tree_head,
    )
    assert result["verdict"] == VERDICT_UNVERIFIED
    assert result["failure_class"] == FAILURE_INVALID
    assert result["statementHash"] != _GOLDEN_HASH
    by_name = {a["name"]: a["result"] for a in result["axes"]}
    assert by_name["statement_hash"] == "FAIL"


    # (d) A good receipt yields verified_keyed, never plain verified (438).
def test_good_receipt_is_not_rederivable_never_plain_verified() -> None:
    receipt, message, verify, tree_head = _good_receipt()
    result = verify_attestation_offline(
        _CLAIM,
        receipt,
        signed_message=message,
        verify_signature=verify,
        signed_tree_head=tree_head,
    )
    assert result["verdict"] == VERDICT_VERIFIED_KEYED
    assert result["verdict"] != "verified"
    assert result["failure_class"] is None
    by_name = {a["name"]: a["result"] for a in result["axes"]}
    assert by_name["statement_hash"] == "PASS"
    assert by_name["signature"] == "PASS"
    assert by_name["inclusion"] == "PASS"
    assert by_name["commitment"] == "NOT_REDERIVABLE"


    # Flipping the signature bytes fails the signature axis and the verdict.
def test_tampered_signature_fails() -> None:
    receipt, message, verify, tree_head = _good_receipt()
    raw = base64.b64decode(receipt["signature"])
    flipped = bytes([raw[0] ^ 0xFF]) + raw[1:]
    receipt["signature"] = base64.b64encode(flipped).decode()

    result = verify_attestation_offline(
        _CLAIM,
        receipt,
        signed_message=message,
        verify_signature=verify,
        signed_tree_head=tree_head,
    )
    assert result["verdict"] == VERDICT_UNVERIFIED
    assert result["failure_class"] == FAILURE_INVALID
    by_name = {a["name"]: a["result"] for a in result["axes"]}
    assert by_name["signature"] == "FAIL"
    assert by_name["statement_hash"] == "PASS"


    # A proof cut against the wrong tree head fails the inclusion axis.
def test_tampered_inclusion_proof_fails() -> None:
    receipt, message, verify, _ = _good_receipt()
    wrong_head = merkle_root([merkle_leaf_hash(b"other-leaf")])
    result = verify_attestation_offline(
        _CLAIM,
        receipt,
        signed_message=message,
        verify_signature=verify,
        signed_tree_head=wrong_head,
    )
    assert result["verdict"] == VERDICT_UNVERIFIED
    assert result["failure_class"] == FAILURE_INVALID
    by_name = {a["name"]: a["result"] for a in result["axes"]}
    assert by_name["inclusion"] == "FAIL"
    assert by_name["signature"] == "PASS"


    # Without a key or tree head the verdict is unverified/unverifiable (418).
def test_missing_optional_inputs_is_unverifiable_not_verified() -> None:
    receipt, _message, _verify, _tree_head = _good_receipt()
    result = verify_attestation_offline(_CLAIM, receipt)
    assert result["verdict"] == VERDICT_UNVERIFIED
    assert result["failure_class"] == FAILURE_UNVERIFIABLE
    by_name = {a["name"]: a["result"] for a in result["axes"]}
    assert by_name["statement_hash"] == "PASS"
    assert by_name["signature"] == "SKIP"
    assert by_name["inclusion"] == "SKIP"
    assert by_name["commitment"] == "NOT_REDERIVABLE"


def test_public_surface_exposed_at_package_root() -> None:
    import asqav

    assert asqav.verify_attestation_offline is verify_attestation_offline
    assert asqav.compute_statement_hash is compute_statement_hash
    assert asqav.VERDICT_VERIFIED_KEYED == "verified_keyed"
    assert asqav.VERDICT_UNVERIFIED == "unverified"
