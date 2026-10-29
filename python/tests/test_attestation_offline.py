"""Anti-vacuous tests for offline attestation verification (criterion 281).

The SDK re-derives the cloud's statementHash byte for byte; a tampered claim or
receipt fails; a good receipt yields "verified_keyed", not plain verified. The
golden statementHash is from the cloud route's canonicalization, not a call.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

import asqav

# Reload only when a runner preloaded an ahead-of-tree asqav; the normal run
# already has this checkout's own module, so nothing is deleted.
if not Path(asqav.__file__).resolve().is_relative_to(_SRC.resolve()):
    for _name in [n for n in sys.modules if n == "asqav" or n.startswith("asqav.")]:
        del sys.modules[_name]
    import asqav

from asqav.attestation import (
    DEFAULT_COMMITMENT_ALG,
    FAILURE_INVALID,
    FAILURE_UNVERIFIABLE,
    SIGNED_MESSAGE_VERSION,
    SIGNED_MESSAGE_VERSION_V1,
    V1_ONLY_FIELDS,
    VERSION_FIELD,
    VERDICT_UNVERIFIED,
    VERDICT_VERIFIED_KEYED,
    compute_statement_hash,
    merkle_inclusion_proof,
    merkle_leaf_hash,
    merkle_root,
    reconstruct_signed_message,
    signed_message_from_receipt,
    signed_message_version,
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

# Captured once from asqav's build_signed_message, never hand-typed (see PR).
_V2_GOLDEN_MESSAGE = (
    b'{"commitment":"abababababababababababababababababababababababababababababababab",'
    b'"commitment_alg":"HMAC-SHA256","key_id":"caller-key-1","mode":"attestation",'
    b'"server_timestamp":"2026-07-26T12:00:00Z","statement_hash":'
    b'"sha256:ffdf04a90b96ede30294d6f4622bf8946fb03e4f2500a201a4e05030886199a0","v":2}'
)

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


    # Current-shape receipt fields: every v2 signed key but mode and v is here.
def _v2_receipt_fields() -> dict:
    return {
        "statementHash": _GOLDEN_HASH,
        "commitment": _COMMITMENT,
        "commitmentAlg": DEFAULT_COMMITMENT_ALG,
        "keyId": _KEY_ID,
        "timestamp": _TIMESTAMP,
    }


    # A v2 receipt signed by a fresh key over the bytes its fields rebuild.
def _v2_receipt():
    sign, verify = _ed25519_signer()
    receipt = _v2_receipt_fields()
    receipt["signature"] = base64.b64encode(
        sign(signed_message_from_receipt(receipt))
    ).decode()
    return receipt, verify


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


    # The v2 rebuild is byte-exact: golden bytes captured from asqav once.
def test_v2_signed_message_matches_cloud_bytes() -> None:
    receipt = _v2_receipt_fields()
    message = signed_message_from_receipt(receipt)
    assert message == _V2_GOLDEN_MESSAGE
    assert signed_message_version(receipt) == SIGNED_MESSAGE_VERSION
    payload = json.loads(message)
    assert payload.keys() == {
        "v", "mode", "statement_hash", "commitment", "commitment_alg",
        "key_id", "server_timestamp",
    }
    assert payload["v"] == SIGNED_MESSAGE_VERSION


    # A legacy receipt keeps rebuilding the exact 11-key v1 bytes it did before.
def test_v1_receipt_still_rebuilds_v1() -> None:
    receipt = {
        "statementHash": _GOLDEN_HASH,
        "commitment": _COMMITMENT,
        "commitmentAlg": DEFAULT_COMMITMENT_ALG,
        "keyId": _KEY_ID,
        "timestamp": _TIMESTAMP,
        **dict(zip(V1_ONLY_FIELDS, (_ACTION_ID, _AGENT_ID, _ORG_ID, _CLAIM["type"]))),
    }
    assert signed_message_version(receipt) == SIGNED_MESSAGE_VERSION_V1
    assert signed_message_from_receipt(receipt) == _signed_message()

    marked = {**receipt, VERSION_FIELD: SIGNED_MESSAGE_VERSION_V1}
    assert signed_message_version(marked) == SIGNED_MESSAGE_VERSION_V1
    assert signed_message_from_receipt(marked) == _signed_message()


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


    # A current v2 receipt verifies from the receipt, the claim and the key.
def test_v2_receipt_verifies_from_receipt_and_claim_alone() -> None:
    receipt, verify = _v2_receipt()
    result = verify_attestation_offline(_CLAIM, receipt, verify_signature=verify)
    assert result["verdict"] == VERDICT_VERIFIED_KEYED
    assert result["failure_class"] is None
    by_name = {a["name"]: a["result"] for a in result["axes"]}
    assert by_name["statement_hash"] == "PASS"
    assert by_name["signature"] == "PASS"


    # A supplied inclusion proof is never waived for want of a tree head.
def test_v2_receipt_with_inclusion_proof_without_tree_head_is_unverified() -> None:
    sign, verify = _ed25519_signer()
    _, proof = _tree_head_and_proof()
    receipt = _v2_receipt_fields()
    receipt["inclusionProof"] = proof
    rebuilt = signed_message_from_receipt(receipt)
    receipt["signature"] = base64.b64encode(sign(rebuilt)).decode()
    for supplied in (None, rebuilt):
        result = verify_attestation_offline(
            _CLAIM, receipt, signed_message=supplied, verify_signature=verify
        )
        assert result["verdict"] == VERDICT_UNVERIFIED
        assert result["failure_class"] == FAILURE_UNVERIFIABLE
        by_name = {a["name"]: a["result"] for a in result["axes"]}
        assert by_name["statement_hash"] == "PASS"
        assert by_name["signature"] == "PASS"
        assert by_name["inclusion"] == "SKIP"


    # Without a proof, explicit bytes and rebuilt bytes reach the same verdict.
def test_explicit_signed_message_matches_rebuilt_verdict_without_proof() -> None:
    receipt, verify = _v2_receipt()
    result = verify_attestation_offline(
        _CLAIM,
        receipt,
        signed_message=signed_message_from_receipt(receipt),
        verify_signature=verify,
    )
    assert result["verdict"] == VERDICT_VERIFIED_KEYED
    assert result["failure_class"] is None
    by_name = {a["name"]: a["result"] for a in result["axes"]}
    assert by_name["statement_hash"] == "PASS"
    assert by_name["signature"] == "PASS"
    assert by_name["inclusion"] == "SKIP"


    # NEGCTL: a flipped commitment with a matching hash breaks the signature.
def test_negctl_one_byte_of_commitment_fails_signature() -> None:
    receipt, verify = _v2_receipt()
    commitment = receipt["commitment"]
    flipped = ("0" if commitment[0] != "0" else "1") + commitment[1:]
    assert flipped != commitment
    receipt["commitment"] = flipped
    receipt["statementHash"] = compute_statement_hash(
        _CLAIM, flipped, receipt["keyId"], receipt["timestamp"]
    )
    result = verify_attestation_offline(_CLAIM, receipt, verify_signature=verify)
    by_name = {a["name"]: a["result"] for a in result["axes"]}
    assert by_name["statement_hash"] == "PASS"
    signature = next(a for a in result["axes"] if a["name"] == "signature")
    assert signature["result"] == "FAIL"
    assert signature["note"] == "signature does not verify over the signed message"
    assert result["verdict"] == VERDICT_UNVERIFIED


    # NEGCTL: v1 bytes never verify a signature the cloud made over v2 bytes.
def test_negctl_v2_receipt_over_v1_bytes_fails() -> None:
    receipt, verify = _v2_receipt()
    result = verify_attestation_offline(
        _CLAIM,
        receipt,
        signed_message=_signed_message(),
        verify_signature=verify,
    )
    by_name = {a["name"]: a["result"] for a in result["axes"]}
    assert by_name["signature"] == "FAIL"
    assert result["verdict"] == VERDICT_UNVERIFIED


    # NEGCTL: a v2 receipt missing a needed field fails naming that field.
def test_negctl_missing_field_is_named() -> None:
    receipt, verify = _v2_receipt()
    field = "commitmentAlg"
    del receipt[field]
    result = verify_attestation_offline(_CLAIM, receipt, verify_signature=verify)
    signature = next(a for a in result["axes"] if a["name"] == "signature")
    assert signature["result"] == "FAIL"
    assert field in signature["note"]
    assert signature["failure_class"] == FAILURE_UNVERIFIABLE
    assert result["verdict"] == VERDICT_UNVERIFIED


def test_public_surface_exposed_at_package_root() -> None:
    import asqav

    assert asqav.verify_attestation_offline is verify_attestation_offline
    assert asqav.compute_statement_hash is compute_statement_hash
    assert asqav.VERDICT_VERIFIED_KEYED == "verified_keyed"
    assert asqav.VERDICT_UNVERIFIED == "unverified"
