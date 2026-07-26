"""Offline verification for opaque-claim attestation receipts.

Mirrors the statement-hash canonicalization the asqav cloud pins for
POST /attestations, so a caller holding the ORIGINAL claim can re-derive
the statement hash a receipt commits to, check asqav's signature over the
signed message, and check the inclusion proof against a signed tree head.

The verdict is deliberately distinct from a plain pass. The receipt's
commitment is an HMAC sealed with a key only the holder has, so a verifier
can confirm the commitment was hashed and signed but can never re-derive it
from the claim. A good receipt therefore yields a "verified, commitment not
re-derivable" outcome, never a plain PASS.

The statement-hash and signed-message shapes mirror the cloud route
src/asqav_cloud/api/routes/attestations.py byte for byte. The Merkle helpers
mirror src/asqav_cloud/core/merkle.py (certificate-transparency 6962-style).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import Callable
from typing import Any

from ._jcs import canonical_json

__all__ = [
    "ATTESTATION_STATEMENT_SCHEMA",
    "VERDICT_VERIFIED_NOT_REDERIVABLE",
    "compute_statement_hash",
    "reconstruct_signed_message",
    "merkle_leaf_hash",
    "merkle_node_hash",
    "merkle_root",
    "merkle_inclusion_proof",
    "verify_merkle_inclusion",
    "verify_attestation_offline",
]

# Pinned by the cloud route. A change here silently breaks every verification.
ATTESTATION_STATEMENT_SCHEMA = "asqav.attestation/1"
DEFAULT_COMMITMENT_ALG = "HMAC-SHA256"

#: Distinct outcome for a good attestation receipt. Never a plain PASS.
VERDICT_VERIFIED_NOT_REDERIVABLE = "VERIFIED_COMMITMENT_NOT_REDERIVABLE"
VERDICT_FAIL = "FAIL"
VERDICT_INCOMPLETE = "INCOMPLETE"

_AXIS_PASS = "PASS"
_AXIS_FAIL = "FAIL"
_AXIS_SKIP = "SKIP"
_AXIS_NOT_REDERIVABLE = "NOT_REDERIVABLE"

# Merkle domain separation, mirroring the cloud (6962-style second-preimage guard).
_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"


def compute_statement_hash(
    claim: dict[str, Any],
    commitment: str,
    key_id: str,
    timestamp: str,
    commitment_alg: str = DEFAULT_COMMITMENT_ALG,
) -> str:
    """Re-derive the receipt's statementHash from the ORIGINAL claim.

    Byte-for-byte mirror of the cloud's statement payload canonicalization.
    Returns the self-describing ``sha256:<hex>`` form.
    """
    statement_payload = {
        "v": 1,
        "type": ATTESTATION_STATEMENT_SCHEMA,
        "claim": claim,
        "commitment": commitment,
        "commitmentAlg": commitment_alg,
        "keyId": key_id,
        "timestamp": timestamp,
    }
    return "sha256:" + hashlib.sha256(canonical_json(statement_payload)).hexdigest()


def reconstruct_signed_message(
    *,
    statement_hash: str,
    commitment: str,
    commitment_alg: str,
    key_id: str,
    claim_type: str,
    server_timestamp: str,
    action_id: str,
    agent_id: str,
    org_id: str,
) -> bytes:
    """Return the canonical signed-message bytes the cloud agent signed.

    Byte-for-byte mirror of the cloud's message payload. A caller rebuilds
    this from the receipt plus its own action_id/agent_id/org_id records and
    verifies the receipt signature over these exact bytes.
    """
    message_payload = {
        "v": 1,
        "mode": "attestation",
        "statement_hash": statement_hash,
        "commitment": commitment,
        "commitment_alg": commitment_alg,
        "key_id": key_id,
        "claim_type": claim_type,
        "server_timestamp": server_timestamp,
        "action_id": action_id,
        "agent_id": agent_id,
        "org_id": org_id,
    }
    return canonical_json(message_payload)


# === Pure Merkle helpers (certificate-transparency 6962-style) ===


def merkle_leaf_hash(entry: bytes) -> bytes:
    """SHA256(0x00 || entry). Mirrors the cloud's leaf domain separation."""
    return hashlib.sha256(_LEAF_PREFIX + entry).digest()


def merkle_node_hash(left: bytes, right: bytes) -> bytes:
    """SHA256(0x01 || left || right). Mirrors the cloud's internal node hash."""
    return hashlib.sha256(_NODE_PREFIX + left + right).digest()


def _largest_power_of_two_below(n: int) -> int:
    """Largest power of two strictly less than n (n > 1). The 6962 split point."""
    k = 1
    while k << 1 < n:
        k <<= 1
    return k


def merkle_root(leaves: list[bytes]) -> bytes:
    """Merkle tree hash over already-hashed leaves. Empty tree hashes SHA256(b"")."""
    n = len(leaves)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return leaves[0]
    k = _largest_power_of_two_below(n)
    return merkle_node_hash(merkle_root(leaves[:k]), merkle_root(leaves[k:]))


def merkle_inclusion_proof(leaves: list[bytes], index: int) -> list[bytes]:
    """Audit path (sibling hashes, leaf-up) for the leaf at ``index``."""
    n = len(leaves)
    if index < 0 or index >= n:
        raise IndexError(f"index {index} out of range for tree size {n}")
    if n == 1:
        return []
    k = _largest_power_of_two_below(n)
    if index < k:
        return merkle_inclusion_proof(leaves[:k], index) + [merkle_root(leaves[k:])]
    return merkle_inclusion_proof(leaves[k:], index - k) + [merkle_root(leaves[:k])]


def verify_merkle_inclusion(
    leaf: bytes,
    index: int,
    tree_size: int,
    proof: list[bytes],
    root: bytes,
) -> bool:
    """Verify an inclusion proof offline (6962 algorithm). False on any mismatch."""
    if index < 0 or index >= tree_size or tree_size <= 0:
        return False
    if tree_size == 1:
        return len(proof) == 0 and hmac.compare_digest(leaf, root)

    fn, sn = index, tree_size - 1
    r = leaf
    for sibling in proof:
        if sn == 0:
            return False
        if (fn & 1) or fn == sn:
            r = merkle_node_hash(sibling, r)
            if not (fn & 1):
                while fn != 0 and not (fn & 1):
                    fn >>= 1
                    sn >>= 1
        else:
            r = merkle_node_hash(r, sibling)
        fn >>= 1
        sn >>= 1
    return sn == 0 and hmac.compare_digest(r, root)


# === Offline attestation verification ===


def _axis(name: str, result: str, note: str) -> dict[str, str]:
    return {"name": name, "result": result, "note": note}


def verify_attestation_offline(
    claim: dict[str, Any],
    receipt: dict[str, Any],
    *,
    signed_message: bytes | None = None,
    verify_signature: Callable[[bytes, bytes], bool] | None = None,
    signed_tree_head: bytes | None = None,
    commitment_alg: str = DEFAULT_COMMITMENT_ALG,
) -> dict[str, Any]:
    """Verify an attestation receipt offline from the ORIGINAL claim.

    Recomputes statementHash with the cloud's exact canonicalization, checks
    asqav's signature over the signed message, and checks the inclusion proof
    against a supplied signed tree head. The commitment is an HMAC sealed with
    a holder-only key, so it can never be re-derived here. A good receipt
    yields VERDICT_VERIFIED_NOT_REDERIVABLE, never a plain PASS.

    Args:
        claim: The original claim dict the caller submitted.
        receipt: The receipt dict (camelCase wire keys: statementHash,
            commitment, keyId, timestamp, signature, inclusionProof).
        signed_message: The canonical signed-message bytes (rebuild with
            reconstruct_signed_message). Optional, signature axis SKIPs without it.
        verify_signature: Callable(message, signature) -> bool for the agent key.
            Optional, signature axis SKIPs without it.
        signed_tree_head: The Merkle root the inclusion proof is checked against.
            Optional, inclusion axis SKIPs without it or without a proof.
        commitment_alg: Commitment algorithm id. Defaults to HMAC-SHA256.

    Returns:
        dict with ``verdict`` (VERIFIED_COMMITMENT_NOT_REDERIVABLE | FAIL |
        INCOMPLETE), ``axes`` (per-axis name/result/note), and ``statementHash``
        (the recomputed value).
    """
    axes: list[dict[str, str]] = []

    recomputed = compute_statement_hash(
        claim,
        receipt.get("commitment", ""),
        receipt.get("keyId", ""),
        receipt.get("timestamp", ""),
        commitment_alg,
    )
    receipt_hash = receipt.get("statementHash", "")
    if recomputed == receipt_hash:
        axes.append(_axis("statement_hash", _AXIS_PASS, "recomputed from the original claim"))
    else:
        axes.append(_axis("statement_hash", _AXIS_FAIL, "claim does not match the receipt hash"))

    # Signature axis: verify over the exact signed message and bind it to the claim.
    if signed_message is None or verify_signature is None:
        axes.append(_axis("signature", _AXIS_SKIP, "no signed message or verifier supplied"))
    else:
        sig_result, sig_note = _check_signature(
            signed_message, receipt, recomputed, verify_signature
        )
        axes.append(_axis("signature", sig_result, sig_note))

    # Inclusion axis: check the proof against the supplied signed tree head.
    proof = receipt.get("inclusionProof")
    if signed_tree_head is None or proof is None:
        axes.append(_axis("inclusion", _AXIS_SKIP, "no signed tree head or proof supplied"))
    else:
        incl_ok = verify_merkle_inclusion(
            bytes.fromhex(proof["leafHash"]),
            int(proof["leafIndex"]),
            int(proof["treeSize"]),
            [bytes.fromhex(h) for h in proof["auditPath"]],
            signed_tree_head,
        )
        if incl_ok:
            axes.append(_axis("inclusion", _AXIS_PASS, "proof verifies against the tree head"))
        else:
            axes.append(_axis("inclusion", _AXIS_FAIL, "proof does not match the tree head"))

    # Commitment axis: the holder key is required and absent, so this is structural.
    axes.append(
        _axis(
            "commitment",
            _AXIS_NOT_REDERIVABLE,
            "HMAC needs the holder key, which the verifier never has",
        )
    )

    return {
        "verdict": _verdict(axes),
        "axes": axes,
        "statementHash": recomputed,
    }


def _check_signature(
    signed_message: bytes,
    receipt: dict[str, Any],
    recomputed_hash: str,
    verify_signature: Callable[[bytes, bytes], bool],
) -> tuple[str, str]:
    """Verify the receipt signature over the signed message and bind it to the claim."""
    try:
        message_obj = json.loads(signed_message)
    except (ValueError, TypeError):
        return _AXIS_FAIL, "signed message is not valid canonical JSON"
    if message_obj.get("statement_hash") != recomputed_hash:
        return _AXIS_FAIL, "signed message does not bind to the original claim"
    try:
        signature = base64.b64decode(receipt.get("signature", ""))
    except (ValueError, TypeError):
        return _AXIS_FAIL, "receipt signature is not valid base64"
    if verify_signature(signed_message, signature):
        return _AXIS_PASS, "signature verifies over the signed message"
    return _AXIS_FAIL, "signature does not verify over the signed message"


def _verdict(axes: list[dict[str, str]]) -> str:
    """Map per-axis results to a verdict. A good receipt is never a plain PASS."""
    results = {a["name"]: a["result"] for a in axes}
    if any(r == _AXIS_FAIL for r in results.values()):
        return VERDICT_FAIL
    checked = (
        results.get("statement_hash") == _AXIS_PASS
        and results.get("signature") == _AXIS_PASS
        and results.get("inclusion") == _AXIS_PASS
    )
    if checked:
        return VERDICT_VERIFIED_NOT_REDERIVABLE
    return VERDICT_INCOMPLETE
