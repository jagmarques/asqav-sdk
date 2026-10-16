"""Deterministic signed-conflict fixtures (488 C21, C7 core).

Two did:key agentreceipts geneses over one subject (same principal and
action) with conflicting outcomes, signed by one fixed TEST-ONLY key.
Re-running reproduces both files byte-for-byte; the seed is published
below and must never protect anything real.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

from asqav.verifier.oracle.canonical import jcs_rfc8785  # noqa: E402

# Fixed TEST-ONLY seed. Publishes the did:key in README; signs fixtures only.
TEST_SEED = bytes.fromhex("c7" * 32)
ISSUED = "2026-10-16T12:00:00Z"
PRINCIPAL = "did:user:alice"
ACTION_ID = "act_c7_conflict_read_001"
OUT_DIR = Path(__file__).resolve().parent / "signed-conflict-fixtures"

_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _base58btc(raw: bytes) -> str:
    """Bitcoin base58, leading-zero bytes as '1's (multibase 'z' body)."""
    value = int.from_bytes(raw, "big")
    out = ""
    while value:
        value, rem = divmod(value, 58)
        out = _B58[rem] + out
    zeros = len(raw) - len(raw.lstrip(b"\x00"))
    return "1" * zeros + (out or "")


def _did_and_key() -> tuple[str, bytes, bytes]:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.from_private_bytes(TEST_SEED)
    public = private.public_key().public_bytes_raw()
    did = "did:key:z" + _base58btc(b"\xed\x01" + public)
    return did, TEST_SEED, public


def _sign(private_seed: bytes, doc: dict) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.from_private_bytes(private_seed)
    sig = key.sign(jcs_rfc8785({k: v for k, v in doc.items() if k != "proof"}))
    return "u" + base64.urlsafe_b64encode(sig).decode().rstrip("=")


def _receipt(label: str, outcome: dict, did: str, seed: bytes) -> dict:
    doc = {
        "@context": ["https://www.w3.org/ns/credentials/v2", "https://agentreceipts.ai/context/v1"],
        "id": f"urn:receipt:c7-conflict-{label}",
        "type": ["VerifiableCredential", "AgentReceipt"],
        "version": "0.1.0",
        "issuer": {"id": did},
        "issuanceDate": ISSUED,
        "credentialSubject": {
            "principal": {"id": PRINCIPAL},
            "action": {
                "id": ACTION_ID,
                "type": "filesystem.file.read",
                "risk_level": "low",
                "timestamp": ISSUED,
            },
            "outcome": outcome,
            "chain": {"sequence": 1, "previous_receipt_hash": None, "chain_id": f"chain_c7_{label}"},
        },
        "proof": {
            "type": "Ed25519Signature2020",
            "created": ISSUED,
            "verificationMethod": did + "#" + did.split("did:key:")[1],
            "proofPurpose": "assertionMethod",
            "proofValue": "",
        },
    }
    doc["proof"]["proofValue"] = _sign(seed, doc)
    return doc


def main() -> None:
    did, seed, _ = _did_and_key()
    OUT_DIR.mkdir(exist_ok=True)
    pair = {
        "conflict-allow.json": {"status": "success"},
        "conflict-deny.json": {"status": "failure", "reason": "policy_denied"},
    }
    for name, outcome in pair.items():
        label = name.split(".")[0].split("-")[1]
        path = OUT_DIR / name
        path.write_text(json.dumps(_receipt(label, outcome, did, seed), indent=1) + "\n")
        print(f"wrote {path}")
    print(f"test did:key: {did}")


if __name__ == "__main__":
    main()
