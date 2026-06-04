"""Generate the ACTA conformance vectors for the oracle corpus.

Deterministic: a fixed Ed25519 test seed so the vectors are reproducible and
reviewable. Produces genesis (PASS), chain-link (PASS), tamper-sig (FAIL), and a
commitment-mode receipt (FAIL under the baseline verifier - the honest outcome
for an unsupported optional mode). Signs exactly what the ACTA adapter verifies:
Ed25519 over jcs(payload), hex-encoded.

Run from the verifier/ dir: python conformance-vectors/gen_acta_vectors.py
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oracle.canonical import jcs  # noqa: E402

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

_HERE = Path(__file__).resolve().parent
_SEED = bytes.fromhex("00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff")
_KID = "sb:issuer:acta-oracle-vec-key"

_sk = Ed25519PrivateKey.from_private_bytes(_SEED)
_pk_raw = _sk.public_key().public_bytes_raw()


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _keyset() -> dict:
    return {"keys": [{"kty": "OKP", "crv": "Ed25519", "kid": _KID, "x": _b64url(_pk_raw)}]}


def _sign(payload: dict) -> dict:
    sig = _sk.sign(jcs(payload))
    return {"payload": payload, "signature": {"alg": "EdDSA", "kid": _KID, "sig": sig.hex()}}


def _genesis_payload() -> dict:
    return {
        "type": "protectmcp:decision",
        "issued_at": "2026-05-04T12:00:00+00:00",
        "issuer_id": _KID,
        "agent_id": "agt_acta_001",
        "action_ref": "sha256:" + "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "decision": "allow",
    }


def _write(name: str, files: dict) -> None:
    d = _HERE / name
    d.mkdir(exist_ok=True)
    for fname, obj in files.items():
        (d / fname).write_text(json.dumps(obj, indent=2) + "\n")


def main() -> None:
    keyset = _keyset()

    genesis = _sign(_genesis_payload())
    _write(
        "acta-01-genesis",
        {"receipt.json": genesis, "acta-keys.json": keyset, "expected.json": {"verdict": "PASS"}},
    )

    succ_payload = _genesis_payload()
    succ_payload["agent_id"] = "agt_acta_002"
    succ_payload["previousReceiptHash"] = hashlib.sha256(jcs(genesis)).hexdigest()
    successor = _sign(succ_payload)
    _write(
        "acta-02-chain-link",
        {
            "receipt.json": successor,
            "predecessor.json": genesis,
            "acta-keys.json": keyset,
            "expected.json": {"verdict": "PASS"},
        },
    )

    tampered = _sign(_genesis_payload())
    bad = bytearray(bytes.fromhex(tampered["signature"]["sig"]))
    bad[0] ^= 0x01
    tampered["signature"]["sig"] = bad.hex()
    _write(
        "acta-03-tamper-sig",
        {"receipt.json": tampered, "acta-keys.json": keyset, "expected.json": {"verdict": "FAIL"}},
    )

    # Optional commitment mode signs SHA-256(JCS); the baseline verifier MUST fail it, never pass.
    commit_payload = _genesis_payload()
    commit_payload["agent_id"] = "agt_acta_commit"
    digest = hashlib.sha256(jcs(commit_payload)).digest()
    commit = {
        "payload": commit_payload,
        "signature": {"alg": "EdDSA", "kid": _KID, "sig": _sk.sign(digest).hex()},
    }
    _write(
        "acta-05-commitment-mode-unsupported",
        {"receipt.json": commit, "acta-keys.json": keyset, "expected.json": {"verdict": "FAIL"}},
    )

    manifest_path = _HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest = [m for m in manifest if not m["dir"].startswith("acta-")]
    manifest += [
        {"dir": "acta-01-genesis", "format": "acta", "outcome": "PASS", "reason_code": "",
         "notes": "Valid ACTA genesis; Ed25519 over JCS(payload) verifies."},
        {"dir": "acta-02-chain-link", "format": "acta", "outcome": "PASS", "reason_code": "",
         "notes": "ACTA successor links via SHA-256 of the JCS of the full predecessor receipt."},
        {"dir": "acta-03-tamper-sig", "format": "acta", "outcome": "FAIL", "reason_code": "sig_mismatch",
         "notes": "Flipped signature byte; Ed25519 verify fails."},
        {"dir": "acta-05-commitment-mode-unsupported", "format": "acta", "outcome": "FAIL",
         "reason_code": "sig_mismatch",
         "notes": "Commitment-mode receipt (signs SHA-256(JCS)) fails the baseline JCS verifier - honest, never a false pass."},
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote ACTA vectors; manifest now {len(manifest)} entries")


if __name__ == "__main__":
    main()
