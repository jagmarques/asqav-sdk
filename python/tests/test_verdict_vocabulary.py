"""Shipped verdict vocabulary + exit-code contract (criteria 418/438).

The public surfaces speak verified / verified_keyed / unverified, and every
unverified verdict carries a failure_class of invalid or unverifiable - the two
are never collapsed. Exit codes keep the stable mapping: verified/verified_keyed
-> 0, unverified+invalid -> 1, unverified+unverifiable -> 2 (the blocked state
the old INCOMPLETE verdict carried).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from asqav.verifier.oracle.__main__ import _exit_code, run as cli_run
from asqav.verifier.oracle import verify
from asqav.verifier.oracle.adapters.asqav_native import AsqavNativeAdapter
from asqav.verifier.oracle.canonical import asqav_jcs
from asqav.verifier.oracle.core import (
    FAILURE_INVALID,
    FAILURE_UNVERIFIABLE,
    VERDICT_UNVERIFIED,
    VERDICT_VERIFIED,
    VERDICT_VERIFIED_KEYED,
)

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"


    # The exit-code mapping is the documented CLI contract.
def test_exit_code_mapping() -> None:
    assert _exit_code(VERDICT_VERIFIED, None) == 0
    assert _exit_code(VERDICT_VERIFIED_KEYED, None) == 0
    assert _exit_code(VERDICT_UNVERIFIED, FAILURE_INVALID) == 1
    assert _exit_code(VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE) == 2


    # A verified corpus vector exits 0 through the CLI entry point.
def test_cli_exits_zero_for_verified() -> None:
    vec = _CORPUS / "asqav-01-genesis-permit"
    code = cli_run(str(vec / "receipt.json"), str(vec / "jwks.json"), None)
    assert code == 0


    # A tampered-signature vector exits 1 (a proven binding failure, invalid).
def test_cli_exits_one_for_invalid() -> None:
    vec = _CORPUS / "asqav-04-tamper-sig"
    code = cli_run(str(vec / "receipt.json"), str(vec / "jwks.json"), None)
    assert code == 1


    # A duplicate-member receipt exits 2 (terminal parse failure, unverifiable).
def test_cli_exits_two_for_unverifiable_parse_failure() -> None:
    vec = _CORPUS / "asqav-11-dup-member-toplevel"
    # The CLI raises SystemExit(2) from _load before any hashing runs.
    with pytest.raises(SystemExit) as exc:
        cli_run(str(vec / "receipt.json"), str(vec / "jwks.json"), None)
    assert exc.value.code == 2


    # The structured CLI report speaks the shipped vocabulary.
def test_cli_report_uses_shipped_vocabulary(capsys: pytest.CaptureFixture) -> None:
    vec = _CORPUS / "asqav-04-tamper-sig"
    cli_run(str(vec / "receipt.json"), str(vec / "jwks.json"), None)
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "unverified"
    assert report["failure_class"] == "invalid"
    # Per-axis rows carry their failure token too.
    for ax in report["axes"]:
        assert "failure_class" in ax


    # Build a signed hash-mode receipt whose digest is keyed (hmac-sha256).
def _keyed_hash_mode_receipt():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    org = "f94f66c0-c580-432d-a041-29374f7aee07"
    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    flat = {
        "v": 1,
        "mode": "hash",
        "hash": "sha256:" + "a" * 64,
        "hash_algo": "hmac-sha256",
        "metadata": {},
        "server_timestamp": "2026-01-01T00:00:00Z",
        "action_id": "act_1",
        "agent_id": "agt_1",
        "org_id": org,
        "policy_digest": "sha256:" + "c" * 64,
        "policy_decision": "allow",
    }
    doc = dict(flat)
    doc["payload"] = None
    doc["algorithm"] = "Ed25519"
    doc["key_id"] = org
    doc["signature_b64"] = base64.b64encode(sk.sign(asqav_jcs(flat))).decode()
    jwks = {
        "keys": [
            {
                "kid": "keyed-key",
                "agent_id": "agt_1",
                "issuer_id": org,
                "org_id": org,
                "alg": "Ed25519",
                "public_key": base64.b64encode(pk).decode(),
                "status": "active",
            }
        ]
    }
    return doc, jwks


    # A keyed digest that fully checks reports verified_keyed, never verified (438).
def test_keyed_digest_reports_verified_keyed_never_plain_verified() -> None:
    pytest.importorskip("cryptography")
    doc, jwks = _keyed_hash_mode_receipt()
    res = verify(doc, [AsqavNativeAdapter()], key_provider=jwks)
    assert res.verdict == VERDICT_VERIFIED_KEYED
    assert res.failure_class is None


    # The same receipt with a plain sha256 digest reports verified, not keyed.
def test_plain_digest_reports_verified_not_keyed() -> None:
    pytest.importorskip("cryptography")
    doc, jwks = _keyed_hash_mode_receipt()
    # Re-sign with hash_algo flipped to sha256 so the signature still binds.
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    flat = {k: v for k, v in doc.items() if k not in ("payload", "algorithm", "key_id", "signature_b64")}
    flat["hash_algo"] = "sha256"
    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    flat_signed = dict(flat)
    flat_signed["payload"] = None
    flat_signed["algorithm"] = "Ed25519"
    flat_signed["key_id"] = flat["org_id"]
    flat_signed["signature_b64"] = base64.b64encode(sk.sign(asqav_jcs(flat))).decode()
    jwks["keys"][0]["public_key"] = base64.b64encode(pk).decode()
    res = verify(flat_signed, [AsqavNativeAdapter()], key_provider=jwks)
    assert res.verdict == VERDICT_VERIFIED
    assert res.failure_class is None
