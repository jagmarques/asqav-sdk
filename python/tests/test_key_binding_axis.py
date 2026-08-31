"""key_thumbprint binding axis, Python half (criterion 458).

The shared table in verifier/axis-parity-cases.json drives this file and
typescript/tests/verifier-key-binding-parity.test.ts, so a rule that drifts in one
language fails a suite.

Beyond the table, the wire tests here are mutation-shaped on purpose: each one
fails if its own wire is deleted. Criterion 457's lesson was that driving only the
helper chain leaves the call sites untested, so every place the axis is appended
and every table entry that makes its FAIL terminal has a test that notices its
removal.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from asqav.verifier import verify_receipt as vr
from asqav.verifier.oracle import ADAPTERS
from asqav.verifier.oracle import verify as oracle_verify
from asqav.verifier.oracle.adapters.asqav_native import _UNSIGNED_CLAIM_FIELDS
from asqav.verifier.oracle.core import _INVALID_FAIL_AXES as ORACLE_INVALID_FAIL_AXES

CASES_FILE = Path(__file__).parent.parent.parent / "verifier" / "axis-parity-cases.json"
TABLE = json.loads(CASES_FILE.read_text())

ALG = "ML-DSA-65"
KEY_GOOD = bytes([0x00]) * 1952
KEY_EVIL = bytes([0x01]) * 1952
TP_GOOD = vr.thumbprint_for_key(alg=ALG, public_key=KEY_GOOD)
TP_EVIL = vr.thumbprint_for_key(alg=ALG, public_key=KEY_EVIL)


    # Expand a table `key` spec into the bytes both languages build it from.
def _expand(spec):
    if spec is None:
        return None, None
    return spec["alg"], bytes([spec["fill"]]) * spec["len"]


def _jwks(public_key: bytes, alg: str = ALG) -> dict:
    return {
        "keys": [
            {
                "kid": "iss_1",
                "issuer_id": "iss_1",
                "agent_id": "ag_1",
                "org_id": "org_1",
                "alg": alg,
                "public_key": base64.b64encode(public_key).decode(),
                "status": "active",
            }
        ]
    }


def _receipt(thumbprint=None) -> dict:
    payload = {
        "v": 1,
        "issuer_id": "iss_1",
        "agent_id": "ag_1",
        "org_id": "org_1",
        "previousReceiptHash": vr.FIRST_RECEIPT_SEED,
        "issued_at": "2026-08-31T10:00:00Z",
        "action": {"type": "x"},
        "hash": "a" * 64,
    }
    if thumbprint is not None:
        payload["key_thumbprint"] = thumbprint
    return {
        "payload": payload,
        "signature": {"alg": ALG, "kid": "iss_1", "sig": base64.b64encode(b"\x00" * 64).decode()},
        "anchors": {},
    }


# --------------------------------------------------------------------------
# The shared cross-language table
# --------------------------------------------------------------------------


    # A table that silently empties would make every case below vacuous.
def test_table_is_populated() -> None:
    assert len(TABLE["key_binding"]) >= 15, f"only {len(TABLE['key_binding'])} cases"
    assert len(TABLE["key_thumbprint_vectors"]) >= 4
    # A table of one outcome cannot tell a working axis from a stuck one.
    assert {c["expect"]["result"] for c in TABLE["key_binding"]} == {"PASS", "FAIL", "SKIPPED"}


@pytest.mark.parametrize("case", TABLE["key_thumbprint_vectors"], ids=lambda c: c["name"])
def test_rfc7638_thumbprint_vectors(case) -> None:
    alg, public_key = _expand(case["key"])
    assert vr.thumbprint_for_key(alg=alg, public_key=public_key) == case["thumbprint"]


@pytest.mark.parametrize("case", TABLE["key_binding"], ids=lambda c: c["name"])
def test_key_binding_axis_table(case) -> None:
    alg, public_key = _expand(case["key"])
    result, note = vr.check_key_binding(case["payload"], alg, public_key)
    assert result == case["expect"]["result"], note
    assert case["expect"]["note_contains"] in note, note


    # pub is unpadded base64url; the directory's own alphabet yields another digest.
def test_pub_member_uses_unpadded_base64url() -> None:
    jwk = vr.akp_jwk(alg="ML-DSA-87", public_key=bytes([0xFF]) * 2592)
    assert set(jwk) == {"alg", "kty", "pub"}
    assert jwk["kty"] == "AKP"
    assert "/" not in jwk["pub"] and "+" not in jwk["pub"] and "=" not in jwk["pub"]
    assert jwk["pub"].startswith("____")


    # RFC 7638 requires lexicographic members and no whitespace, whatever the caller passes.
def test_thumbprint_ignores_member_insertion_order() -> None:
    pk = bytes([0x07]) * 1952
    ordered = {"alg": ALG, "kty": "AKP", "pub": vr._b64url_nopad(pk)}
    shuffled = {"pub": vr._b64url_nopad(pk), "kty": "AKP", "alg": ALG}
    assert vr.jwk_thumbprint(ordered) == vr.jwk_thumbprint(shuffled)
    assert vr.jwk_thumbprint(ordered) == vr.thumbprint_for_key(alg=ALG, public_key=pk)


# --------------------------------------------------------------------------
# Wire tests - each fails if its own call site is deleted
# --------------------------------------------------------------------------


    # Deleting the axis from the oracle adapter's extra_axes fails here.
def test_oracle_adapter_emits_the_axis() -> None:
    result = oracle_verify(_receipt(TP_GOOD), ADAPTERS, _jwks(KEY_GOOD))
    axis = result.axis("key_binding")
    assert axis is not None, "adapter did not emit a key_binding axis"
    assert axis.result == "PASS", axis.note


    # Deleting the axis from run_structured fails here.
def test_run_structured_emits_the_axis() -> None:
    out = vr.run_structured(_receipt(TP_GOOD), _jwks(KEY_GOOD))
    axes = {a["name"]: a for a in out["axes"]}
    assert "key_binding" in axes, f"axes were {sorted(axes)}"
    assert axes["key_binding"]["result"] == "PASS", axes["key_binding"]["note"]


    # Deleting the axis from the printed CLI path (run) fails here.
def test_cli_run_prints_the_axis(capsys) -> None:
    vr.run(_receipt(TP_GOOD), _jwks(KEY_GOOD), None)
    report = capsys.readouterr().out
    assert "key_binding" in report, report
    assert "rederives from the resolved" in report, report

    code = vr.run(_receipt(TP_GOOD), _jwks(KEY_EVIL), None)
    substituted = capsys.readouterr().out
    assert "key_substituted" in substituted, substituted
    assert code != 0


    # Removing key_binding from either _INVALID_FAIL_AXES table fails here.
def test_a_substituted_key_is_terminal_invalid_not_a_warning() -> None:
    assert "key_binding" in ORACLE_INVALID_FAIL_AXES
    assert "key_binding" in vr._INVALID_FAIL_AXES
    result = oracle_verify(_receipt(TP_GOOD), ADAPTERS, _jwks(KEY_EVIL))
    axis = result.axis("key_binding")
    assert (axis.result, axis.failure_class) == ("FAIL", "invalid"), axis.note
    assert "key_substituted" in axis.note
    # The whole point: it is never reported as verified.
    assert result.verdict == "unverified"
    assert result.failure_class == "invalid"

    out = vr.run_structured(_receipt(TP_GOOD), _jwks(KEY_EVIL))
    binding = next(a for a in out["axes"] if a["name"] == "key_binding")
    assert binding["result"] == "FAIL"
    assert out["verdict"] == "unverified"
    assert out["failure_class"] == "invalid"


    # A skip blocks every axis but chain, so absence must PASS or all legacy receipts break.
def test_absence_does_not_block_a_verdict() -> None:
    binding = oracle_verify(_receipt(None), ADAPTERS, _jwks(KEY_GOOD)).axis("key_binding")
    assert binding.result == "PASS", binding.note
    assert binding.failure_class is None
    out = vr.run_structured(_receipt(None), _jwks(KEY_GOOD))
    binding = next(a for a in out["axes"] if a["name"] == "key_binding")
    assert binding["result"] == "PASS"
    assert binding["failure_class"] is None


    # Removing key_thumbprint from _UNSIGNED_CLAIM_FIELDS fails here.
def test_hash_mode_cannot_display_an_unsigned_thumbprint() -> None:
    assert "key_thumbprint" in _UNSIGNED_CLAIM_FIELDS
    doc = {
        "v": 1,
        "mode": "hash",
        "hash": "a" * 64,
        "hash_algo": "sha256",
        "metadata": {},
        "server_timestamp": "2026-08-31T10:00:00Z",
        "action_id": "a1",
        "agent_id": "ag_1",
        "org_id": "org_1",
        "policy_decision": "allow",
        "signature_b64": base64.b64encode(b"\x00" * 64).decode(),
        "key_id": "iss_1",
        "key_thumbprint": TP_GOOD,
    }
    result = oracle_verify(doc, ADAPTERS, _jwks(KEY_GOOD))
    structure = result.axis("structure")
    assert structure.result == "FAIL", structure.note
    assert "key_thumbprint" in structure.note
    assert result.verdict == "unverified"


    # Hash mode signs no thumbprint, so a clean hash-mode receipt still reads not-checked.
def test_hash_mode_without_a_thumbprint_reports_not_checked() -> None:
    doc = {
        "v": 1,
        "mode": "hash",
        "hash": "a" * 64,
        "hash_algo": "sha256",
        "metadata": {},
        "server_timestamp": "2026-08-31T10:00:00Z",
        "action_id": "a1",
        "agent_id": "ag_1",
        "org_id": "org_1",
        "policy_decision": "allow",
        "signature_b64": base64.b64encode(b"\x00" * 64).decode(),
        "key_id": "iss_1",
    }
    axis = oracle_verify(doc, ADAPTERS, _jwks(KEY_GOOD)).axis("key_binding")
    assert axis.result == "PASS"
    assert "binding not checked" in axis.note
