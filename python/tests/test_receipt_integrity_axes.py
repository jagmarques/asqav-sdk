"""Receipt-internal integrity axes, Python half.

Two classes of claim an issuer could previously assert unchecked:

  - `payload_digest` over a `context` the receipt carries itself. Recomputable
    with NO external data, so a receipt whose own context does not hash to its
    own digest is provably inconsistent. Nothing checked it before.
  - `counterparty_binding`, which asserts a counterparty acknowledged the action.
    The hosted verifier resolves it against its database; an offline third party
    has none, so a fabricated binding reached a plain verified verdict.

The shared table in verifier/axis-parity-cases.json drives this file and
typescript/tests/verifier-receipt-integrity-parity.test.ts.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from asqav.verifier import verify_receipt as vr
from asqav.verifier.oracle import ADAPTERS
from asqav.verifier.oracle import verify as oracle_verify
from asqav.verifier.oracle.core import _INVALID_FAIL_AXES as ORACLE_INVALID_FAIL_AXES

CASES_FILE = Path(__file__).parent.parent.parent / "verifier" / "axis-parity-cases.json"
TABLE = json.loads(CASES_FILE.read_text())

ALG = "ML-DSA-65"
KEY = bytes([0x00]) * 1952
CTX = {"amount": 100, "currency": "EUR"}
HONEST = hashlib.sha256(vr.canonical_json(CTX)).hexdigest()


def _jwks() -> dict:
    return {
        "keys": [
            {
                "kid": "iss_1",
                "issuer_id": "iss_1",
                "agent_id": "ag_1",
                "org_id": "org_1",
                "alg": ALG,
                "public_key": base64.b64encode(KEY).decode(),
                "status": "active",
            }
        ]
    }


def _receipt(**extra) -> dict:
    payload = {
        "v": 1,
        "type": "protectmcp:decision",
        "issuer_id": "iss_1",
        "agent_id": "ag_1",
        "org_id": "org_1",
        "previousReceiptHash": vr.FIRST_RECEIPT_SEED,
        "issued_at": "2026-08-31T10:00:00Z",
        "action_ref": "a" * 64,
        "context": CTX,
        "payload_digest": {"hash": HONEST, "size": len(vr.canonical_json(CTX))},
        "policy_digest": "pd",
        "decision": "allow",
    }
    payload.update(extra)
    return {
        "payload": payload,
        "signature": {"alg": ALG, "kid": "iss_1", "sig": base64.b64encode(b"\x00" * 64).decode()},
        "anchors": {},
    }


def test_table_is_populated() -> None:
    assert len(TABLE["payload_digest"]) >= 10
    assert len(TABLE["counterparty_binding"]) >= 8
    assert {c["expect"]["result"] for c in TABLE["payload_digest"]} == {"PASS", "FAIL"}
    assert {c["expect"]["result"] for c in TABLE["counterparty_binding"]} == {
        "PASS",
        "FAIL",
        "SKIPPED",
    }


def test_payload_digest_table() -> None:
    for case in TABLE["payload_digest"]:
        result, note = vr.check_payload_digest(case["payload"])
        assert result == case["expect"]["result"], f"{case['name']}: {note}"
        assert case["expect"]["note_contains"] in note, f"{case['name']}: {note}"


def test_counterparty_binding_table() -> None:
    for case in TABLE["counterparty_binding"]:
        result, note = vr.check_counterparty_binding(case["payload"])
        assert result == case["expect"]["result"], f"{case['name']}: {note}"
        assert case["expect"]["note_contains"] in note, f"{case['name']}: {note}"


    # The originator path the table cannot carry: a real recompute, both directions.
def test_counterparty_binding_resolves_against_a_supplied_originator() -> None:
    originator = _receipt()
    good = base64.b64encode(hashlib.sha256(vr.canonical_json(originator)).digest()).decode()

    payload = _receipt(
        counterparty_binding={"receipt_ref": "sig_orig", "envelope_hash": good}
    )["payload"]
    assert vr.check_counterparty_binding(payload, originator)[0] == "PASS"

    wrong = base64.b64encode(b"\x00" * 32).decode()
    payload = _receipt(
        counterparty_binding={"receipt_ref": "sig_orig", "envelope_hash": wrong}
    )["payload"]
    result, note = vr.check_counterparty_binding(payload, originator)
    assert result == "FAIL"
    assert "counterparty_mismatch" in note

    payload = _receipt(
        counterparty_binding={
            "receipt_ref": "sig_orig",
            "envelope_hash": good,
            "expect_ack_from": "somebody_else",
        }
    )["payload"]
    result, note = vr.check_counterparty_binding(payload, originator)
    assert result == "FAIL"
    assert "expects an acknowledgment from somebody_else" in note


# --------------------------------------------------------------------------
# Wire tests - each fails if its own call site is deleted


def test_oracle_emits_all_three_axes() -> None:
    """The oracle lacked skew entirely, so a postdated receipt passed there."""
    axes = {a.axis for a in oracle_verify(_receipt(), ADAPTERS, _jwks()).axes}
    assert {"payload_digest", "counterparty", "skew"} <= axes, sorted(axes)


def test_run_structured_emits_the_new_axes() -> None:
    names = {a["name"] for a in vr.run_structured(_receipt(), _jwks())["axes"]}
    assert {"payload_digest", "counterparty"} <= names, sorted(names)


def test_a_lying_payload_digest_is_terminal_invalid() -> None:
    assert "payload_digest" in ORACLE_INVALID_FAIL_AXES
    assert "payload_digest" in vr._INVALID_FAIL_AXES
    result = oracle_verify(
        _receipt(payload_digest={"hash": "f" * 64, "size": 31}), ADAPTERS, _jwks()
    )
    axis = result.axis("payload_digest")
    assert (axis.result, axis.failure_class) == ("FAIL", "invalid"), axis.note
    assert "payload_digest_mismatch" in axis.note
    assert result.verdict == "unverified"


def test_a_fabricated_counterparty_binding_cannot_read_as_corroborated() -> None:
    assert "counterparty" in ORACLE_INVALID_FAIL_AXES
    assert "counterparty" in vr._INVALID_FAIL_AXES
    forged = {
        "receipt_ref": "sig_NEVER_EXISTED",
        "envelope_hash": base64.b64encode(b"\x00" * 32).decode(),
    }
    axis = oracle_verify(_receipt(counterparty_binding=forged), ADAPTERS, _jwks()).axis(
        "counterparty"
    )
    # SKIPPED blocks the verdict, so the claim never rides along as corroboration
    assert axis.result == "SKIPPED", axis.note
    assert axis.failure_class == "unverifiable"


def test_the_oracle_refuses_a_postdated_receipt() -> None:
    axis = oracle_verify(_receipt(issued_at="2099-01-01T00:00:00Z"), ADAPTERS, _jwks()).axis(
        "skew"
    )
    assert axis.result == "FAIL", axis.note
    assert axis.failure_class == "invalid"


    # Absence must PASS: a skip blocks every axis but chain, so a receipt that
    # claims neither would otherwise regress to unverified.
def test_absence_of_both_claims_does_not_block() -> None:
    doc = _receipt()
    doc["payload"].pop("counterparty_binding", None)
    doc["payload"].pop("payload_digest", None)
    result = oracle_verify(doc, ADAPTERS, _jwks())
    for name in ("payload_digest", "counterparty"):
        axis = result.axis(name)
        assert axis.result == "PASS", f"{name}: {axis.note}"
        assert axis.failure_class is None
