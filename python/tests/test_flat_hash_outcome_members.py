"""Outcome members a cloud flat hash-mode receipt signs when set."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from asqav.verifier.verify_receipt import FAILURE_INVALID, VERDICT_UNVERIFIED, VERDICT_VERIFIED

_SRC = Path(__file__).resolve().parents[1] / "src"
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "flat_hash_outcome_members.json"
_CHILD = """
import json, sys
from asqav import verify_receipt_offline
from asqav.verifier import verify_receipt as vr
from asqav.verifier.oracle.adapters.asqav_native import _flat_signed_fields
data = json.load(sys.stdin)
receipt = data["receipt"]
print(json.dumps({
    "result": verify_receipt_offline(receipt, data["jwks"]),
    "fields": list(_flat_signed_fields(receipt)),
    "signed": list(vr._FLAT_SIGNED),
    "note": vr.check_outer_copies(receipt, b"")[1],
}))
"""
_OPTIONAL = {
    "plain": (),
    "result_digest": ("result_digest",),
    "action_ref": ("action_ref",),
    "invocation_ref": ("invocation_ref",),
    "all_three": ("result_digest", "action_ref", "invocation_ref"),
}


def _run(name: str, tamper=None) -> dict:
    """Run one fixture receipt through this tree's verifier, out of process,
    so an installed asqav never shadows the source under test."""
    data = json.loads(_FIXTURE.read_text())
    receipt = data["receipts"][name]
    if tamper is not None:
        tamper(receipt)
    done = subprocess.run(
        [sys.executable, "-c", _CHILD],
        input=json.dumps({"receipt": receipt, "jwks": data["jwks"]}),
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(_SRC)}, check=True,
    )
    return json.loads(done.stdout)


@pytest.mark.parametrize("name", list(_OPTIONAL))
def test_cloud_flat_receipt_verifies(name):
    result = _run(name)["result"]
    assert result["verdict"] == VERDICT_VERIFIED, result["axes"]


@pytest.mark.parametrize("name", list(_OPTIONAL))
def test_rebuilt_members_are_the_twelve_plus_each_set_outcome(name):
    out = _run(name)
    assert tuple(out["fields"]) == tuple(out["signed"]) + _OPTIONAL[name]


@pytest.mark.parametrize("member", ["result_digest", "action_ref", "invocation_ref"])
def test_tampered_optional_member_fails(member):
    """NEGCTL: a signed outcome member changed after signing is invalid."""
    def tamper(receipt: dict) -> None:
        receipt[member] = "f" * 64

    assert _run("all_three")["result"]["verdict"] == VERDICT_VERIFIED
    result = _run("all_three", tamper)["result"]
    assert result["verdict"] == VERDICT_UNVERIFIED, result["axes"]
    assert result["failure_class"] == FAILURE_INVALID


def test_optional_member_added_after_signing_fails():
    """NEGCTL: an outcome member the signature never covered is invalid."""
    def tamper(receipt: dict) -> None:
        receipt["result_digest"] = "sha256:" + "d" * 64

    assert _run("plain")["result"]["verdict"] == VERDICT_VERIFIED
    result = _run("plain", tamper)["result"]
    assert result["verdict"] == VERDICT_UNVERIFIED, result["axes"]
    assert result["failure_class"] == FAILURE_INVALID


def test_optional_members_are_not_listed_as_unsigned():
    note = _run("all_three")["note"]
    assert "result_digest" not in note and "invocation_ref" not in note
