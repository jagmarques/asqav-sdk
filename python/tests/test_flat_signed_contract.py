"""The flat hash-mode signed members match contracts/flat-signed.json."""
from __future__ import annotations

import json
from pathlib import Path

from asqav.verifier import verify_receipt as vr
from asqav.verifier.oracle.adapters import asqav_native

_CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "flat-signed.json"


def _contract() -> dict:
    return json.loads(_CONTRACT.read_text())


def _drift(contract: dict, required: tuple, optional: tuple) -> list:
    """Names the contract and the verifier disagree on, in either direction."""
    return sorted(set(contract["required"]) ^ set(required)) + sorted(
        set(contract["optional"]) ^ set(optional)
    )


def test_python_verifier_members_match_contract():
    contract = _contract()
    assert tuple(contract["required"]) == vr._FLAT_SIGNED
    assert tuple(contract["optional"]) == vr._FLAT_SIGNED_OPTIONAL
    assert _drift(contract, vr._FLAT_SIGNED, vr._FLAT_SIGNED_OPTIONAL) == []


def test_native_adapter_rebuilds_the_contract_members():
    contract = _contract()
    names = contract["required"] + contract["optional"]
    doc = {name: "x" for name in names}
    assert asqav_native._vr is vr
    assert list(asqav_native._flat_signed_fields(doc)) == names


def test_negctl_dropped_contract_member_is_reported():
    contract = _contract()
    for key in ("required", "optional"):
        shrunk = {**contract, key: contract[key][1:]}
        assert _drift(shrunk, vr._FLAT_SIGNED, vr._FLAT_SIGNED_OPTIONAL) == [contract[key][0]]
