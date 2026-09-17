"""The numeric-domain rule holds at every object parse boundary (criterion 686).

No field inside the receipt can switch the rule off: v missing, v unknown, and
every adapter all apply it. The cross-language case pins the acta split verdict
the rule exists to prevent (identical bytes, one verdict, both SDKs).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import asqav

VECTORS = Path(__file__).parent.parent.parent / "verifier" / "conformance-vectors"
GENESIS = VECTORS / "asqav-01-genesis-permit"
FIXTURES = Path(__file__).parent / "fixtures"
BIG = 9007199254740993


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _genesis_big(mutate):
    d = _load(GENESIS / "receipt.json")
    d["payload"]["probe"] = BIG
    mutate(d["payload"])
    return d, _load(GENESIS / "jwks.json")


def _refused(result: dict) -> bool:
    return any(
        a.get("result") == "FAIL" and "profile range" in (a.get("note") or "")
        for a in result.get("axes", [])
    )


def test_control_pristine_verifies():
    """Unmutated genesis permit verifies, so the harness path is live."""
    d = _load(GENESIS / "receipt.json")
    result = asqav.verify_receipt_offline(d, _load(GENESIS / "jwks.json"))
    assert result["verdict"] == "verified"


def test_control_gate_on_v1_refuses():
    """v=1 plus an out-of-domain integer is refused by the rule today."""
    d, jwks = _genesis_big(lambda p: None)
    assert d["payload"]["v"] == 1
    assert _refused(asqav.verify_receipt_offline(d, jwks))


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda p: p.pop("v"), id="v-missing"),
    pytest.param(lambda p: p.update(v=2), id="v-2"),
])
def test_numeric_rule_applies_regardless_of_profile_version(mutate):
    """v missing or unknown gets the rule, never a free pass."""
    d, jwks = _genesis_big(mutate)
    result = asqav.verify_receipt_offline(d, jwks)
    assert result["verdict"] == "unverified"
    assert _refused(result)


def test_acta_big_integer_unverified_same_as_typescript():
    """The acta split-verdict bytes read unverified here, matching TS."""
    d = _load(FIXTURES / "t202_acta_big_receipt.json")
    jwks = _load(FIXTURES / "t202_acta_big_jwks.json")
    result = asqav.verify_receipt_offline(d, jwks)
    assert result["verdict"] == "unverified"
    assert _refused(result)
