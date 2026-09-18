"""The numeric-domain rule holds at every object parse boundary (criterion 686).

Versioned round 2: declared v in any form gets the rule, absent keeps legacy.
"""

from __future__ import annotations

import json
from pathlib import Path

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


def test_v_missing_keeps_legacy_verdict():
    """No declared version: pre-rule behaviour, no rule note (round 2 item 1)."""
    d, jwks = _genesis_big(lambda p: p.pop("v"))
    result = asqav.verify_receipt_offline(d, jwks)
    assert result["verdict"] == "unverified"  # signature broken by the probe
    assert len(result["axes"]) == 12
    assert not _refused(result)


def test_v2_gets_the_rule():
    """An unrecognised declared version earns the rule, never less (item 7)."""
    d, jwks = _genesis_big(lambda p: p.update(v=2))
    result = asqav.verify_receipt_offline(d, jwks)
    assert result["verdict"] == "unverified"
    assert _refused(result)


def test_acta_legacy_no_version_keeps_verifying():
    """The grading acta bytes declare no v, so Python keeps verifying (item 6)."""
    d = _load(FIXTURES / "t202_acta_big_receipt.json")
    jwks = _load(FIXTURES / "t202_acta_big_jwks.json")
    assert "v" not in d["payload"]
    result = asqav.verify_receipt_offline(d, jwks)
    assert result["verdict"] == "verified"


def test_acta_v2_refused():
    """The same acta bytes with v declared are refused by the rule (item 7)."""
    d = _load(FIXTURES / "t202_acta_big_receipt.json")
    jwks = _load(FIXTURES / "t202_acta_big_jwks.json")
    d["payload"]["v"] = 2
    result = asqav.verify_receipt_offline(d, jwks)
    assert result["verdict"] == "unverified"
    assert _refused(result)
