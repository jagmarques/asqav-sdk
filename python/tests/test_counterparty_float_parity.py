"""The float table the TypeScript SDK checks equals what Python computes (criterion 744)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav._jcs import canonical_json
from asqav.counterparty import compute_envelope_hash, verify_counterparty_binding

TABLE = json.loads(
    (Path(__file__).resolve().parents[2] / "verifier" / "counterparty-float-cases.json").read_text()
)
CASES = TABLE["cases"]
UNRELATED_HASH = "BwcHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwc="


def _payload_text(case: dict) -> str:
    if "payload_text" in case:
        return case["payload_text"]
    w, leaf = case["wrappers"], case["leaf"]
    if case["nest"] == "object":
        inner = '{"n":' * w + '{"leaf":' + leaf + "}" + "}" * w
    else:
        inner = "[" * w + leaf + "]" * w
    return '{"type":"protectmcp:decision","junk":' + inner + "}"


def _envelope(case: dict) -> dict:
    return {"payload": json.loads(_payload_text(case)), "signature": TABLE["signature"]}


def _ack(envelope_hash: str) -> dict:
    binding = {
        "scope": "envelope_minus_anchors",
        "receipt_ref": "r1",
        "envelope_hash": envelope_hash,
    }
    return {"payload": {"counterparty_binding": binding}, "signature": {"kid": "kid-B"}}


def test_table_has_hashed_and_refused_rows() -> None:
    assert any(c["envelope_hash"] for c in CASES)
    assert any(c["error"] for c in CASES)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_expected_hash_is_python_output(case: dict) -> None:
    if case["envelope_hash"] is None:
        with pytest.raises(ValueError, match=case["error"]):
            compute_envelope_hash(_envelope(case))
    else:
        assert compute_envelope_hash(_envelope(case)) == case["envelope_hash"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_binding_label_follows_the_hash(case: dict) -> None:
    claimed = case["envelope_hash"] or UNRELATED_HASH
    result = verify_counterparty_binding(_ack(claimed), _envelope(case))
    assert result.label == ("matches" if case["envelope_hash"] else "unresolved")


def test_float_repr_rows_are_python_output() -> None:
    assert TABLE["float_reprs"]
    for lexeme, expected in TABLE["float_reprs"]:
        assert canonical_json({"v": json.loads(lexeme)}).decode() == expected, lexeme
