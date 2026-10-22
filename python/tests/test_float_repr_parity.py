"""The float table the TypeScript verifier checks equals what Python computes (criterion 747).
Python writes a float with repr: 1e-07 not 1e-7, 7.635507284768651e+16 not 76355072847686510.0,
-0.0 kept. verifier/float-repr-cases.json pins that for seeded random floats in the split zones
(1e-9 to 1e-4, 1e16 to 1e21, signed zero) and general doubles; the TypeScript half is
typescript/tests/verifier-float-repr.test.ts and reads the same file."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.counterparty import verify_counterparty_binding
from asqav.verifier.oracle import canonical as oracle
from asqav.verifier.verify_receipt import _envelope_hash, canonical_json

VERIFIER_DIR = Path(__file__).resolve().parents[2] / "verifier"
TABLE = json.loads((VERIFIER_DIR / "float-repr-cases.json").read_text())
CASES = TABLE["cases"]


def _envelope(lexeme: str) -> dict:
    return {"payload": json.loads('{"v":' + lexeme + "}"), "signature": TABLE["signature"]}


def test_table_covers_the_split_zones() -> None:
    names = ("small", "large", "mid", "bits", "edge")
    zones = {z: [c for c in CASES if c["zone"] == z] for z in names}
    assert all(len(rows) >= 40 for rows in zones.values()), {z: len(r) for z, r in zones.items()}
    assert all("e-0" in c["repr"] for c in zones["small"])
    assert all("e+" in c["repr"] for c in zones["large"])
    assert any(c["repr"] == "-0.0" for c in zones["edge"])


@pytest.mark.parametrize("case", CASES, ids=[f"{c['zone']}-{c['lexeme']}" for c in CASES])
def test_expected_values_are_python_output(case: dict) -> None:
    value = json.loads('{"v":' + case["lexeme"] + "}")
    assert repr(value["v"]) == case["repr"]
    expected = ('{"v":' + case["repr"] + "}").encode()
    assert canonical_json(value) == expected == case["canonical"].encode()
    # Both oracle dialects and the envelope hash agree on the same bytes
    assert oracle.asqav_jcs(value) == expected
    assert oracle.jcs(value) == expected
    assert _envelope_hash(_envelope(case["lexeme"])) == case["envelope_hash"]


@pytest.mark.parametrize("case", CASES, ids=[f"{c['zone']}-{c['lexeme']}" for c in CASES])
def test_binding_label_is_the_sdk_verdict(case: dict) -> None:
    binding = {
        "scope": "envelope_minus_anchors",
        "receipt_ref": "r1",
        "envelope_hash": case["envelope_hash"],
    }
    ack = {"payload": {"counterparty_binding": binding}, "signature": {"kid": "kid-B"}}
    result = verify_counterparty_binding(ack, _envelope(case["lexeme"]))
    assert result.label == case["binding"]
    assert case["binding"] == ("unresolved" if _out_of_profile(case["lexeme"]) else "matches")


def _out_of_profile(lexeme: str) -> bool:
    value = json.loads(lexeme)
    return value.is_integer() and abs(value) > 2**53 - 1


def test_table_is_the_generator_output() -> None:
    path = VERIFIER_DIR / "generate_float_repr_cases.py"
    spec = importlib.util.spec_from_file_location("gen_float_repr", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert json.loads(json.dumps(gen.build())) == TABLE
