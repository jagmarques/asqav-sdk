"""Independent normalization expectations (488 C6): JSON-table oracle check.
Asserts the named axes plus the resolved-context echo (table in verifier/)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify

_TABLE = Path(__file__).resolve().parents[2] / "verifier" / "normalization-expectations.json"


def _load():
    table = json.loads(_TABLE.read_bytes())
    assert len(table["cases"]) >= 8, "expectation table must stay populated"
    base = table["base_vector"]
    vec = json.loads(
        (_TABLE.parent / "conformance-vectors" / base / "receipt.json").read_bytes()
    )
    return table["cases"], vec


def test_normalization_expectations_hold():
    cases, vec = _load()
    for case in cases:
        doc = copy.deepcopy(vec)
        doc["payload"].update(case["patch"]["payload"])
        for key in case["patch"].get("remove", []):
            doc["payload"].pop(key, None)
        ctx = VerificationContext(**case["context"])
        res = verify(doc, ADAPTERS, context=ctx)
        axes = {a.axis: a.result for a in res.axes}
        for axis, expected in case["expect"]["axes"].items():
            assert axes.get(axis) == expected, (
                f"{case['name']}: axis {axis} = {axes.get(axis)!r}, "
                f"expected {expected!r} (all axes {axes})"
            )
        assert res.context is not None, f"{case['name']}: no resolved context"
        for field, expected in case["expect"]["context_echo"].items():
            assert getattr(res.context, field) == expected, (
                f"{case['name']}: context.{field} = "
                f"{getattr(res.context, field)!r}, expected {expected!r}"
            )
