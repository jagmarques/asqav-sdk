"""Shared adversarial case table, Python half.

One JSON table drives both the Python and the TypeScript verifier. Each case
pins a verdict and the result of every axis it names, so a rule that drifts in
one language fails a suite instead of waiting for a reviewer to notice. The
TypeScript half lives in typescript/tests/cross-language-cases.test.ts and reads
the same file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import asqav

CASES_FILE = Path(__file__).parent.parent.parent / "verifier" / "cross-language-cases.json"
CASES = json.loads(CASES_FILE.read_text())["cases"]


def test_case_table_is_populated() -> None:
    """A table that silently empties would make every case below vacuous."""
    assert len(CASES) >= 20, f"case table has only {len(CASES)} cases"


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_cross_language_case(case: dict) -> None:
    result = asqav.verify_receipt_offline(case["receipt"], case["jwks"])
    axes = {a["name"]: a["result"] for a in result["axes"]}
    assert result["verdict"] == case["expect"]["verdict"], (
        f"{case['name']}: verdict {result['verdict']!r}, axes {axes}"
    )
    for axis, expected in case["expect"]["axes"].items():
        assert axes.get(axis) == expected, (
            f"{case['name']}: axis {axis} is {axes.get(axis)!r}, expected {expected!r}"
        )
