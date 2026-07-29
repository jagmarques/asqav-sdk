"""Anchor-binding and clock-skew axis parity, Python half.

One JSON table drives both the Python and the TypeScript axis implementations, so
a rule that drifts in one language fails a suite. The TypeScript half lives in
typescript/tests/verifier-axis-parity.test.ts and reads the same file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asqav.verifier.oracle.core import MAX_NESTING_DEPTH as ORACLE_MAX_NESTING_DEPTH
from asqav.verifier.verify_receipt import SKEW_BOUND_SECONDS, check_anchors, check_skew

CASES_FILE = Path(__file__).parent.parent.parent / "verifier" / "axis-parity-cases.json"
TABLE = json.loads(CASES_FILE.read_text())


def test_table_is_populated() -> None:
    """A table that silently empties would make every case below vacuous."""
    assert len(TABLE["anchors"]) >= 25, f"only {len(TABLE['anchors'])} anchor cases"
    assert len(TABLE["skew"]) >= 15, f"only {len(TABLE['skew'])} skew cases"


def test_bounds_match_the_table() -> None:
    """The limits are part of the contract the other language mirrors."""
    assert SKEW_BOUND_SECONDS == 300
    assert ORACLE_MAX_NESTING_DEPTH == 200


@pytest.mark.parametrize("case", TABLE["anchors"], ids=[c["name"] for c in TABLE["anchors"]])
def test_anchor_case(case: dict) -> None:
    result, note = check_anchors(case["envelope"])
    assert result == case["expect"]["result"], f"{case['name']}: note {note}"
    # The digest in the note is the JCS of the envelope minus anchors, so an exact
    # match also proves the two canonicalisers agree byte for byte.
    assert note == case["expect"]["note"], case["name"]


@pytest.mark.parametrize("case", TABLE["skew"], ids=[c["name"] for c in TABLE["skew"]])
def test_skew_case(case: dict) -> None:
    result, note = check_skew(case["issued_at"])
    assert result == case["expect"]["result"], f"{case['name']}: note {note}"
    assert case["expect"]["note_contains"] in note, case["name"]
