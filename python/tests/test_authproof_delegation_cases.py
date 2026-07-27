"""Shared authproof delegationId format table, Python half.

One JSON table drives both the Python and the TypeScript authproof adapter. Each
case feeds a delegationId through this language's format-detection validator and
pins the boolean, so a regex that drifts between languages (a Unicode ``\\d``, a
``$`` that swallows a trailing newline) fails a suite instead of waiting for a
reviewer. The TypeScript half lives in
typescript/tests/authproof-delegation-cases.test.ts and reads the same file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asqav.verifier.oracle.adapters.authproof import AuthproofAdapter

CASES_FILE = Path(__file__).parent.parent.parent / "verifier" / "authproof-delegation-cases.json"
CASES = json.loads(CASES_FILE.read_text())["cases"]

_ADAPTER = AuthproofAdapter()


def _skeleton(delegation_id: str) -> dict:
    """A receipt that satisfies every detect() condition except the delegationId.

    Only the delegationId varies across cases, so the detect() boolean isolates
    the delegationId regex - the thing the cross-language table pins.
    """
    return {
        "delegationId": delegation_id,
        "issuedAt": 1719000000000,
        "scope": "session",
        "boundaries": {"maxActions": 1},
        "timeWindow": {"start": 0, "end": 1},
        "operatorInstructions": "carry out the agreed task",
        "instructionsHash": "sha256:" + "a" * 64,
        "signerPublicKey": {"kty": "EC", "crv": "P-256", "x": "AA", "y": "AA"},
        "signature": "ab" * 64,
    }


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_delegation_id_agreement(case: dict) -> None:
    assert _ADAPTER.detect(_skeleton(case["value"])) == case["expected"], (
        f"{case['name']}: {case['value']!r} detected as "
        f"{_ADAPTER.detect(_skeleton(case['value']))}, expected {case['expected']}"
    )
