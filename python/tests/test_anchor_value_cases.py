"""Anchor value base64 tolerance, Python half.

One corpus drives both halves. The anchors field is unsigned and attacker
steerable, so the TypeScript half additionally asserts that no disagreement runs
in the permissive direction. The TypeScript half lives in
typescript/tests/verifier-anchor-value-parity.test.ts and reads the same file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asqav.verifier.verify_receipt import _safe_b64, check_anchors

CASES_FILE = Path(__file__).parent.parent.parent / "verifier" / "anchor-value-cases.json"
VALUES = json.loads(CASES_FILE.read_text())["values"]

ENVELOPE = {
    "payload": {"type": "protectmcp:decision", "issued_at": "2026-06-19T00:00:00+00:00"},
    "signature": {"alg": "ML-DSA-65", "kid": "k1", "sig": "AAAA"},
}


def test_corpus_is_populated() -> None:
    """A corpus that silently empties would make every case below vacuous."""
    assert len(VALUES) >= 150, f"corpus has only {len(VALUES)} values"
    wide = [c for c in VALUES if any(ord(ch) > 127 for ch in c["value"])]
    assert len(wide) >= 40, f"only {len(wide)} values carry a non-ASCII codepoint"


@pytest.mark.parametrize("case", VALUES, ids=[repr(c["value"]) for c in VALUES])
def test_anchor_value(case: dict) -> None:
    env = dict(ENVELOPE)
    env["anchors"] = [{"type": "rfc3161", "value": case["value"]}]
    assert _safe_b64(case["value"]) is case["expect"]["safe_b64"], repr(case["value"])
    assert check_anchors(env)[0] == case["expect"]["axis"], repr(case["value"])
