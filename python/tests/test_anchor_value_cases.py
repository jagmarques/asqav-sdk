"""Anchor value base64 tolerance, Python half.

One corpus drives both halves. The anchors field is unsigned and attacker
steerable, so the TypeScript half additionally asserts that no disagreement runs
in the permissive direction. The TypeScript half lives in
typescript/tests/verifier-anchor-value-parity.test.ts and reads the same file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from asqav.verifier.verify_receipt import _safe_b64, check_anchors

CASES_FILE = Path(__file__).parent.parent.parent / "verifier" / "anchor-value-cases.json"
VALUES = json.loads(CASES_FILE.read_text())["values"]

ENVELOPE = {
    "payload": {"type": "protectmcp:decision", "issued_at": "2026-06-19T00:00:00+00:00"},
    "signature": {"alg": "ML-DSA-65", "kid": "k1", "sig": "AAAA"},
}


    # A corpus that silently empties would make every case below vacuous.
def test_corpus_is_populated() -> None:
    assert len(VALUES) >= 900, f"corpus has only {len(VALUES)} values"
    wide = [c for c in VALUES if any(ord(ch) > 127 for ch in c["value"])]
    assert len(wide) >= 350, f"only {len(wide)} values carry a non-ASCII codepoint"
    # A shape-valid value reports SKIPPED (unverifiable), not PASS. Both directions must be
    # present or the corpus cannot tell a working shape gate from a stuck one.
    failing = [c for c in VALUES if c["expect"]["axis"] == "FAIL"]
    assert len(failing) >= 100, f"only {len(failing)} values are expected to fail"
    skipped = [c for c in VALUES if c["expect"]["axis"] == "SKIPPED"]
    assert len(skipped) >= 100, f"only {len(skipped)} values are expected to skip"


def _surplus_padding(value: str) -> bool:
    """Alphabet characters then more padding than base64 can carry.

    base64.b64decode(validate=True) accepts this shape on 3.11 and raises on
    3.12, so a corpus without it cannot tell a version-stable rule from a
    delegated one.
    """
    s = value.replace("-", "+").replace("_", "/")
    s += "=" * ((-len(s)) % 4)
    return bool(re.fullmatch(r"[A-Za-z0-9+/]+={3,}", s))


def test_corpus_covers_the_excess_padding_class() -> None:
    excess = [c for c in VALUES if _surplus_padding(c["value"])]
    assert len(excess) >= 20, f"only {len(excess)} surplus-padding values"
    passing = [c["value"] for c in excess if c["expect"]["axis"] != "FAIL"]
    assert passing == [], f"surplus padding must not pass: {passing[:4]}"


@pytest.mark.parametrize("case", VALUES, ids=[repr(c["value"]) for c in VALUES])
def test_anchor_value(case: dict) -> None:
    env = dict(ENVELOPE)
    env["anchors"] = [{"type": "rfc3161", "value": case["value"]}]
    assert _safe_b64(case["value"]) is case["expect"]["safe_b64"], repr(case["value"])
    assert check_anchors(env)[0] == case["expect"]["axis"], repr(case["value"])
