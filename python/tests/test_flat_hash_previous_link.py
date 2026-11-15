"""Previous-link coverage for flat hash-mode receipts (criterion 926)."""
from __future__ import annotations

from functools import partial
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tests.conftest import run_flat_receipt

from asqav.verifier.verify_receipt import (
    FAILURE_INVALID,
    VERDICT_UNVERIFIED,
    VERDICT_VERIFIED,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "flat_hash_prev_link.json"
_verdict = partial(run_flat_receipt, _FIXTURE)


def _link_name(out: dict) -> str:
    """The one field a cloud body adds to the 11-field set."""
    (link,) = set(out["signed"]) - set(out["fields"])
    return link


def test_cloud_hash_receipt_with_previous_link_verifies():
    """ACCEPT: linked and null-link cloud bodies verify."""
    for name in ("linked", "null_link"):
        result = _verdict(name)["result"]
        assert result["verdict"] == VERDICT_VERIFIED, f"{name}: {result['axes']}"


def test_legacy_hash_receipt_without_link_verifies():
    out = _verdict("legacy")
    assert out["result"]["verdict"] == VERDICT_VERIFIED, out["result"]["axes"]
    assert _link_name(out) not in out["fields"]


def test_oracle_flat_fields_match_verifier():
    """The oracle rebuilds exactly the field tuple the verifier exports."""
    linked = _verdict("linked")
    link = _link_name(_verdict("legacy"))
    assert link in linked["fields"]
    assert linked["fields"] == linked["signed"]


def test_tampered_previous_receipt_hash_fails():
    """NEGCTL: a linked body with its link changed to 'f'*64 is invalid."""
    link = _link_name(_verdict("legacy"))

    def tamper(receipt: dict) -> None:
        receipt[link] = "f" * 64

    result = _verdict("linked", tamper)["result"]
    assert result["verdict"] == VERDICT_UNVERIFIED, result["axes"]
    assert result["failure_class"] == FAILURE_INVALID


def test_unsigned_link_on_legacy_receipt_fails():
    """NEGCTL: a link the 11-field signature never covered is invalid."""
    link = _link_name(_verdict("legacy"))

    def tamper(receipt: dict) -> None:
        receipt[link] = "f" * 64

    result = _verdict("legacy", tamper)["result"]
    assert result["verdict"] == VERDICT_UNVERIFIED, result["axes"]
    assert result["failure_class"] == FAILURE_INVALID
