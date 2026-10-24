"""xsd:datetime grammar pinned on the W3C VC adapter (criterion 336).

_parse_datetime gates proof.created/validFrom/validUntil on the shape the
TypeScript twin pins, so out-of-grammar stamps read unreadable on every
engine.
"""

from __future__ import annotations

import json
from pathlib import Path

from asqav.verifier.oracle.adapters.w3c_vc import W3cVcAdapter, _parse_datetime

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"
BASIC = "20200101T000000Z"


def _doc() -> dict:
    path = _CORPUS / "w3c-vc-01-didweb-happy-path" / "receipt.json"
    return json.loads(path.read_text())


def test_basic_proof_created_fails_schema() -> None:
    """Basic proof.created FAILs schema as unreadable on every interpreter."""
    doc = _doc()
    doc["proof"]["created"] = BASIC
    result, note = W3cVcAdapter().schema(doc)
    assert result == "FAIL"
    assert "unreadable proof.created" in note


def test_basic_validity_window_fails_expiry_axis() -> None:
    """Basic validFrom/validUntil FAIL the expiry axis as unreadable."""
    doc = _doc()
    doc["validFrom"] = BASIC
    axes = W3cVcAdapter().extra_axes(doc, None)
    assert axes[0][1] == "FAIL"
    assert "unreadable expires_at" in axes[0][2]
    doc = _doc()
    doc["validUntil"] = BASIC
    axes = W3cVcAdapter().extra_axes(doc, None)
    assert axes[0][1] == "FAIL"
    assert "unreadable expires_at" in axes[0][2]


def test_out_of_range_seconds_fail() -> None:
    """A leap second fromisoformat rejects is unreadable, not rolled over."""
    assert _parse_datetime("2020-06-01T00:00:60Z") is None
    assert _parse_datetime("2020-06-01T24:00:00Z") is None
    assert _parse_datetime("2020-W01-1") is None
    assert _parse_datetime("June 1, 2020") is None


def test_extended_stamps_still_parse() -> None:
    """The pin narrows the language, never the RFC 3339 outcomes."""
    stamp = _parse_datetime("2020-06-01T00:00:00Z")
    assert stamp is not None and stamp.tzinfo is not None
    assert _parse_datetime("2020-06-01T00:00:00+02:00") is not None
    assert _parse_datetime("2020-06-01T00:00:00z") is not None
    doc = _doc()
    assert W3cVcAdapter().schema(doc)[0] == "PASS"
