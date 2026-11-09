"""xsd:datetime grammar pinned on the W3C VC adapter (criterion 336).

_parse_datetime gates proof.created/validFrom/validUntil on the shape the
TypeScript twin pins, so out-of-grammar stamps read unreadable on every
engine.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from asqav.verifier.oracle.adapters import w3c_vc
from asqav.verifier.oracle.adapters.w3c_vc import (
    _XSD_DATETIME_RE,
    W3cVcAdapter,
    _parse_datetime,
)

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"
BASIC = "20200101T000000Z"


def _open(narrow: str, wide: str) -> re.Pattern[str]:
    """The grammar with one range clause widened to any two digits."""
    assert narrow in _XSD_DATETIME_RE.pattern
    return re.compile(_XSD_DATETIME_RE.pattern.replace(narrow, wide), re.ASCII)


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


def test_grammar_rejects_hour_24_on_every_interpreter() -> None:
    """The regex bars hour 24, so no fromisoformat build rolls it over.
    NEGCTL: the same regex with a two-digit hour takes every hour-24 stamp."""
    stamps = ("2020-06-01T24:00:00Z", "2020-06-01T24:30:00+02:00", "2020-06-01T24:00:00.5z")
    open_hour = _open(r"T(?:[01]\d|2[0-3]):", r"T\d{2}:")
    for stamp in stamps:
        assert _XSD_DATETIME_RE.match(stamp) is None
        assert open_hour.match(stamp) is not None
    assert _XSD_DATETIME_RE.match("2020-06-01T23:59:59Z") is not None


def test_grammar_bars_minute_second_and_offset_overflow() -> None:
    """The regex owns every time range, not the interpreter's fromisoformat.
    NEGCTL: each clause widened to two digits takes its overflow stamp."""
    cases = (
        ("2020-06-01T00:60:00Z", r":[0-5]\d(?::", r":\d{2}(?::"),
        ("2020-06-01T00:00:60Z", r"(?::[0-5]\d(?:\.", r"(?::\d{2}(?:\."),
        ("2020-06-01T00:00:00+24:00", r"[+-](?:[01]\d|2[0-3]):", r"[+-]\d{2}:"),
        ("2020-06-01T00:00:00+00:60", r"2[0-3]):[0-5]\d)$", r"2[0-3]):\d{2})$"),
    )
    for stamp, narrow, wide in cases:
        assert _XSD_DATETIME_RE.match(stamp) is None
        assert _open(narrow, wide).match(stamp) is not None
        assert _parse_datetime(stamp) is None


class _Py310Datetime(datetime):
    """fromisoformat as before 3.11: fractions of three or six digits only."""

    @classmethod
    def fromisoformat(cls, text: str) -> datetime:
        fraction = re.search(r"\.(\d+)", text)
        if fraction and len(fraction.group(1)) not in (3, 6):
            raise ValueError(text)
        return datetime.fromisoformat(text)


def test_fraction_length_does_not_depend_on_interpreter(monkeypatch) -> None:
    """Any fraction length reads the same where fromisoformat is strict.
    NEGCTL: the strict stand-in itself rejects a one-digit fraction."""
    with pytest.raises(ValueError):
        _Py310Datetime.fromisoformat("2020-06-01T00:00:00.5+00:00")
    monkeypatch.setattr(w3c_vc, "datetime", _Py310Datetime)
    for fraction in ("5", "12", "1234567"):
        assert _parse_datetime(f"2020-06-01T00:00:00.{fraction}Z") is not None
    assert _parse_datetime("2020-06-01T00:00:00.5Z").microsecond == 500000


def test_non_ascii_digits_are_unreadable() -> None:
    """NEGCTL: the ASCII flag off takes the Arabic-Indic year."""
    stamp = "\u0662020-06-01T00:00:00Z"
    assert _XSD_DATETIME_RE.match(stamp) is None
    assert re.compile(_XSD_DATETIME_RE.pattern).match(stamp) is not None


def test_extended_stamps_still_parse() -> None:
    """The pin narrows the language, never the RFC 3339 outcomes."""
    stamp = _parse_datetime("2020-06-01T00:00:00Z")
    assert stamp is not None and stamp.tzinfo is not None
    assert _parse_datetime("2020-06-01T00:00:00+02:00") is not None
    assert _parse_datetime("2020-06-01T00:00:00z") is not None
    doc = _doc()
    assert W3cVcAdapter().schema(doc)[0] == "PASS"
