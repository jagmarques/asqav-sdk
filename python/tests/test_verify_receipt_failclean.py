"""Report tool fails clean on missing / empty / non-object input.

Regression guard for the goal9 SDK example fix: verify_receipt must never leak
a urllib/json traceback on bad input. It exits nonzero with one readable line,
and reads a receipt from stdin via ``--receipt -``.
"""

from __future__ import annotations

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.verifier import verify_receipt as vr


def test_run_rejects_non_object_envelope() -> None:
    # A JSON array (non-object) must not crash run(); it returns 2 (INCOMPLETE).
    assert vr.run([], {"keys": []}, None) == 2


def test_run_structured_rejects_non_object() -> None:
    out = vr.run_structured("not-an-object", {"keys": []})
    assert out["verdict"] == "INCOMPLETE"
    assert out["canonical_sha256"] is None


def test_parse_object_rejects_array() -> None:
    with pytest.raises(vr.VerifierInputError):
        vr._parse_object("[]", "stdin")


def test_parse_object_rejects_empty() -> None:
    with pytest.raises(vr.VerifierInputError):
        vr._parse_object("   ", "stdin")


def test_parse_object_rejects_bad_json() -> None:
    with pytest.raises(vr.VerifierInputError):
        vr._parse_object("{not json", "stdin")


def test_parse_object_accepts_object() -> None:
    assert vr._parse_object('{"a": 1}', "stdin") == {"a": 1}


def test_load_dash_reads_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"a": 1}'))
    assert vr._load("-") == {"a": 1}


def test_load_dash_non_object_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("[]"))
    with pytest.raises(vr.VerifierInputError):
        vr._load("-")


def test_load_missing_file_raises_input_error() -> None:
    # A missing file surfaces as VerifierInputError, not a raw OSError traceback.
    with pytest.raises(vr.VerifierInputError):
        vr._load("/no/such/receipt/file.json")
