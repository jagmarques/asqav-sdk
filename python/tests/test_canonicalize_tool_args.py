"""Tests for :func:`asqav.canonicalize_tool_args`.

The helper pre-validates ``tool_args`` payloads against the
Compliance Receipts canonicalization rule: no floats in the signed
canonical scope, because canonical JSON only fixes byte-stable numeric
output for safe-range integers. Floats drift across runtimes and
break verifier equality.
"""

from __future__ import annotations

import hashlib
import math

import pytest

from asqav import canonicalize, canonicalize_tool_args


def test_happy_path_dict_with_int_str_bool_nested_dict_and_list() -> None:
    args = {
        "tool": "database.query",
        "limit": 100,
        "dry_run": True,
        "filters": {"status": "active", "tags": ["a", "b"]},
        "nullable": None,
    }
    out = canonicalize_tool_args(args)
    assert out == canonicalize(args)
    # Sort-stable JCS keeps the digest fixed across runs.
    assert hashlib.sha256(out).hexdigest() == hashlib.sha256(canonicalize(args)).hexdigest()


def test_rejects_bare_float_with_rfc_pointer() -> None:
    with pytest.raises(ValueError) as exc:
        canonicalize_tool_args({"x": 1.5})
    msg = str(exc.value)
    assert "tool_args float at /x" in msg
    assert "RFC 8785 section 3.2.2" in msg


def test_rejects_nested_float_and_reports_json_pointer() -> None:
    with pytest.raises(ValueError) as exc:
        canonicalize_tool_args({"a": {"b": [1, 2.0]}})
    msg = str(exc.value)
    assert "/a/b/1" in msg


def test_rejects_nan() -> None:
    with pytest.raises(ValueError) as exc:
        canonicalize_tool_args({"x": math.nan})
    assert "tool_args float at /x" in str(exc.value)


def test_rejects_infinity() -> None:
    with pytest.raises(ValueError) as exc:
        canonicalize_tool_args({"x": math.inf})
    assert "tool_args float at /x" in str(exc.value)


def test_accepts_decimal_as_string() -> None:
    out = canonicalize_tool_args({"x": "1.5"})
    assert out == b'{"x":"1.5"}'


def test_accepts_bool_and_does_not_misclassify_as_float() -> None:
    out = canonicalize_tool_args({"flag": True, "count": 0})
    assert out == b'{"count":0,"flag":true}'


def test_accepts_list_root() -> None:
    out = canonicalize_tool_args([{"a": 1}, {"b": 2}])
    assert out == b'[{"a":1},{"b":2}]'


def test_json_pointer_escapes_slash_and_tilde() -> None:
    with pytest.raises(ValueError) as exc:
        canonicalize_tool_args({"a/b": 1.5})
    assert "/a~1b" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        canonicalize_tool_args({"a~b": 1.5})
    assert "/a~0b" in str(exc.value)


def test_first_float_wins_for_pointer() -> None:
    with pytest.raises(ValueError) as exc:
        canonicalize_tool_args({"a": 1.5, "b": 2.5})
    msg = str(exc.value)
    # dict iteration is insertion-ordered in Python 3.7+, so /a comes first.
    assert "/a" in msg


# === Negative conformance: receipt fixture with float in payload.tool_args ===


def test_receipt_fixture_float_in_tool_args_is_rejected() -> None:
    """Receipt-shaped fixture: float under ``payload.tool_args`` must fail.

    Mirrors the wire shape a caller would feed when computing the
    canonical bytes for ``tool_args`` before signing.
    """
    receipt = {
        "payload": {
            "action_type": "tool:call",
            "tool_args": {"price": 1.5},
        }
    }
    with pytest.raises(ValueError) as exc:
        canonicalize_tool_args(receipt["payload"]["tool_args"])
    msg = str(exc.value)
    assert "/price" in msg
    assert "RFC 8785 section 3.2.2" in msg


def test_receipt_fixture_stringified_float_canonicalizes_cleanly() -> None:
    """Same fixture with the float stringified produces stable bytes."""
    receipt = {
        "payload": {
            "action_type": "tool:call",
            "tool_args": {"price": "1.5"},
        }
    }
    out = canonicalize_tool_args(receipt["payload"]["tool_args"])
    assert out == b'{"price":"1.5"}'
    assert hashlib.sha256(out).hexdigest() == hashlib.sha256(b'{"price":"1.5"}').hexdigest()
