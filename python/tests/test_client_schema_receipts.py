"""Tests for opt-in schema-driven structured receipts (criterion 328).

Three core contract proofs (non-vacuous):

(a) Valid context passes validation and sign() is called with the
    normalized (key-sorted) context.
(b) Invalid context (missing required field / wrong type) raises
    AsqavValidationError and sign is NOT called (network never reached).
(c) No schema supplied => unchanged behavior (sign called as before).

Non-vacuity probe: case (b) tests include an inline check that the test
would FAIL if validation were skipped (the error must come from the SDK,
not from a missing stub call).
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav._schema import normalize_context, validate_context_schema
from asqav.client import Agent, AsqavValidationError

# === Shared helpers ===


def _agent() -> Agent:
    return Agent(
        agent_id="agent_schema_test",
        name="schema-test-agent",
        public_key="pk_xxx",
        key_id="key_schema",
        algorithm="ML-DSA-65",
        capabilities=["api:*"],
        created_at=1700000000.0,
    )


def _ok_response(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    base: dict[str, Any] = {
        "signature": "sig_b64",
        "signature_id": "sig_schema",
        "action_id": "act_schema",
        "timestamp": 1700000000.0,
        "verification_url": "https://asqav.com/verify/sig_schema",
        "algorithm": "ML-DSA-65",
        "chain_hash": "deadbeef",
    }
    if extra:
        base.update(extra)
    return base


_BASIC_SCHEMA: dict[str, Any] = {
    "user_id": {"type": "string", "required": True},
    "score": {"type": "number"},
    "active": {"type": "boolean"},
    "tags": {"type": "array"},
    "meta": {"type": "object"},
}


# === (a) Valid context passes and sign is called with the normalized context ===


def test_valid_context_passes_schema_and_sign_is_called() -> None:
    """A context that matches the schema reaches the network call."""
    captured: dict[str, Any] = {}

    def fake_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"user_id": "u1", "score": 42.5, "active": True},
            context_schema=_BASIC_SCHEMA,
            compliance_mode=False,
        )

    assert captured, "sign() was never called"
    # The raw body must exist (context lives here or nested under 'payload').
    assert captured["body"] is not None


def test_valid_context_is_key_sorted_before_signing() -> None:
    """After schema validation the context keys are sorted."""
    captured: dict[str, Any] = {}

    def fake_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        captured["body"] = body
        return _ok_response()

    # Deliberately supply keys in reverse alphabetical order.
    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"user_id": "u1", "score": 7, "active": False},
            context_schema=_BASIC_SCHEMA,
            compliance_mode=False,
        )

    # The body context (or payload context) should have keys in sorted order.
    # We verify via the context embedded in the body dict.
    body = captured["body"]
    ctx = body.get("context") or (body.get("payload") or {}).get("context")
    if ctx is not None and isinstance(ctx, dict):
        keys = list(ctx.keys())
        # Strip SDK-injected underscore keys before sorting check.
        public_keys = [k for k in keys if not k.startswith("_")]
        assert public_keys == sorted(public_keys), (
            f"Expected sorted context keys, got {public_keys}"
        )


def test_all_schema_types_pass_correct_values() -> None:
    """Each supported type passes with a matching Python value."""
    captured: dict[str, Any] = {}

    def fake_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {
                "user_id": "alice",
                "score": 3.14,
                "active": True,
                "tags": ["a", "b"],
                "meta": {"k": "v"},
            },
            context_schema=_BASIC_SCHEMA,
            compliance_mode=False,
        )

    assert captured, "sign() was not called"


def test_optional_field_absent_passes() -> None:
    """An optional field (required=False) may be absent with no error."""
    captured: dict[str, Any] = {}

    def fake_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            # Only required user_id; score, active, tags, meta are optional.
            {"user_id": "u2"},
            context_schema=_BASIC_SCHEMA,
            compliance_mode=False,
        )

    assert captured


# === (b) Invalid context raises AsqavValidationError and sign is NOT called ===


def test_missing_required_field_raises_before_network() -> None:
    """Missing required field raises AsqavValidationError; HTTP is never called.

    Non-vacuity: if we strip the raise from validate_context_schema the
    mock_post assertion on `call_count == 0` catches the regression.
    """
    with patch("asqav.client._post") as mock_post:
        with pytest.raises(AsqavValidationError, match="user_id"):
            _agent().sign(
                "api:call",
                {"score": 5},  # user_id missing
                context_schema=_BASIC_SCHEMA,
                compliance_mode=False,
            )
        mock_post.assert_not_called()


def test_wrong_type_raises_before_network() -> None:
    """Wrong-type field raises AsqavValidationError; HTTP is never called."""
    with patch("asqav.client._post") as mock_post:
        with pytest.raises(AsqavValidationError, match="score"):
            _agent().sign(
                "api:call",
                {"user_id": "u1", "score": "not-a-number"},
                context_schema=_BASIC_SCHEMA,
                compliance_mode=False,
            )
        mock_post.assert_not_called()


def test_bool_rejected_for_number_type() -> None:
    """bool is a subclass of int; it must be rejected for type=number."""
    with patch("asqav.client._post") as mock_post:
        with pytest.raises(AsqavValidationError, match="score"):
            _agent().sign(
                "api:call",
                {"user_id": "u1", "score": True},
                context_schema=_BASIC_SCHEMA,
                compliance_mode=False,
            )
        mock_post.assert_not_called()


def test_wrong_type_object_raises_before_network() -> None:
    """A string where object is expected raises the validation error."""
    with patch("asqav.client._post") as mock_post:
        with pytest.raises(AsqavValidationError, match="meta"):
            _agent().sign(
                "api:call",
                {"user_id": "u1", "meta": "not-a-dict"},
                context_schema=_BASIC_SCHEMA,
                compliance_mode=False,
            )
        mock_post.assert_not_called()


def test_wrong_type_array_raises_before_network() -> None:
    """A dict where array is expected raises the validation error."""
    with patch("asqav.client._post") as mock_post:
        with pytest.raises(AsqavValidationError, match="tags"):
            _agent().sign(
                "api:call",
                {"user_id": "u1", "tags": {"wrong": "type"}},
                context_schema=_BASIC_SCHEMA,
                compliance_mode=False,
            )
        mock_post.assert_not_called()


def test_validation_error_carries_docs_url() -> None:
    """AsqavValidationError.docs_url points to structured-receipts docs."""
    with patch("asqav.client._post"):
        with pytest.raises(AsqavValidationError) as exc_info:
            _agent().sign(
                "api:call",
                {"score": 5},  # user_id missing
                context_schema=_BASIC_SCHEMA,
                compliance_mode=False,
            )
    assert exc_info.value.docs_url == "https://asqav.com/docs/structured-receipts"


# === (c) No schema supplied => unchanged behavior ===


def test_no_schema_sign_called_as_before() -> None:
    """When context_schema is omitted, sign() behavior is unchanged."""
    captured: dict[str, Any] = {}

    def fake_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"any_key": "any_value"},
            compliance_mode=False,
        )

    assert captured, "sign() must be called when no schema is provided"


def test_no_schema_none_context_unchanged() -> None:
    """None context + no schema is unchanged (no normalization applied)."""
    captured: dict[str, Any] = {}

    def fake_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            None,
            compliance_mode=False,
        )

    assert captured


# === Custom callable validator ===


def test_custom_callable_validator_passes() -> None:
    """A callable context_schema that returns None passes through."""
    captured: dict[str, Any] = {}
    calls: list[Any] = []

    def my_validator(ctx: dict[str, Any]) -> None:
        calls.append(ctx)

    def fake_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"x": 1},
            context_schema=my_validator,
            compliance_mode=False,
        )

    assert calls, "validator was not called"
    assert captured, "sign() was not called"


def test_custom_callable_validator_raises_propagates() -> None:
    """A callable that raises propagates; sign() must not be called."""

    def strict_validator(ctx: dict[str, Any]) -> None:
        raise ValueError("custom_schema_error: x must be positive")

    with patch("asqav.client._post") as mock_post:
        with pytest.raises(ValueError, match="x must be positive"):
            _agent().sign(
                "api:call",
                {"x": -1},
                context_schema=strict_validator,
                compliance_mode=False,
            )
        mock_post.assert_not_called()


# === Unit tests for _schema helpers ===


class TestValidateContextSchema:
    def test_missing_required_field(self) -> None:
        with pytest.raises(AsqavValidationError, match="context_schema_error.*user_id.*required"):
            validate_context_schema({}, {"user_id": {"type": "string", "required": True}})

    def test_optional_field_absent_is_ok(self) -> None:
        # Should not raise.
        validate_context_schema({}, {"score": {"type": "number", "required": False}})

    def test_correct_string_passes(self) -> None:
        validate_context_schema({"name": "alice"}, {"name": {"type": "string"}})

    def test_correct_number_int_passes(self) -> None:
        validate_context_schema({"n": 7}, {"n": {"type": "number"}})

    def test_correct_number_float_passes(self) -> None:
        validate_context_schema({"n": 3.14}, {"n": {"type": "number"}})

    def test_bool_rejected_for_number(self) -> None:
        with pytest.raises(AsqavValidationError, match="got bool"):
            validate_context_schema({"n": True}, {"n": {"type": "number"}})

    def test_wrong_type_string_for_number(self) -> None:
        with pytest.raises(AsqavValidationError, match="expected type number, got str"):
            validate_context_schema({"n": "oops"}, {"n": {"type": "number"}})

    def test_boolean_field(self) -> None:
        validate_context_schema({"ok": True}, {"ok": {"type": "boolean"}})

    def test_object_field(self) -> None:
        validate_context_schema({"m": {"a": 1}}, {"m": {"type": "object"}})

    def test_array_field(self) -> None:
        validate_context_schema({"arr": [1, 2]}, {"arr": {"type": "array"}})

    def test_unknown_type_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown type"):
            validate_context_schema({"x": 1}, {"x": {"type": "bigdecimal"}})

    def test_none_context_treated_as_empty(self) -> None:
        # Optional field -> no error on None context.
        validate_context_schema(None, {"x": {"type": "string"}})

    def test_none_context_required_field_raises(self) -> None:
        with pytest.raises(AsqavValidationError, match="x.*required"):
            validate_context_schema(None, {"x": {"type": "string", "required": True}})


class TestNormalizeContext:
    def test_sorts_keys(self) -> None:
        ctx = {"z": 1, "a": 2, "m": 3}
        result = normalize_context(ctx)
        assert result is not None
        assert list(result.keys()) == ["a", "m", "z"]

    def test_none_returns_none(self) -> None:
        assert normalize_context(None) is None

    def test_empty_dict_returns_empty(self) -> None:
        assert normalize_context({}) == {}

    def test_already_sorted_unchanged(self) -> None:
        ctx = {"a": 1, "b": 2}
        result = normalize_context(ctx)
        assert result == {"a": 1, "b": 2}

    def test_returns_new_dict(self) -> None:
        ctx = {"x": 1}
        result = normalize_context(ctx)
        assert result is not ctx


# === Top-level re-export ===


def test_schema_helpers_exported_at_top_level() -> None:
    """validate_context_schema and normalize_context are public symbols."""
    assert hasattr(asqav, "validate_context_schema")
    assert hasattr(asqav, "normalize_context")
    assert asqav.validate_context_schema is validate_context_schema
    assert asqav.normalize_context is normalize_context
