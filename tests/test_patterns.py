"""Tests for semantic action patterns."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.patterns import PATTERNS, list_patterns, resolve_pattern


# ---------------------------------------------------------------------------
# resolve_pattern tests
# ---------------------------------------------------------------------------


def test_resolve_known_sql_read() -> None:
    """resolve_pattern expands 'sql-read' to its glob."""
    assert resolve_pattern("sql-read") == "data:read:sql:*"


def test_resolve_known_sql_write() -> None:
    assert resolve_pattern("sql-write") == "data:write:sql:*"


def test_resolve_known_sql_destructive() -> None:
    assert resolve_pattern("sql-destructive") == "data:delete:*"


def test_resolve_known_file_read() -> None:
    assert resolve_pattern("file-read") == "file:read:*"


def test_resolve_known_file_write() -> None:
    assert resolve_pattern("file-write") == "file:write:*"


def test_resolve_known_file_delete() -> None:
    assert resolve_pattern("file-delete") == "file:delete:*"


def test_resolve_known_http_external() -> None:
    assert resolve_pattern("http-external") == "api:call:external:*"


def test_resolve_known_shell_execute() -> None:
    assert resolve_pattern("shell-execute") == "system:shell:*"


def test_resolve_known_email_send() -> None:
    assert resolve_pattern("email-send") == "comms:email:send:*"


def test_resolve_known_payment_process() -> None:
    assert resolve_pattern("payment-process") == "finance:payment:*"


def test_resolve_known_pii_access() -> None:
    assert resolve_pattern("pii-access") == "data:read:pii:*"


def test_resolve_known_admin_action() -> None:
    assert resolve_pattern("admin-action") == "admin:*"


# ---------------------------------------------------------------------------
# Passthrough for unknown patterns
# ---------------------------------------------------------------------------


def test_resolve_unknown_passthrough() -> None:
    """Unknown patterns are returned as-is."""
    assert resolve_pattern("custom:my:action") == "custom:my:action"


def test_resolve_empty_string_passthrough() -> None:
    assert resolve_pattern("") == ""


def test_resolve_arbitrary_string_passthrough() -> None:
    assert resolve_pattern("some-random-thing") == "some-random-thing"


# ---------------------------------------------------------------------------
# list_patterns tests
# ---------------------------------------------------------------------------


def test_list_patterns_returns_all() -> None:
    """list_patterns returns all 12 entries."""
    result = list_patterns()
    assert len(result) == 12


def test_list_patterns_returns_dict() -> None:
    result = list_patterns()
    assert isinstance(result, dict)


def test_list_patterns_contains_expected_keys() -> None:
    result = list_patterns()
    expected_keys = {
        "sql-read",
        "sql-write",
        "sql-destructive",
        "file-read",
        "file-write",
        "file-delete",
        "http-external",
        "shell-execute",
        "email-send",
        "payment-process",
        "pii-access",
        "admin-action",
    }
    assert set(result.keys()) == expected_keys


def test_list_patterns_is_copy() -> None:
    """list_patterns returns a copy, not the original dict."""
    result = list_patterns()
    result["new-key"] = "new-value"
    assert "new-key" not in PATTERNS
