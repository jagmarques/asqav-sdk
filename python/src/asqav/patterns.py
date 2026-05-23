"""Semantic action patterns mapping human-friendly names to glob action-type strings."""

from __future__ import annotations

PATTERNS: dict[str, str] = {
    "sql-read": "data:read:sql:*",
    "sql-write": "data:write:sql:*",
    "sql-destructive": "data:delete:*",
    "file-read": "file:read:*",
    "file-write": "file:write:*",
    "file-delete": "file:delete:*",
    "http-external": "api:call:external:*",
    "shell-execute": "system:shell:*",
    "email-send": "comms:email:send:*",
    "payment-process": "finance:payment:*",
    "pii-access": "data:read:pii:*",
    "admin-action": "admin:*",
}


def resolve_pattern(pattern: str) -> str:
    """Resolve a semantic pattern name to its glob string; passthrough for unknown patterns."""
    return PATTERNS.get(pattern, pattern)


def list_patterns() -> dict[str, str]:
    """Return all available semantic action patterns."""
    return dict(PATTERNS)
