"""Semantic action patterns for asqav.

Maps human-friendly pattern names to glob-style action type strings,
so users can write agent.sign("sql-destructive", {...}) instead of
remembering the full action namespace.
"""

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
    """Resolve a semantic pattern name to its glob string.

    If the pattern matches a known semantic name, return the corresponding
    glob. Otherwise return the pattern as-is (passthrough for custom patterns).

    Args:
        pattern: A semantic name like "sql-read" or a raw action type.

    Returns:
        The resolved glob pattern string.
    """
    return PATTERNS.get(pattern, pattern)


def list_patterns() -> dict[str, str]:
    """Return all available semantic action patterns.

    Returns:
        Dict mapping semantic names to their glob patterns.
    """
    return dict(PATTERNS)
