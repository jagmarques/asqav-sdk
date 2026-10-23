"""Augment-match helper for destructive SQL action types.
`data:write:sql:DELETE ..` must match `data:write:sql:*` and `data:delete:*`."""

from __future__ import annotations

import re

_WRITE_SQL_PREFIX = "data:write:sql:"
_DELETE_SQL_PREFIX = "data:delete:sql:"

# Destructive SQL verbs as whole words (non-letter boundary on each side).
# Underscore separates words: DELETE_FROM matches, deleted_at does not.
_DESTRUCTIVE_VERB_RE = re.compile(
    r"(?<![A-Za-z])(DELETE|DROP|TRUNCATE|ALTER|GRANT|REVOKE|REPLACE|COPY|UPSERT)(?![A-Za-z])",
    re.IGNORECASE,
)


def action_candidates(action_type: str) -> list[str]:
    """Return match candidates for policy evaluation.
    The normalized action_type, plus a `data:delete:sql:` one if destructive."""
    normalized = action_type.strip().lower()
    if not normalized.startswith(_WRITE_SQL_PREFIX):
        return [normalized]
    suffix = normalized[len(_WRITE_SQL_PREFIX) :]
    if _DESTRUCTIVE_VERB_RE.search(suffix):
        return [normalized, _DELETE_SQL_PREFIX + suffix]
    return [normalized]


def matches_pattern(pattern: str, action_type: str) -> bool:
    """Return True if any candidate for action_type matches the pattern.

    Matching is case-insensitive end to end: both the policy pattern and the
    action candidates are lowercased before comparison.
    """
    norm_pattern = pattern.strip().lower()
    prefix = norm_pattern.rstrip("*")
    for candidate in action_candidates(action_type):
        if norm_pattern == "*" or candidate.startswith(prefix):
            return True
    return False
