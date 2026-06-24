"""Helper for augment-match on destructive SQL action types.

A write action like `data:write:sql:DELETE FROM users` must match BOTH
`data:write:sql:*` (write policies) AND `data:delete:*` (destructive policies).

This is done by returning the full set of action_type candidates for matching:
- Normal: [action_type]
- Destructive write: [action_type, derived] where derived replaces the
  `data:write:sql:` prefix with `data:delete:sql:`.

Only `data:write:sql:` namespaced actions are inspected; reads and non-sql
namespaces are never augmented, so a SELECT mentioning "delete" is unaffected.
"""

from __future__ import annotations

import re

_WRITE_SQL_PREFIX = "data:write:sql:"
_DELETE_SQL_PREFIX = "data:delete:sql:"

# Matches DELETE, DROP, TRUNCATE, ALTER as whole words (non-letter boundary on each side).
# Underscore counts as a word separator (e.g. DELETE_FROM matches, deleted_at does not).
_DESTRUCTIVE_VERB_RE = re.compile(
    r"(?<![A-Za-z])(DELETE|DROP|TRUNCATE|ALTER)(?![A-Za-z])",
    re.IGNORECASE,
)


def action_candidates(action_type: str) -> list[str]:
    """Return match candidates for policy evaluation.

    For most actions returns [action_type]. For a destructive SQL write, also
    returns a derived `data:delete:sql:` entry so destructive policies fire.
    """
    if not action_type.startswith(_WRITE_SQL_PREFIX):
        return [action_type]
    suffix = action_type[len(_WRITE_SQL_PREFIX):]
    if _DESTRUCTIVE_VERB_RE.search(suffix):
        return [action_type, _DELETE_SQL_PREFIX + suffix]
    return [action_type]


def matches_pattern(pattern: str, action_type: str) -> bool:
    """Return True if any candidate for action_type matches the pattern."""
    prefix = pattern.rstrip("*")
    for candidate in action_candidates(action_type):
        if pattern == "*" or candidate.startswith(prefix):
            return True
    return False
