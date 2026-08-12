"""Strict JSON ingest (criterion 419) - duplicate member names are terminal.

Every receipt- and record-parsing path in the SDK parses through this module so
a repeated JSON member name at ANY depth is a parse failure that lands BEFORE
any hashing, canonicalisation, or signature check. The RFC 8259 default of
silently letting the last duplicate win is a smuggling primitive: one parser
reads the first value, another reads the last, and a signature checked over the
collapsed bytes binds neither reading. Rejecting at ingest means the bytes of a
receipt are always read exactly once, identically, by every parser.

There is no lenient mode and no fallback to partially accepted bytes.
"""
from __future__ import annotations

import json
from typing import IO, Any

__all__ = ["DuplicateJsonMemberError", "strict_loads", "strict_load"]


class DuplicateJsonMemberError(ValueError):
    """A JSON text repeated a member name; strict ingest rejects it terminally."""


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """object_pairs_hook: raise on a repeated member name in ANY JSON object."""
    seen: set[str] = set()
    for key, _value in pairs:
        if key in seen:
            raise DuplicateJsonMemberError(
                f"duplicate member name {key!r}: one member may not override another"
            )
        seen.add(key)
    return dict(pairs)


def strict_loads(text: str) -> Any:
    """Parse JSON text, refusing duplicate member names at any depth."""
    return json.loads(text, object_pairs_hook=_reject_duplicate_members)


def strict_load(fh: IO[str]) -> Any:
    """Parse a JSON stream, refusing duplicate member names at any depth."""
    return json.load(fh, object_pairs_hook=_reject_duplicate_members)
