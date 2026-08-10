"""Strict JSON ingest for receipts and records (criterion 419).

Every receipt- and record-parsing path routes through ``loads`` so a duplicated
JSON member name at ANY nesting depth is a terminal parse failure, raised before
any hashing, canonicalisation, or signature check. The stdlib ``json`` decoder is
last-wins on duplicate members, which would silently hash the bytes an attacker
kept and drop the ones they replaced; refusing the document outright is the only
fail-closed reading.

The standalone verifier (``verifier/verify_receipt.py``) ships as one file and
carries its own copy of this hook rather than importing this module.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["DuplicateMemberError", "reject_duplicate_members", "loads"]


class DuplicateMemberError(ValueError):
    """A JSON object repeated a member name; last-wins ingest is never allowed."""


    # object_pairs_hook: reject a repeated member name, else build the object.
def reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise DuplicateMemberError(f"duplicate JSON member name: {key!r}")
        out[key] = value
    return out


    # Parse JSON with duplicate-member rejection at every depth.
def loads(text: str | bytes, **kwargs: Any) -> Any:
    if "object_pairs_hook" in kwargs:
        raise ValueError("object_pairs_hook is reserved for duplicate-member rejection")
    return json.loads(text, object_pairs_hook=reject_duplicate_members, **kwargs)
