"""Strict JSON ingest for receipts and records (criterion 419).

Every receipt- and record-parsing path routes through ``loads`` so a duplicated
JSON member name at ANY nesting depth is a terminal parse failure, raised before
any hashing, canonicalisation, or signature check. The stdlib ``json`` decoder is
last-wins on duplicate members, which would silently hash the bytes an attacker
kept and drop the ones they replaced; refusing the document outright is the only
fail-closed reading.

The standalone verifier (``verifier/verify_receipt.py``) ships as one file and
carries its own copy of this hook rather than importing this module.

Integers outside +/-2**53 are refused here too. Beyond that magnitude an integer
has no exact IEEE-754 double, so a JavaScript reader rounds it while Python keeps
it: the same receipt canonicalises to two different byte strings and its signature
verifies in one implementation and fails in the other. Refusing at ingest is the
only reading that keeps the two SDKs byte-identical.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "DuplicateMemberError",
    "UnsafeIntegerError",
    "MAX_CANONICAL_INTEGER",
    "reject_duplicate_members",
    "reject_unsafe_integer",
    "loads",
]

#: Largest integer magnitude both SDKs canonicalise identically: 2**53 is exactly
#: representable and both emit the same digits for it, while 2**53 + 1 has no exact
#: double and JavaScript rounds it. One ABOVE JavaScript's Number.isSafeInteger bound,
#: deliberately: the upstream interop vector `number_2_to_53` pins 2**53 as canonical.
#: The bound also excludes every integer at or above 1e21, where JavaScript's toString
#: switches to exponential notation and Python's str does not.
MAX_CANONICAL_INTEGER = 2**53


class DuplicateMemberError(ValueError):
    """A JSON object repeated a member name; last-wins ingest is never allowed."""


class UnsafeIntegerError(ValueError):
    """An integer outside the safe range; the two SDKs would canonicalise it differently."""


    # object_pairs_hook: reject a repeated member name, else build the object.
def reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise DuplicateMemberError(f"duplicate JSON member name: {key!r}")
        out[key] = value
    return out


    # parse_int: refuse an integer literal with no exact double (finding 8).
def reject_unsafe_integer(literal: str) -> int:
    value = int(literal)
    if not -MAX_CANONICAL_INTEGER <= value <= MAX_CANONICAL_INTEGER:
        raise UnsafeIntegerError(
            f"integer outside the canonical integer range +/-2**53: {literal}; serialise it "
            "as a JSON string or an integer-rational pair"
        )
    return value


    # Parse JSON with duplicate-member and unsafe-integer rejection at every depth.
def loads(text: str | bytes, **kwargs: Any) -> Any:
    for reserved in ("object_pairs_hook", "parse_int"):
        if reserved in kwargs:
            raise ValueError(f"{reserved} is reserved for strict ingest")
    return json.loads(
        text,
        object_pairs_hook=reject_duplicate_members,
        parse_int=reject_unsafe_integer,
        **kwargs,
    )
