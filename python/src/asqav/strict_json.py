"""Strict JSON ingest for receipts and records.
Duplicate member names and integers beyond +/-2**53 fail before any hashing."""

from __future__ import annotations

import json
import re
from typing import Any

#: A float literal without fraction or exponent parts is an exact integer.
_INTEGER_LEXEME_RE = re.compile(r"-?\d+$")

__all__ = [
    "DuplicateMemberError",
    "UnsafeIntegerError",
    "ProfileIntegerError",
    "MAX_CANONICAL_INTEGER",
    "MAX_PROFILE_INTEGER",
    "reject_duplicate_members",
    "reject_unsafe_integer",
    "loads",
    "loads_profile",
    "assert_profile_integers",
    "assert_canonical_integers",
]

#: Largest integer both SDKs canonicalise identically (no exact double above).
#: One above Number.isSafeInteger; vector `number_2_to_53` pins 2**53.
MAX_CANONICAL_INTEGER = 2**53

#: Largest integer magnitude the Asqav profile admits: draft Section 4 ends the
#: safe interval at 2**53 - 1, one below the shared generic bound above.
MAX_PROFILE_INTEGER = 2**53 - 1


class DuplicateMemberError(ValueError):
    """A JSON object repeated a member name; last-wins ingest is never allowed."""


class UnsafeIntegerError(ValueError):
    """An integer outside the safe range; the two SDKs would canonicalise it differently."""


class ProfileIntegerError(ValueError):
    """An integer the Asqav profile refuses; exactly representable yet out of range."""


class RawFloat(float):
    """A float literal preserved with its source lexeme for profile judgment.

    Behaves as a plain float everywhere; the profile range judges the
    lexeme digits, never the parsed double.
    """

    def __new__(cls, value: float, lexeme: str) -> "RawFloat":
        obj = super().__new__(cls, value)
        obj.lexeme = lexeme
        return obj


def _raw_float(literal: str) -> RawFloat:
    return RawFloat(float(literal), literal)


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


    # Parse JSON rejecting duplicate members and unsafe integers at any depth.
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


    # Refuse any parsed int or integral float outside the profile interval.
def assert_profile_integers(obj: Any) -> None:
    stack: list[tuple[Any, str]] = [(obj, "$")]
    while stack:
        node, path = stack.pop()
        if isinstance(node, bool):
            continue
        if isinstance(node, float):
            # A parsed float came from a float lexeme: 9007199254740991.0 equals
            # 9007199254740991 but the digest binds unvalidated digits.
            lexeme = node.lexeme if isinstance(node, RawFloat) else repr(node)
            if (
                not _INTEGER_LEXEME_RE.match(lexeme)
                and node.is_integer()
                and -MAX_PROFILE_INTEGER <= node <= MAX_PROFILE_INTEGER
            ):
                raise ProfileIntegerError(
                    f"non-integer lexeme in the Asqav profile range: "
                    f"{lexeme!r} at {path}; serialise it as a JSON string"
                )
            # Fall through: out-of-range magnitudes still fail below.
        if isinstance(node, int):
            if not -MAX_PROFILE_INTEGER <= node <= MAX_PROFILE_INTEGER:
                raise ProfileIntegerError(
                    f"integer outside the Asqav profile range +/-(2**53 - 1): "
                    f"{node} at {path}; serialise it as a JSON string"
                )
            continue
        if isinstance(node, float) and node.is_integer():
            if not -MAX_PROFILE_INTEGER <= node <= MAX_PROFILE_INTEGER:
                raise ProfileIntegerError(
                    f"number outside the Asqav profile range +/-(2**53 - 1): "
                    f"{node!r} at {path}; serialise it as a JSON string"
                )
            continue
        if isinstance(node, dict):
            stack.extend((v, f"{path}.{k}") for k, v in node.items())
        elif isinstance(node, (list, tuple)):
            stack.extend((v, f"{path}[{i}]") for i, v in enumerate(node))


    # Refuse a parsed int outside +/-2**53 at any depth.
    # Dict API callers get the same refusal as raw text callers.
def assert_canonical_integers(obj: Any) -> None:
    stack: list[Any] = [obj]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if isinstance(node, (dict, list, tuple)):
            if id(node) in seen:  # a shared or cyclic container is walked once
                continue
            seen.add(id(node))
        if isinstance(node, int) and not isinstance(node, bool):
            if not -MAX_CANONICAL_INTEGER <= node <= MAX_CANONICAL_INTEGER:
                raise UnsafeIntegerError(
                    "integer outside the canonical integer range +/-2**53; serialise it "
                    "as a JSON string or an integer-rational pair"
                )
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)


    # Strict ingest plus the narrower Asqav profile range.
def loads_profile(text: str | bytes, **kwargs: Any) -> Any:
    for reserved in ("object_pairs_hook", "parse_int", "parse_float"):
        if reserved in kwargs:
            raise ValueError(f"{reserved} is reserved for strict ingest")
    obj = json.loads(
        text,
        object_pairs_hook=reject_duplicate_members,
        parse_int=reject_unsafe_integer,
        parse_float=_raw_float,
        **kwargs,
    )
    assert_profile_integers(obj)
    return obj
