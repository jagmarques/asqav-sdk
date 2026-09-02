"""JCS canonicalization for the IETF Compliance Receipts profile.

Byte-identical to the cloud's ``core/canonical.py:canonical_json``; see
draft-marques-asqav-compliance-receipts. ``asqav.canonicalize.canonicalize`` is
the back-compat alias and delegates here, so the two cannot drift apart.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["canonical_json"]


    # UTF-16 code-unit sort key for a JSON member name (RFC 8785 section 3.2.3).
def _utf16_order(key: str) -> bytes:
    # Big-endian UTF-16 bytes compare in the same order as the code units they encode.
    return key.encode("utf-16-be")


    # The member name ``json.dumps`` would emit for a non-string key.
def _coerce_member_name(key: Any) -> str:
    if key is True:
        return "true"
    if key is False:
        return "false"
    if key is None:
        return "null"
    if isinstance(key, float):
        return repr(key)
    return str(key)


    # Rebuild ``obj`` with every object's members in RFC 8785 key order.
def _utf16_ordered(obj: Any) -> Any:
    if isinstance(obj, dict):
        keyed = [(k if isinstance(k, str) else _coerce_member_name(k), k) for k in obj]
        keyed.sort(key=lambda pair: _utf16_order(pair[0]))
        return {name: _utf16_ordered(obj[original]) for name, original in keyed}
    if isinstance(obj, list):
        return [_utf16_ordered(v) for v in obj]
    return obj


    # Return canonical JCS bytes for ``obj``, byte-identical to the cloud.
#: Largest integer magnitude both SDKs canonicalise identically: 2**53 is exactly
#: representable and both emit the same digits for it, while 2**53 + 1 has no exact
#: double and JavaScript rounds it. One ABOVE JavaScript's Number.isSafeInteger bound,
#: deliberately: the upstream interop vector `number_2_to_53` pins 2**53 as canonical.
#: The bound also excludes every integer at or above 1e21, where JavaScript's toString
#: switches to exponential notation and Python's str does not.
MAX_CANONICAL_INTEGER = 2**53


class UnsafeIntegerError(ValueError):
    """An integer outside the safe range reached the canonicaliser."""


    # Refuse an int with no exact double, at every depth, before any bytes are produced.
def _reject_unsafe_integers(obj: Any) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, int):
        if not -MAX_CANONICAL_INTEGER <= obj <= MAX_CANONICAL_INTEGER:
            raise UnsafeIntegerError(
                f"integer outside the canonical integer range +/-2**53: {obj}; serialise it "
                "as a JSON string or an integer-rational pair"
            )
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _reject_unsafe_integers(value)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _reject_unsafe_integers(value)


def canonical_json(obj: Any) -> bytes:
    # UTF-16 key order per RFC 8785 3.2.3; ``sort_keys=True`` is code-point
    # order, which diverges only above U+FFFF.
    _reject_unsafe_integers(obj)
    return json.dumps(
        _utf16_ordered(obj),
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
