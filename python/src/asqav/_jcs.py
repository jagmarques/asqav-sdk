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
def canonical_json(obj: Any) -> bytes:
    # UTF-16 key order per RFC 8785 3.2.3; ``sort_keys=True`` is code-point
    # order, which diverges only above U+FFFF.
    return json.dumps(
        _utf16_ordered(obj),
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
