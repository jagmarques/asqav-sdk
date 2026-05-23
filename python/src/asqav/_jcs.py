"""JCS canonicalization for the IETF Compliance Receipts profile.

Byte-identical to the cloud's ``core/canonical.py:canonical_json``; see
draft-marques-asqav-compliance-receipts. ``asqav.canonicalize.canonicalize`` is
the back-compat alias.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["canonical_json"]


def canonical_json(obj: Any) -> bytes:
    """Return canonical JCS bytes for ``obj``, byte-identical to the cloud."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
