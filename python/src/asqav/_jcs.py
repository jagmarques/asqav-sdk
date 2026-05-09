"""JCS canonicalization helper for the IETF Compliance Receipts profile.

Public surface: :func:`canonical_json` returning ``bytes``. The output
is the exact bytes the cloud's ``core/canonical.py:canonical_json``
produces and the bytes the chain hashes over.

See https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/

This module is a named landing for the IETF Compliance Receipts profile
so callers can ``from asqav._jcs import canonical_json`` to get the same
function under the spec-mandated name.
``asqav.canonicalize.canonicalize`` is kept as a public alias for
backwards compatibility; the two are byte-identical for the JSON value
types the profile actually signs (objects whose leaves are strings,
booleans, null, integers, and nested arrays / objects).

Implementation notes:

  - Lexicographic ordering of object keys (UTF-16 code unit order, which
    Python ``sort_keys=True`` produces for the ASCII / BMP keys this
    codebase uses).
  - No insignificant whitespace; ``,`` and ``:`` separators only.
  - UTF-8 byte output with no BOM. Non-ASCII passes through verbatim.
  - ``NaN`` / ``Infinity`` rejected.
  - String escapes follow standard JSON rules (``\\``, ``\"``, ``\b``,
    ``\f``, ``\n``, ``\r``, ``\t``, plus ``\\u00xx`` for control chars
    U+0000..U+001F). All other characters pass through, matching
    ECMAScript ``JSON.stringify``.

Float caveat: Python ``json.dumps`` and ECMAScript ``Number.toString``
agree byte-for-byte on integer values within the IEEE-754 safe range,
which is what the Compliance Receipts profile signs. Floats are not
covered by the conformance vectors and SHOULD NOT appear in signed
envelopes.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["canonical_json"]


def canonical_json(obj: Any) -> bytes:
    """Return canonical JCS bytes for ``obj``.

    Same function as :func:`asqav.canonicalize.canonicalize`; exposed
    under the spec-mandated name so IETF profile code reads naturally.

    Identical bytes to the cloud's ``core/canonical.py:canonical_json``
    so customer-side digests reproduce the bytes the server signs over
    and the chain hashes over.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
