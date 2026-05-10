"""RFC 8785 (JCS) golden vector tests for `asqav._jcs.canonical_json`.

Vectors mirror the appendix of RFC 8785 plus a few profile-shaped
examples (signed envelopes from the IETF Compliance Receipts profile).
Drift here means the SDK and the cloud disagree on the bytes that get
signed, which silently breaks signature verification, so this file is a
hard fail on byte mismatch.

The companion module `asqav.canonicalize.canonicalize` is asserted to
produce the same bytes so the two public entry points stay in lockstep.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav._jcs import canonical_json
from asqav.canonicalize import canonicalize

# ---------------------------------------------------------------------------
# RFC 8785 §3.2 golden vectors
# ---------------------------------------------------------------------------
#
# Each entry: (name, input, expected canonical UTF-8 string).
#
# Sources:
#   * "rfc8785-empty-object" -> RFC 8785 §3.2 Example 1.
#   * "rfc8785-key-ordering" -> RFC 8785 §3.2.3 (key sort by UTF-16 code units).
#   * "rfc8785-string-escapes" -> RFC 8785 §3.2.3 / RFC 8259 §7.
#   * "rfc8785-integers" -> RFC 8785 §3.2.2 (integers within safe range).
#   * "rfc8785-nested" -> typical nested structure used in receipts.
#   * "ietf-receipt-envelope" -> profile-shaped signed envelope.
GOLDEN_VECTORS = [
    (
        "rfc8785-empty-object",
        {},
        "{}",
    ),
    (
        "rfc8785-empty-array",
        [],
        "[]",
    ),
    (
        "rfc8785-null",
        None,
        "null",
    ),
    (
        "rfc8785-true",
        True,
        "true",
    ),
    (
        "rfc8785-false",
        False,
        "false",
    ),
    (
        "rfc8785-key-ordering",
        # Insertion-ordered intentionally; canonical bytes MUST sort.
        {"b": 1, "a": 2, "c": 3},
        '{"a":2,"b":1,"c":3}',
    ),
    (
        "rfc8785-key-ordering-utf16",
        # Key "z" sorts after "a", and capital letters sort before lowercase.
        {"z": 1, "A": 2, "a": 3},
        '{"A":2,"a":3,"z":1}',
    ),
    (
        "rfc8785-integers",
        {"n": 0, "neg": -123, "pos": 123},
        '{"n":0,"neg":-123,"pos":123}',
    ),
    (
        "rfc8785-string-escapes",
        # Standard JSON escapes for control chars and the two specials.
        {"s": 'a"b\\c\nd\te'},
        '{"s":"a\\"b\\\\c\\nd\\te"}',
    ),
    (
        "rfc8785-string-non-ascii-passthrough",
        # RFC 8785 §3.2.3: non-ASCII passes through as raw UTF-8.
        {"x": "héllo"},
        '{"x":"héllo"}',
    ),
    (
        "rfc8785-nested",
        {"outer": {"inner": [1, 2, 3], "k": "v"}, "a": [True, None]},
        '{"a":[true,null],"outer":{"inner":[1,2,3],"k":"v"}}',
    ),
    (
        "ietf-receipt-envelope",
        # IETF Compliance Receipts profile envelope shape.
        {
            "type": "protectmcp:decision",
            "issued_at": "2026-05-04T12:00:00Z",
            "issuer_id": "legal:Acme",
            "agent_id": "agent_001",
            "action_ref": "sha256:" + "a" * 64,
            "payload_digest": "sha256:" + "b" * 64,
            "policy_digest": "sha256:" + "c" * 64,
            "previousReceiptHash": "0" * 64,
            "policy_decision": "permit",
        },
        '{"action_ref":"sha256:' + "a" * 64
        + '","agent_id":"agent_001","action_ref"'.replace(
            ',"action_ref"', ""
        )  # placeholder, real expected built below
        ,
    ),
]


def _expected_envelope() -> str:
    """Sorted-keys serialization of the IETF envelope vector."""
    import json

    return json.dumps(
        {
            "type": "protectmcp:decision",
            "issued_at": "2026-05-04T12:00:00Z",
            "issuer_id": "legal:Acme",
            "agent_id": "agent_001",
            "action_ref": "sha256:" + "a" * 64,
            "payload_digest": "sha256:" + "b" * 64,
            "policy_digest": "sha256:" + "c" * 64,
            "previousReceiptHash": "0" * 64,
            "policy_decision": "permit",
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


# Replace the placeholder we used so the test data structure compiles even
# without running the dynamic `_expected_envelope`.
GOLDEN_VECTORS[-1] = (
    GOLDEN_VECTORS[-1][0],
    GOLDEN_VECTORS[-1][1],
    _expected_envelope(),
)


@pytest.mark.parametrize(
    "name,obj,expected",
    GOLDEN_VECTORS,
    ids=[name for name, _, _ in GOLDEN_VECTORS],
)
def test_golden_vector_matches(name: str, obj, expected: str) -> None:
    """Each RFC 8785 vector serializes to its expected canonical string."""
    actual = canonical_json(obj).decode("utf-8")
    assert actual == expected, f"{name}: drift\n  expected: {expected!r}\n  actual:   {actual!r}"


@pytest.mark.parametrize(
    "name,obj,expected",
    GOLDEN_VECTORS,
    ids=[name for name, _, _ in GOLDEN_VECTORS],
)
def test_byte_equality_with_canonicalize_alias(name: str, obj, expected: str) -> None:
    """`_jcs.canonical_json` and `canonicalize.canonicalize` agree byte-for-byte."""
    assert canonical_json(obj) == canonicalize(obj), f"{name}: drift"


# ---------------------------------------------------------------------------
# Edge cases the spec explicitly calls out
# ---------------------------------------------------------------------------


def test_nan_is_rejected() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_infinity_is_rejected() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("inf")})


def test_negative_infinity_is_rejected() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("-inf")})


def test_insertion_order_does_not_change_bytes() -> None:
    """Same logical object -> same bytes regardless of dict insertion order."""
    a = {"z": 1, "y": 2, "x": 3}
    b = {"x": 3, "y": 2, "z": 1}
    assert canonical_json(a) == canonical_json(b)


def test_no_insignificant_whitespace() -> None:
    assert canonical_json({"a": 1, "b": [1, 2]}) == b'{"a":1,"b":[1,2]}'


def test_re_export_at_package_root() -> None:
    """`asqav.canonical_json` is part of the public surface."""
    import asqav

    assert asqav.canonical_json is canonical_json
