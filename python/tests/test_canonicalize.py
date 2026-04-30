"""Verify the SDK fingerprint helpers agree with conformance/vectors.json byte-for-byte.

If this file fails, either:

1. ``canonicalize.py`` drifted from RFC 8785 JCS subset, or
2. ``conformance/vectors.json`` was edited without regenerating derived
   fields (hash, canonical).

A test in ``test_conformance_vectors.py`` already checks the vectors are
internally consistent. This file additionally checks that the SDK's
public helpers match those vectors, which is what customers rely on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from asqav.canonicalize import canonicalize, hash_action

VECTORS_PATH = Path(__file__).parent.parent.parent / "conformance" / "vectors.json"


def _load_vectors() -> list[dict]:
    return json.loads(VECTORS_PATH.read_text())["vectors"]


@pytest.mark.parametrize("vector", _load_vectors(), ids=lambda v: v["name"])
def test_canonicalize_matches_vector(vector: dict) -> None:
    """The SDK's canonicalize() must produce the same bytes as vectors.json."""
    expected = vector["canonical"]
    actual = canonicalize(vector["input"]).decode("utf-8")
    assert actual == expected, f"{vector['name']}: canonical drift"


@pytest.mark.parametrize("vector", _load_vectors(), ids=lambda v: v["name"])
def test_sha256_matches_vector(vector: dict) -> None:
    """SHA-256 of the SDK's canonical bytes must match vectors.json."""
    canonical_bytes = canonicalize(vector["input"])
    actual = hashlib.sha256(canonical_bytes).hexdigest()
    assert actual == vector["sha256"], f"{vector['name']}: sha256 drift"


def test_hash_action_no_salt_matches_action_context_vector() -> None:
    """For vectors shaped {action_type, context}, hash_action() must agree."""
    vectors = _load_vectors()
    happy = [
        v for v in vectors
        if set(v["input"].keys()) == {"action_type", "context"}
    ]
    assert happy, "expected at least one {action_type, context} vector"
    for v in happy:
        result = hash_action(v["input"]["action_type"], v["input"]["context"])
        assert result == f"sha256:{v['sha256']}", f"{v['name']}: hash_action drift"


def test_hash_action_with_salt_uses_hmac() -> None:
    """With a salt, hash_action returns HMAC-SHA-256 not plain SHA-256."""
    salt = bytes.fromhex(
        "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"
    )
    plain = hash_action("api:call", {"k": 3})
    keyed = hash_action("api:call", {"k": 3}, salt=salt)
    assert plain != keyed
    assert plain.startswith("sha256:")
    assert keyed.startswith("sha256:")
    assert len(keyed) == len("sha256:") + 64


def test_canonicalize_is_deterministic_across_key_order() -> None:
    """Inserting keys in a different order yields the same canonical bytes."""
    a = {"b": 1, "a": 2, "c": [{"y": 1, "x": 2}]}
    b = {"a": 2, "c": [{"x": 2, "y": 1}], "b": 1}
    assert canonicalize(a) == canonicalize(b)


def test_canonicalize_preserves_unicode_raw() -> None:
    """Non-ASCII chars are emitted as raw UTF-8, not \\uXXXX escapes."""
    out = canonicalize({"name": "café"})
    assert "café".encode("utf-8") in out
    assert b"\\u" not in out


def test_canonicalize_rejects_nan() -> None:
    """NaN/Infinity are not valid JSON and must raise."""
    with pytest.raises(ValueError):
        canonicalize({"x": float("nan")})
