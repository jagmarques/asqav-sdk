"""A forged anchor value must never read as present on the anchors axis.

The anchors field sits outside the signed bytes, so it is attacker steerable. A
value that is not base64, or that carries no bytes, must not take the axis to
PASS. A lenient base64 decode drops out-of-alphabet characters, which lets an
all-punctuation value decode to zero bytes and still report as present.
"""

from __future__ import annotations

import pytest

from asqav.verifier.verify_receipt import _safe_b64, check_anchors

ENVELOPE = {
    "payload": {"type": "protectmcp:decision", "issued_at": "2026-06-19T00:00:00+00:00"},
    "signature": {"alg": "ML-DSA-65", "kid": "k1", "sig": "AAAA"},
}

# Real base64, including the url-safe alphabet and the conformance vector.
LEGITIMATE = [
    "AA",
    "AAA",
    "AAAA",
    "YQ==",
    "MTIzNA==",
    "AAAA-_AA",
    "AAAA_synthetic_tsr_base64_placeholder_AAAA",
    "A" * 64,
]

# Not base64, or base64 carrying no bytes at all.
FORGED = [
    "!!!!",  # the reported case
    "@@@@",
    "....",
    "****",
    "!",
    "!!",
    "!!!",
    "=",
    "==",
    "===",
    "====",
    " ",
    "\t",
    "\n",
    "\r\n",
    "<script>alert(1)</script>",
    "YQ==!!!!",  # junk after a decodable group
    "!!!!YQ==",  # junk before one
    "YQ==YQ==",  # a pad in the middle
    "MTIzNA==\n",
    "\x00",
    "é",
    "中文",
    "😀",
    " ",
]

NON_STRINGS = [None, 0, 123, 1.5, True, [], {}, ["AAAA"], {"value": "AAAA"}, b"AAAA"]


def _axis(value: object) -> tuple[str, str]:
    env = dict(ENVELOPE)
    env["anchors"] = [{"type": "rfc3161", "value": value}]
    return check_anchors(env)


def test_reported_forged_punctuation_anchor_fails() -> None:
    """The reported case: an all-punctuation value decoded to nothing and passed."""
    state, note = _axis("!!!!")
    assert state == "FAIL"
    assert "base64-ok" not in note


@pytest.mark.parametrize("value", FORGED, ids=repr)
def test_forged_value_never_reads_as_present(value: str) -> None:
    assert _safe_b64(value) is False, repr(value)
    state, note = _axis(value)
    assert state == "FAIL", repr(value)
    assert "base64-ok" not in note, repr(value)


@pytest.mark.parametrize("value", LEGITIMATE, ids=repr)
def test_legitimate_anchor_still_passes(value: str) -> None:
    assert _safe_b64(value) is True, repr(value)
    state, note = _axis(value)
    assert state == "PASS", repr(value)
    assert "present, base64-ok" in note, repr(value)


@pytest.mark.parametrize("value", NON_STRINGS, ids=repr)
def test_non_string_value_never_reads_as_present(value: object) -> None:
    assert _safe_b64(value) is False, repr(value)
    assert _axis(value)[0] == "FAIL", repr(value)


def test_valid_sibling_does_not_launder_a_forged_anchor() -> None:
    env = dict(ENVELOPE)
    env["anchors"] = [
        {"type": "rfc3161", "value": "AAAA"},
        {"type": "forged", "value": "!!!!"},
    ]
    assert check_anchors(env)[0] == "FAIL"


def test_a_decoding_anchor_carries_bytes() -> None:
    """Every accepted value yields at least one byte, which is the property asserted."""
    import base64

    for value in LEGITIMATE:
        s = value.replace("-", "+").replace("_", "/")
        s += "=" * ((-len(s)) % 4)
        assert len(base64.b64decode(s, validate=True)) > 0, repr(value)
