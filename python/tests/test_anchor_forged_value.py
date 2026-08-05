"""A forged anchor value must never read as present on the anchors axis.

The anchors field sits outside the signed bytes, so it is attacker steerable. A
value that is not base64, or that carries no bytes, must not take the axis to
PASS. A lenient base64 decode drops out-of-alphabet characters, which lets an
all-punctuation value decode to zero bytes and still report as present.
"""

from __future__ import annotations

import base64
import random
import re

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

# Codepoints that read like base64 to a human but sit outside the alphabet
LOOKALIKE = ["ＡＢＣＤ", "АВСЕ", "MTIzNА==", "⁰¹²³", "MTIzNA==​", "﻿MTIzNA==", "MTIz​NA=="]

# Surplus padding on real base64: b64decode(validate=True) accepts these on
# 3.11 and raises on 3.12, so a delegated rule answers per interpreter
EXCESS_PADDING = [
    "AAAA=",
    "AAAA==",
    "AAAA===",
    "AAAA====",
    "AAAA=====",
    "AAAA======",
    "MTIzNA====",
    "dGVzdA====",
    "/NTpk6v8HIk8U2RJ/JRrGsPlghKY=",
    "/NTpk6v8HIk8U2RJ/JRrGsPlghKY====",
    "YW5j-aG9y-IHBh-eWxv-YWQg-aGVy-ZQ==",
    "YW5j-aG9y-IHBh-eWxv-YWQg-aGVy-ZQ====",
]

# What GNU base64 and openssl base64 emit by default, refused because an
# embedded newline is the same laundering channel as "MTIz NA=="
MIME_WRAPPED = [
    base64.encodebytes(bytes(range(200))).decode(),
    base64.encodebytes(bytes(range(200))).decode().rstrip("\n"),
    base64.encodebytes(bytes(range(64))).decode(),
    "MTIzNA==\r\nMTIzNA==",
]


def _axis(value: object) -> tuple[str, str]:
    env = dict(ENVELOPE)
    env["anchors"] = [{"type": "rfc3161", "value": value}]
    return check_anchors(env)


    # The reported case: an all-punctuation value decoded to nothing and passed.
def test_reported_forged_punctuation_anchor_fails() -> None:
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


@pytest.mark.parametrize("value", LOOKALIKE, ids=repr)
def test_lookalike_codepoints_are_refused(value: str) -> None:
    assert _safe_b64(value) is False, repr(value)
    assert _axis(value)[0] == "FAIL", repr(value)


@pytest.mark.parametrize("value", EXCESS_PADDING, ids=repr)
def test_surplus_padding_is_refused_on_every_interpreter(value: str) -> None:
    """The verdict must come from the alphabet rule, not from b64decode's mood.

    base64.b64decode(b"AAAA====", validate=True) returns 3 bytes on 3.11 and
    raises on 3.12, and both are declared-supported targets.
    """
    assert _safe_b64(value) is False, repr(value)
    assert _axis(value)[0] == "FAIL", repr(value)


    # Documented behaviour change: an anchor value is one unwrapped token.
@pytest.mark.parametrize("value", MIME_WRAPPED, ids=repr)
def test_mime_line_wrapped_base64_is_refused(value: str) -> None:
    assert _safe_b64(value) is False, repr(value)
    assert _axis(value)[0] == "FAIL", repr(value)


    # Every encoding a real signer emits keeps passing, padded or not.
def test_legitimate_anchor_values_still_pass() -> None:
    rng = random.Random(358)
    for n in range(1, 129):
        raw = bytes(rng.randrange(256) for _ in range(n))
        std = base64.b64encode(raw).decode()
        url = base64.urlsafe_b64encode(raw).decode()
        for value in (std, std.rstrip("="), url, url.rstrip("=")):
            assert _safe_b64(value) is True, repr(value)
            assert _axis(value)[0] == "PASS", repr(value)


def test_verdict_matches_the_grammar_rule_not_a_decoder() -> None:
    """Pin the verdict to a rule no CPython release can move underneath it.

    The oracle here is the base64 grammar, deliberately not
    base64.b64decode(validate=True), whose strictness changed between 3.11
    and 3.12. _safe_b64 must also never raise on a hostile value.
    """
    grammar = re.compile(r"[A-Za-z0-9+/]+={0,2}")
    rng = random.Random(1358)
    alphabet = "".join(chr(c) for c in range(32, 127)) + "\xa0é中​﻿\U0001f600"
    for _ in range(4000):
        value = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 24)))
        padded = value.replace("-", "+").replace("_", "/")
        padded += "=" * ((-len(padded)) % 4)
        ok = bool(grammar.fullmatch(padded)) and (len(padded) // 4) * 3 - padded.count("=") > 0
        assert _safe_b64(value) is ok, repr(value)
