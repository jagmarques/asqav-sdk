"""Anchor-verification hardening: malformed input never crashes or over-claims.

`anchors` sits outside the signed bytes, so any relay can steer these values.
Two invariants: no input escapes as an unhandled exception (the CLI maps one to
exit 1, indistinguishable from a proven binding failure), and no input reaches
PASS without a real cryptographic check.
"""

from __future__ import annotations

import base64
import hashlib
import time

import pytest

from asqav.verifier import verify_receipt as vr

from .tsa_testkit import (  # type: ignore[import-not-found]
    _len,
    _oid,
    _seq,
    _tlv,
    make_signed_attrs,
    make_timestamp_resp,
    make_tst_info,
    signed_attrs_signing_input,
)


def _envelope(anchors: list[dict]) -> dict:
    return {
        "payload": {"type": "protectmcp:decision", "issuer_id": "x", "decision": "allow"},
        "signature": {"alg": "ML-DSA-65", "kid": "x", "sig": "AA=="},
        "anchors": anchors,
    }


def _bound_for(envelope: dict, algo: str = "sha256") -> bytes:
    """The digest an anchor must commit: H(JCS(envelope minus anchors))."""
    return hashlib.new(algo, vr.envelope_minus_anchors_jcs(envelope)).digest()


def _anchor(atype: str, blob: bytes) -> dict:
    return {"type": atype, "value": base64.b64encode(blob).decode()}


# --- 1. Nothing escapes as a crash -----------------------------------------


def test_giant_oid_arc_reports_unverifiable_not_crash():
    """A multi-thousand-byte OID arc must not raise the interpreter's
    int-to-str ValueError out of the anchors axis."""
    giant = _tlv(0x06, b"\x2a" + b"\xff" * 2100 + b"\x01")
    poison = _seq(_seq(_tlv(0x02, b"\x00")), _seq(giant))
    ev = vr.evaluate_anchors(_envelope([_anchor("rfc3161", poison)]))
    assert ev.result == "SKIPPED"
    assert ev.trusted_times == []


def test_giant_oid_arc_through_run_exits_unverifiable(capsys):
    """Exit 2 (unverifiable), never 1 (invalid) - an uncaught exception in the
    DER walk would also exit 1, collapsing the distinction."""
    giant = _tlv(0x06, b"\x2a" + b"\xff" * 2100 + b"\x01")
    poison = _seq(_seq(_tlv(0x02, b"\x00")), _seq(giant))
    code = vr.run(_envelope([_anchor("rfc3161", poison)]), {"keys": []}, None)
    assert code == 2, "a parse failure must not be reported as `invalid`"


def test_giant_oid_arc_structured_is_unverifiable():
    giant = _tlv(0x06, b"\x2a" + b"\xff" * 2100 + b"\x01")
    poison = _seq(_seq(_tlv(0x02, b"\x00")), _seq(giant))
    res = vr.run_structured(_envelope([_anchor("rfc3161", poison)]), {"keys": []})
    assert res["verdict"] == vr.VERDICT_UNVERIFIED
    assert res["failure_class"] == vr.FAILURE_UNVERIFIABLE


def test_truncated_signed_attrs_with_matching_imprint_is_unverifiable():
    """The imprint check passes, then the signedAttrs walk hits truncated DER.
    That walk sits after the imprint comparison, so it must be inside the
    same parse guard."""
    env = _envelope([])
    bound = _bound_for(env)
    tst = make_tst_info(bound)
    attrs = make_signed_attrs(tst)
    token = make_timestamp_resp(tst, b"\x00" * 64, signed_attrs=attrs[:-4])
    ev = vr.evaluate_anchors(_envelope([_anchor("rfc3161", token)]))
    assert ev.result in ("SKIPPED", "FAIL")
    assert ev.trusted_times == []


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        b"\x30",
        b"\x30\x80" + b"\x00" * 8,  # indefinite length
        b"\x30\x84\xff\xff\xff\xff",  # length overruns the buffer
        b"\x30\x03" + b"\x00" * 32,  # trailing garbage
        b"\x3f\x1f\x01\x02",  # high-tag-number form
        b"\x30" + b"\x30" * 5000,  # deep-ish nesting attempt
    ],
)
def test_malformed_der_never_raises(blob):
    ev = vr.evaluate_anchors(_envelope([_anchor("rfc3161", blob)]))
    assert ev.result in ("SKIPPED", "FAIL")
    assert ev.trusted_times == []


def test_deeply_nested_der_does_not_recurse():
    """The DER walk is iterative; 10k nested SEQUENCEs must not blow the stack."""
    blob = b""
    for _ in range(10000):
        blob = b"\x30" + _len(len(blob)) + blob
    ev = vr.evaluate_anchors(_envelope([_anchor("rfc3161", blob)]))
    assert ev.result in ("SKIPPED", "FAIL")


# --- 2. No quadratic blow-up on caller-supplied bytes -----------------------


def test_ots_varuint_is_width_bounded():
    """An unbounded varuint builds a multi-megabit int one shift at a time;
    900k continuation bytes measured at 326 seconds before the cap."""
    blob = vr._OTS_MAGIC + bytes([1, 0x08]) + b"\x00" * 32 + b"\x00" + b"\xff" * 900_000
    start = time.monotonic()
    ev = vr.evaluate_anchors(_envelope([_anchor("opentimestamps", blob)]))
    elapsed = time.monotonic() - start
    assert ev.result in ("SKIPPED", "FAIL")
    assert elapsed < 5.0, f"ots varuint took {elapsed:.1f}s; the width cap is gone"


# --- 3. CMS content binding (RFC 5652 s11.1) -------------------------------


def _ml_dsa_token(tst: bytes, *, signed_attrs: bytes | None = None, **kw):
    """Sign whatever the CMS rules say is covered, with a real ML-DSA key."""
    from dilithium_py.ml_dsa import ML_DSA_65

    pk, sk = ML_DSA_65.keygen()
    covered = tst if signed_attrs is None else signed_attrs_signing_input(signed_attrs)
    token = make_timestamp_resp(
        tst, ML_DSA_65.sign(sk, covered), signed_attrs=signed_attrs, **kw
    )
    return token, pk


def test_signed_attrs_token_with_full_binding_verifies():
    """Control case: a well-formed signedAttrs token DOES verify, so the
    rejection tests below are not passing for the wrong reason."""
    env = _envelope([])
    tst = make_tst_info(_bound_for(env))
    attrs = make_signed_attrs(tst)
    token, pk = _ml_dsa_token(tst, signed_attrs=attrs)
    ev = vr.evaluate_anchors(
        _envelope([_anchor("rfc3161", token)]), trusted_tsa_keys=[pk]
    )
    assert ev.result == "PASS"
    assert len(ev.trusted_times) == 1


def test_signed_attrs_without_content_type_is_rejected():
    """contentType is mandatory when signedAttrs is present; without it a TSA
    signature over other content can be replayed as a timestamp."""
    env = _envelope([])
    tst = make_tst_info(_bound_for(env))
    attrs = make_signed_attrs(tst, include_content_type=False)
    token, pk = _ml_dsa_token(tst, signed_attrs=attrs)
    ev = vr.evaluate_anchors(
        _envelope([_anchor("rfc3161", token)]), trusted_tsa_keys=[pk]
    )
    assert ev.result != "PASS"
    assert ev.trusted_times == []


def test_signed_attrs_with_wrong_content_type_is_rejected():
    env = _envelope([])
    tst = make_tst_info(_bound_for(env))
    attrs = make_signed_attrs(tst, content_type_oid="1.2.840.113549.1.7.1")  # id-data
    token, pk = _ml_dsa_token(tst, signed_attrs=attrs)
    ev = vr.evaluate_anchors(
        _envelope([_anchor("rfc3161", token)]), trusted_tsa_keys=[pk]
    )
    assert ev.result != "PASS"
    assert ev.trusted_times == []


def test_wrong_econtent_type_is_rejected():
    """The encapsulated content must be declared id-ct-TSTInfo."""
    env = _envelope([])
    tst = make_tst_info(_bound_for(env))
    token, pk = _ml_dsa_token(tst, econtent_type_oid="1.2.840.113549.1.7.1")
    ev = vr.evaluate_anchors(
        _envelope([_anchor("rfc3161", token)]), trusted_tsa_keys=[pk]
    )
    assert ev.result != "PASS"
    assert ev.trusted_times == []


def test_message_digest_not_committing_tst_is_invalid():
    env = _envelope([])
    tst = make_tst_info(_bound_for(env))
    attrs = make_signed_attrs(tst, message_digest=b"\x11" * 32)
    token, pk = _ml_dsa_token(tst, signed_attrs=attrs)
    ev = vr.evaluate_anchors(
        _envelope([_anchor("rfc3161", token)]), trusted_tsa_keys=[pk]
    )
    assert ev.result == "FAIL"


# --- 4. Imprint algorithm honesty ------------------------------------------


def test_sha512_imprint_committing_this_envelope_verifies():
    """A sha512 messageImprint the verifier never computed is not a proven
    mismatch; recompute under the token's own algorithm instead of failing."""
    env = _envelope([])
    tst = _seq(
        _tlv(0x02, b"\x01"),
        _oid("1.2.3.4"),
        _seq(
            _seq(_oid("2.16.840.1.101.3.4.2.3"), _tlv(0x05, b"")),  # sha512
            _tlv(0x04, _bound_for(env, "sha512")),
        ),
        _tlv(0x02, b"\x2a"),
        _tlv(0x18, b"20260601000000Z"),
    )
    attrs = make_signed_attrs(tst)
    token, pk = _ml_dsa_token(tst, signed_attrs=attrs)
    ev = vr.evaluate_anchors(
        _envelope([_anchor("rfc3161", token)]), trusted_tsa_keys=[pk]
    )
    assert ev.result == "PASS", ev.note


def test_sha512_imprint_of_other_bytes_is_invalid():
    tst = _seq(
        _tlv(0x02, b"\x01"),
        _oid("1.2.3.4"),
        _seq(
            _seq(_oid("2.16.840.1.101.3.4.2.3"), _tlv(0x05, b"")),
            _tlv(0x04, hashlib.sha512(b"some other bytes").digest()),
        ),
        _tlv(0x02, b"\x2a"),
        _tlv(0x18, b"20260601000000Z"),
    )
    attrs = make_signed_attrs(tst)
    token, pk = _ml_dsa_token(tst, signed_attrs=attrs)
    ev = vr.evaluate_anchors(
        _envelope([_anchor("rfc3161", token)]), trusted_tsa_keys=[pk]
    )
    assert ev.result == "FAIL"


def test_unknown_imprint_oid_is_unverifiable_not_invalid():
    env = _envelope([])
    tst = _seq(
        _tlv(0x02, b"\x01"),
        _oid("1.2.3.4"),
        _seq(_seq(_oid("1.9.9.9.9"), _tlv(0x05, b"")), _tlv(0x04, _bound_for(env))),
        _tlv(0x02, b"\x2a"),
        _tlv(0x18, b"20260601000000Z"),
    )
    token, pk = _ml_dsa_token(tst)
    ev = vr.evaluate_anchors(
        _envelope([_anchor("rfc3161", token)]), trusted_tsa_keys=[pk]
    )
    assert ev.result == "SKIPPED", ev.note


# --- 5. Keyed digests never report plain `verified` ------------------------


@pytest.mark.parametrize(
    "algo,expected",
    [
        ("sha256", False),
        (None, False),
        ("hmac-sha256", True),
        ("HMAC-SHA256", True),  # near-miss spelling is still keyed
        ("hmac_sha256", True),
        ("hmac-sha512", True),
        ("blake3", True),  # unknown label: assume not re-derivable
        (12345, True),  # non-string: assume not re-derivable
    ],
)
def test_keyed_digest_detection(algo, expected):
    payload = {} if algo is None else {"hash_algo": algo}
    assert vr.is_keyed_digest(payload) is expected


def test_fold_verdict_keyed_never_collapses_to_verified():
    passing = [("signature", "PASS", ""), ("chain", "PASS", "")]
    assert vr._fold_verdict(passing, keyed=False)[0] == vr.VERDICT_VERIFIED
    assert vr._fold_verdict(passing, keyed=True)[0] == vr.VERDICT_VERIFIED_KEYED


def test_keyed_receipt_failing_a_check_stays_unverified():
    """`verified_keyed` is a PASSING verdict; it must not mask a real failure."""
    failing = [("signature", "FAIL", "bad signature")]
    verdict, fc = vr._fold_verdict(failing, keyed=True)
    assert verdict == vr.VERDICT_UNVERIFIED
    assert fc == vr.FAILURE_INVALID
