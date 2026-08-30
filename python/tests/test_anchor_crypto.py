"""Cryptographic anchor verification (draft: presence without a successful
check MUST NOT yield valid).

Covers the three anchor outcomes:
  - verified:     token commits this envelope and its own proof checks out
                  against caller-pinned trust material (trusted anchor)
  - invalid:      the check ran and failed (wrong digest, bad TSA signature,
                  merkle path missing the stated block)
  - unverifiable: the check could not run offline (junk token, no pinned TSA
                  key, no bitcoin header source, pending/failed status)
and the key_status wiring: only a verified anchor whose proven time lands at
or before revoked_at activates the pre-revocation historical-verify path.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from asqav.verifier import verify_receipt as v
from tests.tsa_testkit import make_timestamp_resp, make_tst_info, mint_ml_dsa_anchor

FIXTURES = Path(__file__).resolve().parents[2] / "verifier" / "docs" / "fixtures"


def _envelope() -> dict:
    return {
        "payload": {
            "type": "protectmcp:decision",
            "issued_at": "2026-06-01T00:00:00Z",
            "issuer_id": "org-anchor",
            "action_ref": "sha256:" + "a" * 64,
            "payload_digest": {"hash": "b" * 64, "size": 128},
            "policy_digest": "sha256:" + "c" * 64,
            "previousReceiptHash": "0" * 64,
            "decision": "allow",
        },
        "signature": {"alg": "ML-DSA-65", "kid": "org-anchor", "sig": "AAAA"},
    }


def _bound(env: dict) -> bytes:
    return hashlib.sha256(v.envelope_minus_anchors_jcs(env)).digest()


def _anchored(env: dict, anchor: dict) -> dict:
    out = dict(env)
    out["anchors"] = [anchor]
    return out


# --- RFC 3161 ---------------------------------------------------------------


def test_rfc3161_verified_against_pinned_tsa_key() -> None:
    pytest.importorskip("dilithium_py")
    env = _envelope()
    tok, pk = mint_ml_dsa_anchor(_bound(env))
    ev = v.evaluate_anchors(
        _anchored(env, {"type": "rfc3161", "value": tok}), trusted_tsa_keys=[pk]
    )
    assert ev.result == "PASS", ev.note
    assert ev.trusted_times and ev.trusted_times[0].isoformat().startswith("2026-06-01")


def test_rfc3161_without_pinned_key_is_unverifiable_never_pass() -> None:
    pytest.importorskip("dilithium_py")
    env = _envelope()
    tok, _pk = mint_ml_dsa_anchor(_bound(env))
    res, note = v.check_anchors(_anchored(env, {"type": "rfc3161", "value": tok}))
    assert res == "SKIPPED", note
    assert "imprint matches" in note
    assert v.evaluate_anchors(_anchored(env, {"type": "rfc3161", "value": tok})).trusted_times == []


def test_rfc3161_wrong_tsa_key_is_invalid() -> None:
    pytest.importorskip("dilithium_py")
    from dilithium_py.ml_dsa import ML_DSA_65

    env = _envelope()
    tok, _pk = mint_ml_dsa_anchor(_bound(env))
    other_pk, _sk = ML_DSA_65.keygen()
    res, note = v.check_anchors(
        _anchored(env, {"type": "rfc3161", "value": tok}), trusted_tsa_keys=[other_pk]
    )
    assert res == "FAIL", note
    assert "TSA signature" in note


def test_rfc3161_imprint_mismatch_is_invalid() -> None:
    pytest.importorskip("dilithium_py")
    env = _envelope()
    tok, pk = mint_ml_dsa_anchor(b"\x00" * 32)  # commits some other digest
    res, note = v.check_anchors(
        _anchored(env, {"type": "rfc3161", "value": tok}), trusted_tsa_keys=[pk]
    )
    assert res == "FAIL", note
    assert "different digest" in note


def test_rfc3161_rejection_status_is_invalid() -> None:
    env = _envelope()
    tst = make_tst_info(_bound(env))
    token = base64.b64encode(make_timestamp_resp(tst, b"\x00" * 16, status=2)).decode()
    res, note = v.check_anchors(_anchored(env, {"type": "rfc3161", "value": token}))
    assert res == "FAIL", note
    assert "status 2" in note


def test_rfc3161_junk_token_is_unverifiable_not_invalid() -> None:
    env = _envelope()
    res, note = v.check_anchors(_anchored(env, {"type": "rfc3161", "value": "dGVzdA=="}))
    assert res == "SKIPPED", note
    assert "offline RFC3161 check did not complete" in note


def test_rsa_tsa_cert_allowlist_verifies() -> None:
    """The cryptography path: TSA cert pinned by the caller, sid issuerAndSerial."""
    pytest.importorskip("cryptography")
    import datetime as dt

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.x509.oid import NameOID

    from tests.tsa_testkit import _int, _seq

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-tsa")])
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(7)
        .not_valid_before(now)
        .not_valid_after(now + dt.timedelta(days=3650))
        .sign(key, hashes.SHA256())
    )
    env = _envelope()
    tst = make_tst_info(_bound(env))
    signature = key.sign(tst, padding.PKCS1v15(), hashes.SHA256())
    sid = _seq(name.public_bytes(), _int(7))  # issuerAndSerialNumber
    token = base64.b64encode(
        make_timestamp_resp(
            tst,
            signature,
            sig_alg_oid="1.2.840.113549.1.1.11",
            sid=sid,
            certs=[cert.public_bytes(serialization.Encoding.DER)],
        )
    ).decode()
    der = cert.public_bytes(serialization.Encoding.DER)
    res, note = v.check_anchors(
        _anchored(env, {"type": "rfc3161", "value": token}), trusted_tsa_keys=[der]
    )
    assert res == "PASS", note


def test_published_fixture_imprint_matches_but_tsa_untrusted() -> None:
    """Parser check against the real production token: imprint must match."""
    env = json.loads((FIXTURES / "published-receipt.json").read_text())
    res, note = v.check_anchors(env)
    assert res == "SKIPPED", note
    assert "imprint matches" in note, note
    assert "status pending" in note, note


# --- OpenTimestamps ----------------------------------------------------------


def _varuint(n: int) -> bytes:
    out = bytearray([n & 0x7F])
    n >>= 7
    while n:
        out.insert(0, 0x80 | (n & 0x7F))
        n >>= 7
    return bytes(out)


def _mint_ots(digest32: bytes, height: int) -> tuple[str, bytes]:
    """A proof digest->append->sha256->bitcoin attestation; returns (b64, root)."""
    suffix = b"merkle-sibling"
    root = hashlib.sha256(digest32 + suffix).digest()
    body = (
        bytes([0xF0, len(suffix)])
        + suffix
        + b"\x08"  # OpSHA256
        + b"\x00"  # attestation marker
        + bytes.fromhex("0588960d73d71901")
        + _varuint(height)
    )
    blob = v._OTS_MAGIC + b"\x01\x08" + digest32 + body
    return base64.b64encode(blob).decode(), root


def test_ots_commitment_mismatch_is_invalid() -> None:
    env = _envelope()
    proof, _root = _mint_ots(b"\x11" * 32, 900000)
    res, note = v.check_anchors(_anchored(env, {"type": "opentimestamps", "value": proof}))
    assert res == "FAIL", note
    assert "different digest" in note


def test_ots_without_header_source_is_unverifiable() -> None:
    env = _envelope()
    proof, _root = _mint_ots(_bound(env), 900000)
    res, note = v.check_anchors(_anchored(env, {"type": "opentimestamps", "value": proof}))
    assert res == "SKIPPED", note
    assert "commitment matches" in note


def test_ots_merkle_root_mismatch_is_invalid() -> None:
    env = _envelope()
    proof, _root = _mint_ots(_bound(env), 900000)
    headers = {"900000": {"merkle_root": "00" * 32, "time": "2026-06-01T00:00:00Z"}}
    res, note = v.check_anchors(
        _anchored(env, {"type": "opentimestamps", "value": proof}), bitcoin_headers=headers
    )
    assert res == "FAIL", note
    assert "does not land" in note


def test_ots_verified_against_supplied_header() -> None:
    env = _envelope()
    proof, root = _mint_ots(_bound(env), 900000)
    headers = {900000: {"merkle_root": root[::-1].hex(), "time": "2026-06-01T12:00:00Z"}}
    ev = v.evaluate_anchors(
        _anchored(env, {"type": "opentimestamps", "value": proof}), bitcoin_headers=headers
    )
    assert ev.result == "PASS", ev.note
    assert ev.trusted_times and ev.trusted_times[0].isoformat().startswith("2026-06-01T12:00")


# --- status / shape rules ----------------------------------------------------


@pytest.mark.parametrize("status", ["pending", "failed"])
def test_declared_pending_or_failed_status_is_never_trusted(status: str) -> None:
    pytest.importorskip("dilithium_py")
    env = _envelope()
    tok, pk = mint_ml_dsa_anchor(_bound(env))
    anchor = {"type": "rfc3161", "value": tok, "status": status}
    ev = v.evaluate_anchors(_anchored(env, anchor), trusted_tsa_keys=[pk])
    assert ev.result == "SKIPPED", ev.note
    assert ev.trusted_times == []
    assert f"status {status}" in ev.note


def test_unknown_anchor_type_is_unverifiable() -> None:
    env = _envelope()
    res, note = v.check_anchors(_anchored(env, {"type": "merkle", "value": "YWJjZA=="}))
    assert res == "SKIPPED", note
    assert "no offline verifier for this anchor type" in note


def test_bad_base64_still_fails_invalid() -> None:
    env = _envelope()
    res, note = v.check_anchors(_anchored(env, {"type": "rfc3161", "value": "!!!!"}))
    assert res == "FAIL", note
    assert "MISSING or malformed" in note


def test_one_invalid_anchor_dominates_a_verified_sibling() -> None:
    pytest.importorskip("dilithium_py")
    env = _envelope()
    tok, pk = mint_ml_dsa_anchor(_bound(env))
    out = dict(env)
    out["anchors"] = [
        {"type": "rfc3161", "value": tok},
        {"type": "rfc3161", "value": "!!!!"},
    ]
    res, _note = v.check_anchors(out, trusted_tsa_keys=[pk])
    assert res == "FAIL"


# --- key_status wiring (item 3) ----------------------------------------------


def _signed_revoked_env(gen_time: str):
    from dilithium_py.ml_dsa import ML_DSA_65

    pk, sk = ML_DSA_65.keygen()
    env = _envelope()
    # The anchor commits payload+signature, so sign first, then anchor the
    # digest of the envelope as it will ship.
    env["signature"] = {
        "alg": "ML-DSA-65",
        "kid": "org-anchor",
        "sig": base64.b64encode(ML_DSA_65.sign(sk, v.canonical_json(env["payload"]))).decode(),
    }
    tok, tsa_pk = mint_ml_dsa_anchor(_bound(env), gen_time=gen_time)
    env["anchors"] = [{"type": "rfc3161", "value": tok}]
    jwks = {
        "keys": [
            {
                "kid": "org-anchor",
                "issuer_id": "org-anchor",
                "alg": "ML-DSA-65",
                "public_key": base64.b64encode(pk).decode(),
                "status": "revoked",
                "revoked_at": "2026-07-01T00:00:00Z",
            }
        ]
    }
    return env, jwks, tsa_pk


def test_pre_revocation_receipt_passes_with_verified_anchor() -> None:
    """Draft historical verify: signed before revocation, proven by the anchor."""
    pytest.importorskip("dilithium_py")
    env, jwks, tsa_pk = _signed_revoked_env("20260601000000Z")  # before revoked_at
    result = v.run_structured(env, jwks, None, trusted_tsa_keys=[tsa_pk])
    ks = next(a for a in result["axes"] if a["name"] == "key_status")
    assert ks["result"] == "PASS", ks
    assert result["verdict"] == "verified", result["axes"]


def test_anchor_after_revocation_does_not_prove_pre_revocation_timing() -> None:
    pytest.importorskip("dilithium_py")
    env, jwks, tsa_pk = _signed_revoked_env("20260801000000Z")  # after revoked_at
    result = v.run_structured(env, jwks, None, trusted_tsa_keys=[tsa_pk])
    ks = next(a for a in result["axes"] if a["name"] == "key_status")
    assert ks["result"] == "SKIPPED", ks
    assert result["verdict"] != "verified"


def test_unverified_anchor_never_upgrades_a_revoked_key() -> None:
    pytest.importorskip("dilithium_py")
    env, jwks, _tsa_pk = _signed_revoked_env("20260601000000Z")
    # No pinned TSA key: the anchor cannot be trusted, so key_status stays SKIPPED.
    result = v.run_structured(env, jwks, None)
    ks = next(a for a in result["axes"] if a["name"] == "key_status")
    assert ks["result"] == "SKIPPED", ks
    assert result["verdict"] != "verified"
