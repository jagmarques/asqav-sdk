"""The anchors axis hashes the two-key {payload, signature} object the signer committed.

Pinned against a REAL production receipt exported through the Audit Pack on 2026-09-02:
the export carries thirteen top-level members and re-encodes the signature string in
standard base64, while the signer anchored the base64url wire string over exactly
{payload, signature}. Before this fix the standalone verifier reported the untouched
receipt as invalid.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from asqav.verifier import verify_receipt as vr

FIXTURES = Path(__file__).parent / "fixtures"
RECEIPT = json.loads((FIXTURES / "prod_audit_pack_receipt_2026_09_02.json").read_text())
JWKS = json.loads((FIXTURES / "prod_audit_pack_jwks_2026_09_02.json").read_text())


def test_the_export_carries_more_members_than_the_signer_anchored() -> None:
    assert len(RECEIPT) > 3
    assert {"payload", "signature", "anchors"} <= set(RECEIPT)


def test_normalise_envelope_keeps_exactly_the_three_canonical_members() -> None:
    env = vr.normalise_envelope(RECEIPT)
    assert set(env) == {"payload", "signature", "anchors"}
    assert env["payload"] is RECEIPT["payload"]
    assert env["signature"] is RECEIPT["signature"]


def test_the_anchored_bytes_are_the_two_key_object_only() -> None:
    two_key = vr.canonical_json({"payload": RECEIPT["payload"], "signature": RECEIPT["signature"]})
    assert vr.envelope_minus_anchors_jcs(RECEIPT) == two_key
    assert vr.envelope_minus_anchors_jcs(vr.normalise_envelope(RECEIPT)) == two_key


def test_the_rfc3161_imprint_of_a_real_receipt_is_never_reported_invalid() -> None:
    """The gate for the false accusation: an untouched production receipt must not FAIL."""
    ev = vr.evaluate_anchors(vr.normalise_envelope(RECEIPT))
    assert ev.result != "FAIL", ev.note
    assert "different digest" not in ev.note, ev.note
    assert "rfc3161" in ev.note


def test_the_signature_axis_still_verifies_the_real_receipt() -> None:
    pytest.importorskip("dilithium_py")
    report = vr.run_structured(vr.normalise_envelope(RECEIPT), JWKS)
    axes = {a["name"]: a for a in report["axes"]} if isinstance(report, dict) else {}
    assert axes, report
    assert axes["signature"]["result"] == "PASS", axes["signature"]
    assert axes["anchors"]["result"] != "FAIL", axes["anchors"]


def test_a_standard_base64_export_of_a_base64url_commitment_is_recognised() -> None:
    """The migration-window tolerance: the commitment was made over the base64url string."""
    sig_std = RECEIPT["signature"]["sig"]
    raw = base64.b64decode(sig_std + "=" * (-len(sig_std) % 4))
    sig_url = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    assert sig_url != sig_std, "fixture must contain alphabet-sensitive bytes"
    twin = vr._sig_alphabet_twin({"payload": RECEIPT["payload"], "signature": RECEIPT["signature"]})
    assert twin is not None and twin["signature"]["sig"] == sig_url
    committed = hashlib.sha256(vr.envelope_minus_anchors_jcs(twin)).digest()
    exported = hashlib.sha256(vr.envelope_minus_anchors_jcs(RECEIPT)).digest()
    assert committed != exported
    ev = vr.evaluate_anchors(vr.normalise_envelope(RECEIPT))
    assert "re-encoded to the alphabet the signer committed" in ev.note, ev.note


def test_a_tampered_payload_is_still_a_proven_mismatch() -> None:
    """The tolerance must not launder a real tamper: a changed payload fails both alphabets."""
    tampered = json.loads(json.dumps(RECEIPT))
    tampered["payload"]["seq"] = (tampered["payload"].get("seq") or 0) + 1
    ev = vr.evaluate_anchors(vr.normalise_envelope(tampered))
    assert ev.result == "FAIL"
    assert "different digest" in ev.note
