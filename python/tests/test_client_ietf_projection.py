"""SDK projection helpers for the three-key Compliance Receipts envelope.

`SignatureResponse.signature_envelope()` returns `{alg, kid, sig}` when
the response carries the object form, None otherwise. `anchors_array()`
returns the cloud-supplied `anchors` list (possibly empty), None on
non-compliance receipts.
"""

from __future__ import annotations

from asqav.client import SignatureResponse
from asqav.ietf_projection import (
    anchors_from_response,
    signature_envelope_from_response,
)


def test_flat_response_signature_envelope_is_none():
    resp = SignatureResponse(
        signature="ZmFrZQ==",
        signature_id="sig_test",
        action_id="act_test",
        timestamp=0.0,
        verification_url="https://verify/example",
    )
    assert resp.signature_envelope() is None


def test_compliance_response_signature_envelope_returns_object_form():
    envelope = {"alg": "ML-DSA-65", "kid": "key_abc", "sig": "ZmFrZQ=="}
    resp = SignatureResponse(
        signature=envelope,
        signature_id="sig_test",
        action_id="act_test",
        timestamp=0.0,
        verification_url="https://verify/example",
        compliance_mode=True,
    )
    out = resp.signature_envelope()
    assert out == envelope


def test_signature_envelope_helper_dict_input():
    envelope = {"alg": "Ed25519", "kid": "k1", "sig": "s"}
    out = signature_envelope_from_response(
        {"compliance_mode": True, "signature": envelope}
    )
    assert out == envelope


def test_signature_envelope_helper_string_returns_none():
    assert (
        signature_envelope_from_response({"signature": "ZmFrZQ=="})
        is None
    )


def test_anchors_array_passthrough():
    cloud_anchors = [
        {"type": "rfc3161", "value": "ZmFrZQ==", "commit_hash": "h" * 64}
    ]
    resp = SignatureResponse(
        signature={"alg": "ML-DSA-65", "kid": "k", "sig": "s"},
        signature_id="sig_test",
        action_id="act_test",
        timestamp=0.0,
        verification_url="https://verify/example",
        compliance_mode=True,
        anchors=cloud_anchors,
    )
    assert resp.anchors_array() == cloud_anchors


def test_anchors_array_string_signature_is_none():
    resp = SignatureResponse(
        signature="ZmFrZQ==",
        signature_id="sig_test",
        action_id="act_test",
        timestamp=0.0,
        verification_url="https://verify/example",
    )
    assert resp.anchors_array() is None


def test_anchors_helper_dict_input():
    out = anchors_from_response({"anchors": [{"type": "rfc3161"}]})
    assert out == [{"type": "rfc3161"}]
    assert anchors_from_response({}) is None
