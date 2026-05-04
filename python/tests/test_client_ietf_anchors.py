"""IETF -01 N7 / S3: SDK-side `anchors_array()` projection helper.

`anchors_array()` returns the spec-shape anchors list with `type`
discriminator per Section 4.4. Gated on `compliance_mode=True`; legacy
receipts get None back.
"""

from __future__ import annotations

from asqav.client import BitcoinAnchor, SignatureResponse
from asqav.ietf_projection import (
    anchors_from_response,
    encode_anchor_value,
)


def test_anchors_array_none_for_non_compliance_response():
    """Non-compliance receipts MUST get None back."""
    resp = SignatureResponse(
        signature="ZmFrZQ==",
        signature_id="sig_test",
        action_id="act_test",
        timestamp=0.0,
        verification_url="https://verify/example",
    )
    assert resp.anchors_array() is None


def test_anchors_array_returns_cloud_supplied_when_present():
    """When the cloud emits `anchors`, the SDK passes it through."""
    cloud_anchors = [
        {"type": "rfc3161", "value": "ZmFrZS1kZXI=", "tsa": "freetsa.org"},
        {"type": "opentimestamps", "value": "ZmFrZS1vdHM=", "status": "pending"},
    ]
    resp = SignatureResponse(
        signature="ZmFrZQ==",
        signature_id="sig_test",
        action_id="act_test",
        timestamp=0.0,
        verification_url="https://verify/example",
        compliance_mode=True,
        anchors=cloud_anchors,
    )
    out = resp.anchors_array()
    assert out == cloud_anchors
    # Returned list is a copy of each entry (mutations do not leak back).
    out[0]["tsa"] = "MUTATED"
    assert resp.anchors[0]["tsa"] == "freetsa.org"


def test_anchors_array_projects_rfc3161_from_flat_fields():
    """Older clouds without `anchors` get an RFC 3161 entry from the flats."""
    resp = SignatureResponse(
        signature="ZmFrZQ==",
        signature_id="sig_test",
        action_id="act_test",
        timestamp=0.0,
        verification_url="https://verify/example",
        rfc3161_serial="0xabc123",
        rfc3161_tsa="freetsa.org",
        compliance_mode=True,
    )
    out = resp.anchors_array()
    assert len(out) == 1
    assert out[0]["type"] == "rfc3161"
    assert out[0]["serial"] == "0xabc123"
    assert out[0]["tsa"] == "freetsa.org"


def test_anchors_array_projects_opentimestamps_from_bitcoin_anchor():
    """Older clouds with a BitcoinAnchor object get an OTS presence entry."""
    resp = SignatureResponse(
        signature="ZmFrZQ==",
        signature_id="sig_test",
        action_id="act_test",
        timestamp=0.0,
        verification_url="https://verify/example",
        bitcoin_anchor=BitcoinAnchor(status="pending"),
        compliance_mode=True,
    )
    out = resp.anchors_array()
    assert any(e["type"] == "opentimestamps" for e in out)


def test_anchors_array_empty_list_when_compliance_but_no_anchors():
    """compliance_mode=True with no anchors -> [] not None."""
    resp = SignatureResponse(
        signature="ZmFrZQ==",
        signature_id="sig_test",
        action_id="act_test",
        timestamp=0.0,
        verification_url="https://verify/example",
        compliance_mode=True,
    )
    out = resp.anchors_array()
    assert out == []


def test_anchors_helper_works_on_dict_input():
    """The free helper accepts both objects and dicts (verify response)."""
    cloud_anchors = [{"type": "rfc3161", "value": "ZmFrZQ=="}]
    out = anchors_from_response(
        {"compliance_mode": True, "anchors": cloud_anchors}
    )
    assert out == cloud_anchors


def test_anchors_helper_none_for_legacy_dict():
    """Dict input without compliance_mode returns None."""
    assert anchors_from_response({"rfc3161_serial": "0x1"}) is None


def test_encode_anchor_value_is_base64():
    """Helper for callers that hold raw anchor bytes locally."""
    assert encode_anchor_value(b"hello") == "aGVsbG8="
