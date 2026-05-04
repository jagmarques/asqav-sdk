"""IETF -01 N6 / S2: SDK-side `signature_object()` projection helper.

`signature_object()` returns `{alg, kid, sig}` per Section 4
"JSON Receipt Structure". Gated on `compliance_mode=True`; legacy
receipts get None back so the additive overlay stays byte-stable.
"""

from __future__ import annotations

from asqav.client import SignatureResponse
from asqav.ietf_projection import signature_object_from_response


def test_signature_object_none_for_non_compliance_response():
    """Non-compliance receipts MUST get None back."""
    resp = SignatureResponse(
        signature="ZmFrZQ==",
        signature_id="sig_test",
        action_id="act_test",
        timestamp=0.0,
        verification_url="https://verify/example",
    )
    assert resp.signature_object() is None


def test_signature_object_returns_cloud_supplied_when_present():
    """When the cloud emits `signatureObject`, the SDK passes it through."""
    envelope = {"alg": "ML-DSA-65", "kid": "key_abc", "sig": "ZmFrZQ=="}
    resp = SignatureResponse(
        signature="ZmFrZQ==",
        signature_id="sig_test",
        action_id="act_test",
        timestamp=0.0,
        verification_url="https://verify/example",
        compliance_mode=True,
        signatureObject=envelope,
    )
    out = resp.signature_object()
    assert out == envelope
    # Returned dict is a copy (callers cannot mutate the response).
    out["sig"] = "MUTATED"
    assert resp.signatureObject["sig"] == "ZmFrZQ=="


def test_signature_object_projects_from_flat_fields_under_compliance():
    """Older clouds without `signatureObject` get projected from flats."""
    resp = SignatureResponse(
        signature="ZmFrZQ==",
        signature_id="sig_test",
        action_id="act_test",
        timestamp=0.0,
        verification_url="https://verify/example",
        algorithm="ML-DSA-65",
        compliance_mode=True,
    )
    out = resp.signature_object()
    assert set(out.keys()) == {"alg", "kid", "sig"}
    assert out["alg"] == "ML-DSA-65"
    # kid not exposed on the dataclass yet -> empty string.
    assert out["kid"] == ""
    assert out["sig"] == "ZmFrZQ=="


def test_signature_object_helper_works_on_dict_input():
    """The free helper accepts both objects and dicts (verify response)."""
    envelope = {"alg": "ML-DSA-65", "kid": "key_abc", "sig": "ZmFrZQ=="}
    out = signature_object_from_response(
        {"compliance_mode": True, "signatureObject": envelope}
    )
    assert out == envelope


def test_signature_object_helper_none_for_legacy_dict():
    """Dict input without compliance_mode returns None."""
    assert signature_object_from_response({"algorithm": "ML-DSA-65"}) is None
