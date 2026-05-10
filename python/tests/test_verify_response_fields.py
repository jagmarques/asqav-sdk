"""SDK parser surfaces every field the cloud `/verify` returns.

The cloud's `VerificationResponse` (in `asqav_cloud/api/routes/verify.py`)
emits the IETF projection (`signature_envelope`, `anchors`,
`algorithm_registry_version`), the Bitcoin anchor block, the receipt
`type` discriminator, and four extra `VerificationDetail` sub-axes
(`anchor_valid_ots`, `anchor_valid_rfc3161`, `policy_digest_resolved`,
`duplicate_emission_candidate`). These tests pin that the SDK parser
populates the dataclasses with all of them.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import (
    BitcoinAnchorStatus,
    VerificationDetail,
    VerificationResponse,
    verify_signature,
)


def _full_cloud_payload() -> dict:
    """Mirror the cloud's `VerificationResponse` shape with every field set."""
    return {
        "type": "signature",
        "signature_id": "sig_test_full",
        "agent_id": "agent_full",
        "agent_name": "full-agent",
        "action_id": "act_full",
        "action_type": "payment.wire_transfer",
        "payload": {"amount": 100},
        "signature": "ZmFrZQ==",
        "algorithm": "ML-DSA-65",
        "signed_at": "2026-05-10T12:00:00+00:00",
        "verified": True,
        "verification_url": "https://verify.example/sig_test_full",
        "bitcoin_anchor": {
            "status": "confirmed",
            "bitcoin_tx": "abc123",
            "bitcoin_block": 850000,
        },
        "verification_detail": {
            "signer_key_match": True,
            "signature_valid": True,
            "algorithm_match": True,
            "agent_active": True,
            "validation_label": "valid",
            "signed_at_skew_seconds": 1.5,
            "chain_valid": True,
            "anchor_valid_ots": True,
            "anchor_valid_rfc3161": True,
            "anchor_status_ots": "valid",
            "anchor_status_rfc3161": "valid",
            "missing_fields": [],
            "policy_digest_resolved": True,
            "duplicate_emission_candidate": False,
        },
        "signature_envelope": {"alg": "ML-DSA-65", "kid": "key_full"},
        "anchors": [
            {"type": "rfc3161", "value": "ZmFrZQ=="},
            {"type": "ots", "value": "b3RzZmFrZQ=="},
        ],
        "algorithm_registry_version": "2026-05",
    }


def _mock_httpx(payload: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    return response


def test_verification_response_exposes_ietf_projection_fields() -> None:
    """`type`, `bitcoin_anchor`, `signature_envelope`, `anchors`,
    `algorithm_registry_version` are accessible on the parsed response."""
    payload = _full_cloud_payload()
    with patch("asqav.client.httpx.get", return_value=_mock_httpx(payload)):
        response = verify_signature("sig_test_full")

    assert isinstance(response, VerificationResponse)
    assert response.type == "signature"
    assert isinstance(response.bitcoin_anchor, BitcoinAnchorStatus)
    assert response.bitcoin_anchor.status == "confirmed"
    assert response.bitcoin_anchor.bitcoin_tx == "abc123"
    assert response.bitcoin_anchor.bitcoin_block == 850000
    assert response.signature_envelope == {"alg": "ML-DSA-65", "kid": "key_full"}
    assert response.anchors is not None
    assert len(response.anchors) == 2
    assert response.anchors[0]["type"] == "rfc3161"
    assert response.algorithm_registry_version == "2026-05"


def test_verification_detail_exposes_extra_sub_axes() -> None:
    """`anchor_valid_ots`, `anchor_valid_rfc3161`, `policy_digest_resolved`,
    `duplicate_emission_candidate` populate from the parser."""
    payload = _full_cloud_payload()
    with patch("asqav.client.httpx.get", return_value=_mock_httpx(payload)):
        response = verify_signature("sig_test_full")

    detail = response.verification_detail
    assert isinstance(detail, VerificationDetail)
    assert detail.anchor_valid_ots is True
    assert detail.anchor_valid_rfc3161 is True
    assert detail.policy_digest_resolved is True
    assert detail.duplicate_emission_candidate is False


def test_minimal_response_leaves_new_fields_none() -> None:
    """A response that omits the new fields parses without crashing and
    leaves them None for back-compat."""
    payload = {
        "signature_id": "sig_minimal",
        "agent_id": "agent_minimal",
        "agent_name": "minimal",
        "action_id": "act_minimal",
        "action_type": "x.y",
        "payload": None,
        "signature": "ZmFrZQ==",
        "algorithm": "Ed25519",
        "signed_at": "2026-05-10T12:00:00+00:00",
        "verified": True,
        "verification_url": "https://verify.example/sig_minimal",
    }
    with patch("asqav.client.httpx.get", return_value=_mock_httpx(payload)):
        response = verify_signature("sig_minimal")

    assert response.type == "signature"
    assert response.bitcoin_anchor is None
    assert response.signature_envelope is None
    assert response.anchors is None
    assert response.algorithm_registry_version is None
    assert response.verification_detail is None


def test_dataclass_defaults_construct_without_new_fields() -> None:
    """`VerificationResponse` and `VerificationDetail` keep all new
    fields optional so call sites that hand-build them keep working."""
    detail = VerificationDetail(
        signer_key_match=True,
        signature_valid=True,
        algorithm_match=True,
        agent_active=True,
        validation_label="valid",
    )
    assert detail.anchor_valid_ots is None
    assert detail.anchor_valid_rfc3161 is None
    assert detail.policy_digest_resolved is None
    assert detail.duplicate_emission_candidate is None

    response = VerificationResponse(
        signature_id="sig",
        agent_id="agent",
        agent_name="name",
        action_id="act",
        action_type="x",
        payload=None,
        signature="ZmFrZQ==",
        algorithm="Ed25519",
        signed_at=0.0,
        verified=True,
        verification_url="https://verify.example/sig",
    )
    assert response.type == "signature"
    assert response.bitcoin_anchor is None
    assert response.signature_envelope is None
    assert response.anchors is None
    assert response.algorithm_registry_version is None
