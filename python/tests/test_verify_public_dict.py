"""asqav.verify(id) returns a plain dict with a locally reproduced chain_hash.

Regression guard for the goal9 flagship README example
(``asqav.verify`` -> ``result["verified"]`` / ``result["agent_id"]`` /
``result["chain_hash"]``). It must resolve to the real, no-key public verify
path, and chain_hash must equal sha256(canonical_json(payload)).
"""

from __future__ import annotations

import hashlib
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav._jcs import canonical_json


def _cloud_payload() -> dict:
    return {
        "type": "signature",
        "signature_id": "sig_test_dict",
        "agent_id": "agent_dict",
        "agent_name": "dict-agent",
        "action_id": "act_dict",
        "action_type": "decision:approve",
        "payload": {
            "type": "protectmcp:decision",
            "issued_at": "2026-05-04T09:14:22.118Z",
            "issuer_id": "00000000000000000098",
        },
        "signature": "ZmFrZQ==",
        "algorithm": "ML-DSA-65",
        "signed_at": "2026-05-10T12:00:00+00:00",
        "verified": True,
        "verification_url": "https://www.asqav.com/verify/sig_test_dict",
    }


def _mock_httpx(payload: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    return response


def test_verify_returns_plain_dict_shape() -> None:
    payload = _cloud_payload()
    with patch("asqav.client.httpx.get", return_value=_mock_httpx(payload)):
        result = asqav.verify("sig_test_dict")
    assert isinstance(result, dict)
    assert result["verified"] is True
    assert result["agent_id"] == "agent_dict"
    assert result["algorithm"] == "ML-DSA-65"
    assert result["verification_url"].endswith("sig_test_dict")


def test_verify_chain_hash_matches_local_canonical() -> None:
    payload = _cloud_payload()
    expected = hashlib.sha256(canonical_json(payload["payload"])).hexdigest()
    with patch("asqav.client.httpx.get", return_value=_mock_httpx(payload)):
        result = asqav.verify("sig_test_dict")
    assert result["chain_hash"] == expected


def test_verify_chain_hash_none_when_no_payload() -> None:
    payload = _cloud_payload()
    payload["payload"] = None
    with patch("asqav.client.httpx.get", return_value=_mock_httpx(payload)):
        result = asqav.verify("sig_test_dict")
    assert result["chain_hash"] is None
