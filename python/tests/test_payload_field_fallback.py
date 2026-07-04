"""The sign response resolves action_ref and previous_receipt_hash from payload.

Compliance-mode clouds nest ``action_ref`` and ``previousReceiptHash`` inside
the receipt ``payload`` rather than at the top level of the sign response, so a
top-level-only read surfaced None for both on the quickstart flow. The client
reads the top level first, then falls back to the payload copy, and leaves the
payload object itself unchanged.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav.async_client import AsyncAgent

_ACTION_REF = "sha256:" + "a" * 64
_PREV_HASH = "0" * 64


def _agent() -> "asqav.Agent":
    return asqav.Agent(
        agent_id="agent_001",
        name="test-agent",
        public_key="pk_xxx",
        key_id="key_001",
        algorithm="ML-DSA-65",
        capabilities=["api:*"],
        created_at=1700000000.0,
    )


def _async_agent() -> AsyncAgent:
    return AsyncAgent(
        agent_id="agent_001",
        name="test-agent",
        public_key="pk_xxx",
        key_id="key_001",
        algorithm="ML-DSA-65",
        capabilities=["api:*"],
        created_at=1700000000.0,
    )


def _wire_with_payload_only() -> dict:
    """Compliance-mode sign response: the two fields live only inside payload."""
    return {
        "signature": "sig_b64",
        "signature_id": "sig_abc",
        "action_id": "act_abc",
        "timestamp": 1700000000.0,
        "verification_url": "https://asqav.com/verify/sig_abc",
        "algorithm": "ML-DSA-65",
        "compliance_mode": True,
        "payload": {
            "v": 1,
            "action_ref": _ACTION_REF,
            "previousReceiptHash": _PREV_HASH,
        },
    }


def test_sync_sign_resolves_action_ref_and_prev_hash_from_payload() -> None:
    with patch("asqav.client._post", return_value=_wire_with_payload_only()):
        resp = _agent().sign("api:call", {"k": "v"}, compliance_mode=True)
    assert resp.action_ref == _ACTION_REF
    assert resp.previous_receipt_hash == _PREV_HASH


def test_sync_top_level_wins_over_payload() -> None:
    """A top-level value is authoritative; the payload copy is only a fallback."""
    wire = _wire_with_payload_only()
    wire["action_ref"] = "sha256:" + "b" * 64
    wire["previous_receipt_hash"] = "1" * 64
    with patch("asqav.client._post", return_value=wire):
        resp = _agent().sign("api:call", {"k": "v"}, compliance_mode=True)
    assert resp.action_ref == "sha256:" + "b" * 64
    assert resp.previous_receipt_hash == "1" * 64


def test_sync_payload_object_returned_untouched() -> None:
    with patch("asqav.client._post", return_value=_wire_with_payload_only()):
        resp = _agent().sign("api:call", {"k": "v"}, compliance_mode=True)
    assert resp.payload == {
        "v": 1,
        "action_ref": _ACTION_REF,
        "previousReceiptHash": _PREV_HASH,
    }


@pytest.mark.asyncio
@patch("asqav.async_client._async_post", new_callable=AsyncMock)
async def test_async_sign_resolves_from_payload(mock_post: AsyncMock) -> None:
    mock_post.return_value = _wire_with_payload_only()
    resp = await _async_agent().sign(action_type="api:call", context={"k": "v"})
    assert resp.action_ref == _ACTION_REF
    assert resp.previous_receipt_hash == _PREV_HASH
