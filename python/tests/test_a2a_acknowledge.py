"""Agent-to-agent receipts: the sender hands over its receipt, the receiver acknowledges it.
Agent A signs an action and passes ``response.receipt()`` to agent B (in the ``x-asqav-receipt``
header via ``encode_receipt_header``). B signs a ``protectmcp:acknowledgment`` with
``Agent.acknowledge``; its ``counterparty_binding`` commits to A's exact envelope bytes, which is
what the cloud's admit_counterparty_binding and the offline verifier recompute.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.acceptor import _decode_receipt, encode_receipt_header
from asqav.client import Agent, SignatureResponse
from asqav.counterparty import (
    ACKNOWLEDGMENT_RECEIPT_TYPE,
    compute_envelope_hash,
    verify_counterparty_binding,
)

PEER = {
    "signature_id": "sig_peer",
    "payload": {"v": 1, "type": "protectmcp:decision", "action_id": "act_peer", "n": 3},
    "signature": {"alg": "ML-DSA-65", "kid": "kid_peer", "sig": "c2lnX2J5dGVz"},
}


def _agent(agent_id: str = "agent_b") -> Agent:
    return Agent(
        agent_id=agent_id,
        name=agent_id,
        public_key="pk_test",
        key_id=f"key_{agent_id}",
        algorithm="ML-DSA-65",
        capabilities=["a2a:*"],
        created_at=1700000000.0,
    )


def _ok(extra: dict | None = None) -> dict:
    base = {
        "signature": {"alg": "ML-DSA-65", "kid": "kid_b", "sig": "YWNr"},
        "signature_id": "sig_ack",
        "action_id": "act_ack",
        "timestamp": 1700000000.0,
        "verification_url": "https://asqav.com/verify/sig_ack",
        "algorithm": "ML-DSA-65",
        "compliance_mode": True,
        "payload": {"v": 1, "type": ACKNOWLEDGMENT_RECEIPT_TYPE},
    }
    base.update(extra or {})
    return base


def _acknowledge(peer: dict, **kwargs) -> dict:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["path"], captured["body"] = path, body
        return _ok()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().acknowledge(peer, **kwargs)
    return captured


def test_acknowledge_binds_the_peer_envelope_bytes() -> None:
    sent = _acknowledge(PEER, expect_ack_from="kid_b", transport_label="https")
    body = sent["body"]
    assert sent["path"] == "/agents/agent_b/sign"
    assert body["receipt_type"] == ACKNOWLEDGMENT_RECEIPT_TYPE
    assert body["compliance_mode"] is True
    binding = body["counterparty_binding"]
    assert binding["receipt_ref"] == "sig_peer"
    assert binding["envelope_hash"] == compute_envelope_hash(PEER)
    assert binding["expect_ack_from"] == "kid_b"
    assert binding["transport_label"] == "https"


def test_binding_verifies_against_the_peer_receipt() -> None:
    binding = _acknowledge(PEER)["body"]["counterparty_binding"]
    ack = {"payload": {"counterparty_binding": binding}, "signature": {"kid": "kid_b"}}
    assert verify_counterparty_binding(ack, PEER).label == "matches"
    tampered = dict(PEER, payload=dict(PEER["payload"], n=4))
    assert verify_counterparty_binding(ack, tampered).label == "mismatch"


def test_explicit_receipt_ref_wins_and_one_is_required() -> None:
    peer = {k: v for k, v in PEER.items() if k != "signature_id"}
    assert (
        _acknowledge(peer, receipt_ref="sig_given")["body"]["counterparty_binding"]["receipt_ref"]
        == "sig_given"
    )
    with pytest.raises(ValueError, match="receipt_ref is required"):
        _acknowledge(peer)


def test_binding_is_fenced_to_acknowledgment_receipts() -> None:
    with patch("asqav.client._post") as post:
        with pytest.raises(
            ValueError, match="counterparty_binding_requires_acknowledgment_receipt"
        ):
            _agent().sign(
                "a2a:send",
                receipt_type="protectmcp:decision",
                counterparty_binding={"envelope_hash": "x", "receipt_ref": "y"},
            )
        with pytest.raises(ValueError, match="counterparty_binding_malformed"):
            _agent().sign(
                "a2a:ack",
                receipt_type=ACKNOWLEDGMENT_RECEIPT_TYPE,
                counterparty_binding={"envelope_hash": "x"},
            )
        post.assert_not_called()


def test_receipt_hands_over_the_signed_envelope() -> None:
    with patch("asqav.client._post", return_value=_ok()):
        response = _agent("agent_a").sign("a2a:send", {"to": "agent_b"})
    receipt = response.receipt()
    assert receipt == {
        "signature_id": "sig_ack",
        "payload": {"v": 1, "type": ACKNOWLEDGMENT_RECEIPT_TYPE},
        "signature": {"alg": "ML-DSA-65", "kid": "kid_b", "sig": "YWNr"},
    }


def test_receipt_is_none_without_a_signed_envelope() -> None:
    plain = SignatureResponse(
        signature="sig_b64", signature_id="s", action_id="a", timestamp=0.0, verification_url="u"
    )
    assert plain.receipt() is None


def test_receipt_header_round_trips_through_the_acceptor() -> None:
    header = encode_receipt_header(PEER)
    assert header.isascii() and " " not in header
    assert _decode_receipt(header.encode("ascii")) == PEER
