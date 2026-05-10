"""Spec-shape `decision` on SignatureResponse mirrors `policy_decision`."""

from __future__ import annotations

from asqav.client import (
    DECISION_MAP,
    SignatureResponse,
    _map_policy_decision_to_decision,
)


def test_decision_map_covers_full_alias_vocabulary():
    """Every aliased `policy_decision` token resolves to a spec token."""
    assert DECISION_MAP["permit"] == "allow"
    assert DECISION_MAP["deny"] == "deny"
    assert DECISION_MAP["rate_limit"] == "rate_limit"


def test_decision_map_idempotent_for_already_normalised_input():
    """Spec-shape `allow` round-trips."""
    assert DECISION_MAP["allow"] == "allow"
    assert _map_policy_decision_to_decision("allow") == "allow"


def test_unknown_token_falls_back_to_deny():
    """Closed-by-default keeps misconfigured callers safe."""
    assert _map_policy_decision_to_decision("yolo") == "deny"
    assert _map_policy_decision_to_decision(None) == "deny"
    assert _map_policy_decision_to_decision("") == "deny"


def test_signature_response_decision_default_none():
    """Non-compliance receipts MUST NOT carry a `decision` token."""
    resp = SignatureResponse(
        signature="ZmFrZQ==",
        signature_id="sig_test",
        action_id="act_test",
        timestamp=0.0,
        verification_url="https://verify/example",
    )
    assert resp.decision is None
    # Aliased field default preserved.
    assert resp.policy_decision == "permit"


def test_signature_response_decision_alongside_policy_decision():
    """Compliance receipts surface BOTH fields."""
    resp = SignatureResponse(
        signature="ZmFrZQ==",
        signature_id="sig_test",
        action_id="act_test",
        timestamp=0.0,
        verification_url="https://verify/example",
        policy_decision="permit",
        decision="allow",
        compliance_mode=True,
    )
    assert resp.policy_decision == "permit"
    assert resp.decision == "allow"


def test_signature_response_deny_round_trip():
    assert _map_policy_decision_to_decision("deny") == "deny"


def test_signature_response_rate_limit_round_trip():
    assert _map_policy_decision_to_decision("rate_limit") == "rate_limit"
