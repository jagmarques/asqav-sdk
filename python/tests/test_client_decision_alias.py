"""Spec-shape `decision` on SignatureResponse mirrors `policy_decision`."""

from __future__ import annotations

from asqav.client import (
    DECISION_MAP,
    SignatureResponse,
    _map_policy_decision_to_decision,
)


    # Every aliased `policy_decision` token resolves to a spec token.
def test_decision_map_covers_full_alias_vocabulary():
    assert DECISION_MAP["permit"] == "allow"
    assert DECISION_MAP["deny"] == "deny"
    assert DECISION_MAP["rate_limit"] == "rate_limit"


    # Spec-shape `allow` round-trips.
def test_decision_map_idempotent_for_already_normalised_input():
    assert DECISION_MAP["allow"] == "allow"
    assert _map_policy_decision_to_decision("allow") == "allow"


    # Closed-by-default keeps misconfigured callers safe.
def test_unknown_token_falls_back_to_deny():
    assert _map_policy_decision_to_decision("yolo") == "deny"
    assert _map_policy_decision_to_decision(None) == "deny"
    assert _map_policy_decision_to_decision("") == "deny"


    # Non-compliance receipts MUST NOT carry a `decision` token.
def test_signature_response_decision_default_none():
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


    # Compliance receipts surface BOTH fields.
def test_signature_response_decision_alongside_policy_decision():
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
