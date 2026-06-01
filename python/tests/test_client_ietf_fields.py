"""Tests for IETF Compliance Receipts profile sign-path fields.

Each test asserts one behaviour the SDK is responsible for:

* the new kwargs travel to the cloud body,
* `receipt_type` outside the `protectmcp:*` namespace is rejected
  client-side before any HTTP call,
* `compliance_mode=True` with no caller-supplied `action_ref` triggers
  the SDK to compute it as `sha256:` + sha256 over canonical JSON of
  ``{"action_type", "context"}``,
* the response parser surfaces the new fields the cloud emits.

These are the SDK side of the contract. The cloud's own pydantic
validator is exercised in `tests/conformance/test_ietf_vectors.py`.
"""

from __future__ import annotations

import hashlib
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav.canonicalize import canonicalize
from asqav.client import (
    DORA_INCIDENT_CLASS_NAMESPACE,
    RECEIPT_TYPE_NAMESPACE,
    Agent,
    SignatureResponse,
    _build_sign_body,
    _compute_action_ref,
)

# === Fixtures ===


def _agent() -> Agent:
    return Agent(
        agent_id="agent_001",
        name="test-agent",
        public_key="pk_xxx",
        key_id="key_001",
        algorithm="ML-DSA-65",
        capabilities=["api:*"],
        created_at=1700000000.0,
    )


def _ok_response(extra: dict | None = None) -> dict:
    base = {
        "signature": "sig_b64",
        "signature_id": "sig_abc",
        "action_id": "act_abc",
        "timestamp": 1700000000.0,
        "verification_url": "https://asqav.com/verify/sig_abc",
        "algorithm": "ML-DSA-65",
        "chain_hash": "deadbeef",
    }
    if extra:
        base.update(extra)
    return base


# === Namespace + validation ===


def test_receipt_type_namespace_constant() -> None:
    """Namespace matches the cloud's pydantic validator exactly."""
    assert RECEIPT_TYPE_NAMESPACE == frozenset(
        {
            "protectmcp:decision",
            "protectmcp:restraint",
            "protectmcp:lifecycle",
            "protectmcp:lifecycle:configuration_change",
            # risk-acceptance / exception receipt; no-policy lifecycle opt-out.
            "protectmcp:lifecycle:risk_acceptance",
            "protectmcp:acknowledgment",
            "protectmcp:observation",
            # NSA CSI U/OO/6030316-26 alignment (cloud 0.5.0): result-bound
            # observation receipts use the ``:result_bound`` suffix.
            "protectmcp:observation:result_bound",
        }
    )


def test_invalid_receipt_type_raises_before_http() -> None:
    """SDK rejects bad receipt_type without making a network call."""
    with patch("asqav.client._post") as p:
        with pytest.raises(ValueError, match="invalid_receipt_type"):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                receipt_type="not:in:namespace",
            )
        p.assert_not_called()


def test_deny_without_reason_raises_before_http() -> None:
    """`policy_decision=deny` without `reason` is rejected client-side."""
    with patch("asqav.client._post") as p:
        with pytest.raises(ValueError, match="missing_reason"):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                policy_decision="deny",
            )
        p.assert_not_called()


# === action_ref auto-compute ===


def test_compute_action_ref_matches_canonical_sha256() -> None:
    """Helper hash matches the JCS-canonical sha256 over the action object."""
    action_type = "api:call"
    context = {"model": "gpt-4", "params": {"temp": 0.0}}
    expected = "sha256:" + hashlib.sha256(
        canonicalize({"action_type": action_type, "context": context})
    ).hexdigest()
    assert _compute_action_ref(action_type, context) == expected


def test_compliance_mode_auto_computes_action_ref() -> None:
    """When the caller omits action_ref under compliance_mode, the SDK
    computes it from canonical JSON of the action object so the cloud's
    `action_ref` matches its own digest of the action."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"a": 1, "b": 2},
            compliance_mode=True,
            receipt_type="protectmcp:decision",
        )

    body = captured["body"]
    assert body["compliance_mode"] is True
    assert body["receipt_type"] == "protectmcp:decision"
    expected = _compute_action_ref("api:call", {"a": 1, "b": 2})
    assert body["action_ref"] == expected


def test_compliance_mode_caller_supplied_action_ref_wins() -> None:
    """A caller-supplied action_ref is forwarded verbatim, no recompute."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    forced = "sha256:" + "a" * 64
    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"x": 1},
            compliance_mode=True,
            action_ref=forced,
        )
    assert captured["body"]["action_ref"] == forced


def test_no_compliance_mode_means_no_extra_fields_on_body() -> None:
    """compliance_mode is now on by default, so the body carries the
    Compliance Receipts fields without the caller having to opt in.

    The "no extra fields" shape is preserved by passing
    ``compliance_mode=False`` explicitly.
    """
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign("api:call", {"k": "v"}, compliance_mode=False)

    body = captured["body"]
    for k in (
        "compliance_mode",
        "receipt_type",
        "action_ref",
        "iteration_id",
        "sandbox_state",
        "risk_class",
        "incident_class",
        "reason",
        "issuer_id",
        "payload_digest",
    ):
        assert k not in body, f"unexpected compliance-mode-off field: {k}"


# === Pass-through of all profile fields ===


def test_all_profile_fields_passthrough_in_body() -> None:
    """Every profile field the caller sets reaches the cloud body verbatim."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            receipt_type="protectmcp:lifecycle",
            iteration_id="iter_42",
            sandbox_state="enabled",
            risk_class="high",
            incident_class="cybersecurity_related",
            issuer_id="legal:Acme GmbH",
            payload_digest="sha256:" + "b" * 64,
        )

    body = captured["body"]
    assert body["compliance_mode"] is True
    assert body["receipt_type"] == "protectmcp:lifecycle"
    assert body["iteration_id"] == "iter_42"
    assert body["sandbox_state"] == "enabled"
    assert body["risk_class"] == "high"
    assert body["incident_class"] == "cybersecurity_related"
    assert body["issuer_id"] == "legal:Acme GmbH"
    assert body["payload_digest"] == "sha256:" + "b" * 64


def test_deny_with_reason_passes_through() -> None:
    """`policy_decision=deny` plus `reason` reaches the body without raising."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            policy_decision="deny",
            reason="policy_violation_outbound_url",
        )
    assert captured["body"]["policy_decision"] == "deny"
    assert captured["body"]["reason"] == "policy_violation_outbound_url"


# === Response parsing ===


def test_signature_response_surfaces_new_fields() -> None:
    """SignatureResponse exposes IETF fields when the cloud returns them."""
    extra = {
        "compliance_mode": True,
        "receipt_type": "protectmcp:decision",
        "action_ref": "sha256:" + "c" * 64,
        "payload_digest": "sha256:" + "d" * 64,
        "issuer_id": "legal:Acme GmbH",
        "iteration_id": "iter_42",
        "sandbox_state": "enabled",
        "risk_class": "high",
        "incident_class": "cybersecurity_related",
        "reason": None,
        "previousReceiptHash": "0" * 64,
    }
    with patch("asqav.client._post", return_value=_ok_response(extra)):
        resp: SignatureResponse = _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
        )
    assert resp.compliance_mode is True
    assert resp.receipt_type == "protectmcp:decision"
    assert resp.action_ref == "sha256:" + "c" * 64
    assert resp.payload_digest == "sha256:" + "d" * 64
    assert resp.issuer_id == "legal:Acme GmbH"
    assert resp.iteration_id == "iter_42"
    assert resp.sandbox_state == "enabled"
    assert resp.risk_class == "high"
    assert resp.incident_class == "cybersecurity_related"
    assert resp.previous_receipt_hash == "0" * 64


def test_response_accepts_snake_case_previous_receipt_hash() -> None:
    """Older or self-hosted servers may emit `previous_receipt_hash`."""
    extra = {
        "compliance_mode": True,
        "receipt_type": "protectmcp:decision",
        "previous_receipt_hash": "f" * 64,
    }
    with patch("asqav.client._post", return_value=_ok_response(extra)):
        resp = _agent().sign("api:call", {}, compliance_mode=True)
    assert resp.previous_receipt_hash == "f" * 64


# === Re-export ===


def test_constant_is_re_exported_at_top_level() -> None:
    """`asqav.RECEIPT_TYPE_NAMESPACE` is part of the public surface."""
    assert asqav.RECEIPT_TYPE_NAMESPACE == RECEIPT_TYPE_NAMESPACE


# === _build_sign_body branches ===


def test_build_sign_body_full_payload_includes_compliance_fields() -> None:
    body = _build_sign_body(
        action_type="api:call",
        context={"k": "v"},
        session_id="sess_1",
        agent_id="agent_001",
        compliance_fields={
            "compliance_mode": True,
            "receipt_type": "protectmcp:decision",
            "iteration_id": None,  # None entries dropped
        },
    )
    assert body["compliance_mode"] is True
    assert body["receipt_type"] == "protectmcp:decision"
    assert "iteration_id" not in body


def test_build_sign_body_omits_compliance_when_no_fields() -> None:
    body = _build_sign_body(
        action_type="api:call",
        context={"k": "v"},
        session_id=None,
        agent_id="agent_001",
    )
    assert "compliance_mode" not in body


# === DORA RTS JC 2024-33 Annex II vocabulary ===


def test_dora_incident_class_namespace_is_canonical_six() -> None:
    """Namespace matches Annex II of JC 2024-33 (RTS published 17 July 2024)."""
    assert DORA_INCIDENT_CLASS_NAMESPACE == frozenset(
        {
            "cybersecurity_related",
            "process_failure",
            "system_failure",
            "external_event",
            "payment_related",
            "other",
        }
    )


@pytest.mark.parametrize("value", sorted(DORA_INCIDENT_CLASS_NAMESPACE))
def test_canonical_incident_class_passes_validation(value: str) -> None:
    """All six canonical values are accepted and forwarded verbatim."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            incident_class=value,
        )
    assert captured["body"]["incident_class"] == value


def test_invalid_incident_class_raises_before_http() -> None:
    """A token outside the canonical six is rejected before the HTTP call."""
    with patch("asqav.client._post") as p:
        with pytest.raises(ValueError, match="invalid_incident_class"):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                incident_class="ICT-INC-001",
            )
        p.assert_not_called()


def test_pre_alignment_token_now_rejected() -> None:
    """The cloud purged the alias map; the SDK rejects the same tokens."""
    for token in (
        "cyberattack",
        "payment_fraud",
        "malicious_actions",
        "human_error",
        "unknown",
    ):
        with patch("asqav.client._post") as p:
            with pytest.raises(ValueError, match="invalid_incident_class"):
                _agent().sign(
                    "api:call",
                    {"k": "v"},
                    compliance_mode=True,
                    incident_class=token,
                )
            p.assert_not_called()


# === HIPAA Security Rule vocabulary (GAP-E13) ===


def test_hipaa_security_incident_token_accepted() -> None:
    """HIPAA 45 CFR 164.304 is a canonical `incident_class` source regime
    alongside DORA. The SDK MUST accept the HIPAA token so a Covered
    Entity is not forced into a DORA category."""
    from asqav.client import HIPAA_INCIDENT_CLASS_NAMESPACE

    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    assert "hipaa_security_incident" in HIPAA_INCIDENT_CLASS_NAMESPACE
    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            incident_class="hipaa_security_incident",
        )
    assert captured["body"]["incident_class"] == "hipaa_security_incident"


def test_incident_class_list_form_accepted_with_mixed_regimes() -> None:
    """`incident_class` MAY be a JSON array of canonical strings to preserve
    cross-regime classification (e.g. one Action that is both a DORA
    ICT-related incident and a HIPAA security incident)."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            incident_class=["cybersecurity_related", "hipaa_security_incident"],
        )
    assert captured["body"]["incident_class"] == [
        "cybersecurity_related",
        "hipaa_security_incident",
    ]


def test_incident_class_list_rejects_unknown_token_in_array() -> None:
    """A single bad token in an array MUST fail preflight; otherwise a
    typo travels to the cloud and lands a non-conformant receipt."""
    with patch("asqav.client._post") as p:
        with pytest.raises(ValueError, match="invalid_incident_class"):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                incident_class=["cybersecurity_related", "not_a_real_regime"],
            )
        p.assert_not_called()


# === Sandbox state enum (GAP-E3) ===


def test_sandbox_state_namespace_matches_upstream_enum() -> None:
    """The `sandbox_state` enum inherits the upstream values unchanged:
    {enabled, disabled, unavailable}."""
    from asqav.client import SANDBOX_STATE_NAMESPACE

    assert SANDBOX_STATE_NAMESPACE == frozenset(
        {"enabled", "disabled", "unavailable"}
    )


@pytest.mark.parametrize("value", ["enabled", "disabled", "unavailable"])
def test_canonical_sandbox_state_passes_preflight(value: str) -> None:
    """All three canonical values are accepted and forwarded verbatim."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            sandbox_state=value,
        )
    assert captured["body"]["sandbox_state"] == value


def test_invalid_sandbox_state_raises_before_http() -> None:
    """A value outside the upstream enum is rejected client-side; this
    closes the SDK side of GAP-E3 (cloud was the only enforcer)."""
    with patch("asqav.client._post") as p:
        with pytest.raises(ValueError, match="invalid_sandbox_state"):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                sandbox_state="sandboxed",  # plausible-looking typo
            )
        p.assert_not_called()
