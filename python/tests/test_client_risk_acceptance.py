"""SDK parity for the risk-acceptance / exception receipt.

Mirrors the deployed cloud SignRequest validators (routes/agents.py
``_validate_risk_acceptance_extensions`` + RiskSnapshot) for
``receipt_type='protectmcp:lifecycle:risk_acceptance'``. The SDK forwards the
nine extension fields verbatim, validates the sarif_digest shape, the
risk_snapshot numeric-requires-source rule, the field-fence to the
risk-acceptance type, and requires compliance_mode so the fields are never
silently dropped from the signed bytes. Each rejection carries a docs_url
(rule 3.9).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import (
    RECEIPT_TYPE_NAMESPACE,
    RISK_ACCEPTANCE_DOCS_URL,
    Agent,
    AsqavValidationError,
)

RISK_TYPE = "protectmcp:lifecycle:risk_acceptance"
GOOD_SARIF = "sha256:" + "a" * 64
GOOD_SNAPSHOT = {
    "snapshot_at": "2026-06-01T19:00:00Z",
    "snapshot_source": "CISA KEV catalog + FIRST EPSS feed",
    "epss": "0.94210",
    "cvss": "9.8",
    "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
    "kev_listed": True,
    "cve_ids": ["CVE-2026-1234", "CVE-2026-5678"],
}


def _agent() -> Agent:
    return Agent(
        agent_id="agent_risk",
        name="risk-agent",
        public_key="pk_test",
        key_id="key_risk",
        algorithm="ML-DSA-65",
        capabilities=["risk:*"],
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


def _good_kwargs() -> dict:
    return dict(
        compliance_mode=True,
        receipt_type=RISK_TYPE,
        policy_decision="none",
        approver_id="approver:alice@example.com",
        initiator_id="initiator:bob@example.com",
        acceptance_reason="Compensating control in place; accepted 90 days.",
        accepted_at="2026-06-01T19:05:00Z",
        supersedes="act_prior_0001",
        sarif_digest=GOOD_SARIF,
        finding_ref="github.com/acme/repo/security/code-scanning/42",
        approval_ref="JIRA-SEC-9001",
        risk_snapshot=dict(GOOD_SNAPSHOT),
    )


def _sign(**overrides) -> dict:
    """Sign with the good kwargs (plus overrides) and return the wire body."""
    kwargs = _good_kwargs()
    kwargs.update(overrides)
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response(
            {"compliance_mode": True, "policy_decision": "none", "decision": "observation"}
        )

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign("risk:accept", {"k": "v"}, **kwargs)
    return captured["body"]


# === namespace ===


def test_namespace_includes_risk_acceptance() -> None:
    assert RISK_TYPE in RECEIPT_TYPE_NAMESPACE


# === valid: all nine fields project ===


def test_all_risk_fields_project_to_wire() -> None:
    body = _sign()
    assert body["approver_id"] == "approver:alice@example.com"
    assert body["initiator_id"] == "initiator:bob@example.com"
    assert body["acceptance_reason"].startswith("Compensating control")
    assert body["accepted_at"] == "2026-06-01T19:05:00Z"
    assert body["supersedes"] == "act_prior_0001"
    assert body["sarif_digest"] == GOOD_SARIF
    assert body["finding_ref"].startswith("github.com")
    assert body["approval_ref"] == "JIRA-SEC-9001"
    assert body["risk_snapshot"]["snapshot_source"].startswith("CISA KEV")
    assert body["receipt_type"] == RISK_TYPE


def test_cvss_epss_are_strings_on_the_wire() -> None:
    body = _sign()
    assert isinstance(body["risk_snapshot"]["cvss"], str)
    assert isinstance(body["risk_snapshot"]["epss"], str)


# === rejections ===


def test_rejects_bad_sarif_digest_shape() -> None:
    with pytest.raises(AsqavValidationError) as exc:
        _sign(sarif_digest="not-a-sha256-digest")
    assert "sarif_digest_not_sha256_wire_form" in str(exc.value)
    assert exc.value.docs_url == RISK_ACCEPTANCE_DOCS_URL


def test_rejects_numeric_without_snapshot_source() -> None:
    bad_snapshot = {"snapshot_at": "2026-06-01T19:00:00Z", "snapshot_source": "", "cvss": "9.8"}
    with pytest.raises(AsqavValidationError) as exc:
        _sign(risk_snapshot=bad_snapshot)
    assert "risk_snapshot_numeric_requires_snapshot_source" in str(exc.value)


def test_rejects_risk_fields_on_wrong_receipt_type() -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):  # noqa: SIM117
        with pytest.raises(AsqavValidationError) as exc:
            _agent().sign(
                "lifecycle:change",
                {"k": "v"},
                compliance_mode=True,
                receipt_type="protectmcp:lifecycle",
                policy_decision="none",
                sarif_digest=GOOD_SARIF,
            )
    assert "risk_acceptance_fields_require_risk_acceptance_receipt" in str(exc.value)


def test_rejects_missing_required_field() -> None:
    with pytest.raises(AsqavValidationError) as exc:
        _sign(approver_id=None, acceptance_reason=None)
    assert "risk_acceptance_missing_required_field" in str(exc.value)


def test_rejects_without_compliance_mode() -> None:
    with pytest.raises(AsqavValidationError) as exc:
        _sign(compliance_mode=False)
    assert "risk_acceptance_requires_compliance_mode" in str(exc.value)


def test_rejects_non_none_policy_decision() -> None:
    with pytest.raises(AsqavValidationError) as exc:
        _sign(policy_decision="permit")
    assert "risk_acceptance_requires_no_policy_decision" in str(exc.value)


def test_rejects_cve_ids_empty_list() -> None:
    bad_snapshot = dict(GOOD_SNAPSHOT)
    bad_snapshot["cve_ids"] = []
    with pytest.raises(AsqavValidationError) as exc:
        _sign(risk_snapshot=bad_snapshot)
    assert "risk_snapshot_cve_ids_must_be_non_empty_list" in str(exc.value)


def test_validation_error_is_a_valueerror() -> None:
    """AsqavValidationError subclasses ValueError for backward compat."""
    with pytest.raises(ValueError):
        _sign(sarif_digest="bad")


# === SoD guard: risk_acceptance_self_approval_guard ===


@pytest.mark.parametrize("same_id", ["user:alice@example.com", "uuid-1234", "alice"])
def test_rejects_self_approval(same_id: str) -> None:
    """risk_acceptance_self_approval_guard fires when initiator_id == approver_id."""
    with pytest.raises(AsqavValidationError) as exc:
        _sign(initiator_id=same_id, approver_id=same_id)
    assert "risk_acceptance_self_approval_guard: initiator_id equals" in str(exc.value)
    assert exc.value.docs_url == RISK_ACCEPTANCE_DOCS_URL


def test_self_approval_guard_message_bytes() -> None:
    """Guard message prefix is byte-identical to the cloud validator."""
    with pytest.raises(AsqavValidationError) as exc:
        _sign(initiator_id="x", approver_id="x")
    msg = str(exc.value)
    assert msg.startswith(
        "risk_acceptance_self_approval_guard: initiator_id equals approver_id; "
        "a self-approved risk acceptance is incoherent"
    )


def test_allows_different_initiator_and_approver() -> None:
    """Different initiator and approver IDs do not trigger the SoD guard."""
    body = _sign(initiator_id="initiator:bob@example.com", approver_id="approver:alice@example.com")
    assert body["initiator_id"] == "initiator:bob@example.com"
    assert body["approver_id"] == "approver:alice@example.com"


def test_self_approval_guard_absent_initiator_allowed() -> None:
    """Guard does not fire when initiator_id is absent (only approver_id set)."""
    body = _sign(initiator_id=None)
    assert body.get("initiator_id") is None
    assert body["approver_id"] == "approver:alice@example.com"
