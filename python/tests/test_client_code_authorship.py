"""SDK parity for the code-authorship receipt.

Mirrors the deployed cloud SignRequest validators (routes/agents.py
``_validate_code_authorship_extensions`` + AuthoredBy) for
``receipt_type='protectmcp:lifecycle:code_authorship'``. The SDK forwards the
eight extension fields verbatim, validates the change_digest shape, the
change_class closed vocabulary, the authored_by model-requires-attestation_source
rule, the field-fence to the code-authorship type, and requires compliance_mode
so the fields are never silently dropped from the signed bytes. Honest-scope:
every field is producer-asserted / recorded-not-verified; Asqav never re-runs the
diff or attests the model. Each rejection carries a docs_url (rule 3.9).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import (
    CODE_AUTHORSHIP_DOCS_URL,
    RECEIPT_TYPE_NAMESPACE,
    Agent,
    AsqavValidationError,
)

CODE_TYPE = "protectmcp:lifecycle:code_authorship"
GOOD_DIGEST = "sha256:" + "a" * 64
GOOD_AUTHORED_BY = {
    "human_id": "human:alice@example.com",
    "model_id": "claude-opus-4-8",
    "model_version": "2026-01",
    "attestation_source": "GitHub Copilot session export",
}


def _agent() -> Agent:
    return Agent(
        agent_id="agent_code",
        name="code-agent",
        public_key="pk_test",
        key_id="key_code",
        algorithm="ML-DSA-65",
        capabilities=["code:*"],
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
        receipt_type=CODE_TYPE,
        policy_decision="none",
        repo_ref="github.com/acme/repo",
        commit_sha="c0ffee0000000000000000000000000000000000",
        base_sha="ba5e0000000000000000000000000000000000000",
        change_digest=GOOD_DIGEST,
        change_ref="github.com/acme/repo/pull/42",
        change_approval_ref="JIRA-ENG-9001",
        change_class="write",
        authored_by=dict(GOOD_AUTHORED_BY),
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
        _agent().sign("code:author", {"k": "v"}, **kwargs)
    return captured["body"]


# === namespace ===


def test_namespace_includes_code_authorship() -> None:
    assert CODE_TYPE in RECEIPT_TYPE_NAMESPACE


# === valid: all eight fields project ===


def test_all_code_fields_project_to_wire() -> None:
    body = _sign()
    assert body["repo_ref"] == "github.com/acme/repo"
    assert body["commit_sha"].startswith("c0ffee")
    assert body["base_sha"].startswith("ba5e")
    assert body["change_digest"] == GOOD_DIGEST
    assert body["change_ref"] == "github.com/acme/repo/pull/42"
    assert body["change_approval_ref"] == "JIRA-ENG-9001"
    assert body["change_class"] == "write"
    assert body["authored_by"]["attestation_source"].startswith("GitHub Copilot")
    assert body["receipt_type"] == CODE_TYPE


def test_valid_human_only_author_without_attestation_source() -> None:
    """A human-only author needs no attestation_source (no model claim)."""
    body = _sign(authored_by={"human_id": "human:alice@example.com"})
    assert body["authored_by"]["human_id"] == "human:alice@example.com"


# === rejections ===


def test_rejects_bad_change_digest_shape() -> None:
    with pytest.raises(AsqavValidationError) as exc:
        _sign(change_digest="not-a-sha256-digest")
    assert "change_digest_not_sha256_wire_form" in str(exc.value)
    assert exc.value.docs_url == CODE_AUTHORSHIP_DOCS_URL


def test_rejects_change_class_outside_vocabulary() -> None:
    with pytest.raises(AsqavValidationError) as exc:
        _sign(change_class="merge")
    assert "code_authorship_change_class_invalid" in str(exc.value)


def test_rejects_model_author_without_attestation_source() -> None:
    bad_author = {"model_id": "claude-opus-4-8", "attestation_source": ""}
    with pytest.raises(AsqavValidationError) as exc:
        _sign(authored_by=bad_author)
    assert "authored_by_model_requires_attestation_source" in str(exc.value)


def test_rejects_code_fields_on_wrong_receipt_type() -> None:
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
                change_digest=GOOD_DIGEST,
            )
    assert "code_authorship_fields_require_code_authorship_receipt" in str(exc.value)


def test_rejects_missing_required_field() -> None:
    with pytest.raises(AsqavValidationError) as exc:
        _sign(repo_ref=None, commit_sha=None)
    assert "code_authorship_missing_required_field" in str(exc.value)


def test_rejects_without_compliance_mode() -> None:
    with pytest.raises(AsqavValidationError) as exc:
        _sign(compliance_mode=False)
    assert "code_authorship_requires_compliance_mode" in str(exc.value)


def test_rejects_non_none_policy_decision() -> None:
    with pytest.raises(AsqavValidationError) as exc:
        _sign(policy_decision="permit")
    assert "code_authorship_requires_no_policy_decision" in str(exc.value)


def test_validation_error_is_a_valueerror() -> None:
    """AsqavValidationError subclasses ValueError for backward compat."""
    with pytest.raises(ValueError):
        _sign(change_digest="bad")
