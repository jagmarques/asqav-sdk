"""Pure unit tests for the risk-acceptance action.

No network, no real Asqav API. The SDK sign and verify calls are mocked. These
tests cover the load-bearing pieces of the action:

1. expires parsing (durations and ISO dates to valid_seconds).
2. sarif_digest computation (wire form).
3. approver and initiator extraction from the event payload.
4. the mint sign call invariants (receipt type, compliance mode, field map).
5. manifest parsing and the client-side audit filter (expired, superseded,
   missing, wrong type).
"""

import hashlib
import json
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import risk_acceptance as mod  # noqa: E402


# --- expires parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("90d", 90 * 86400),
        ("12h", 12 * 3600),
        ("30m", 30 * 60),
        ("3600s", 3600),
        ("2w", 2 * 604800),
        ("7200", 7200),
    ],
)
def test_parse_expires_durations(value, expected):
    assert mod.parse_expires(value) == expected


def test_parse_expires_iso_date_converts_to_seconds():
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert mod.parse_expires("2026-06-02", now=now) == 86400


@pytest.mark.parametrize("value", ["", "soon", "-5d", "0d", "2020-01-01"])
def test_parse_expires_rejects_bad_values(value):
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        mod.parse_expires(value, now=now)


# --- sarif digest --------------------------------------------------------------


def test_sarif_digest_is_sha256_wire_form(tmp_path):
    sarif = tmp_path / "scan.sarif"
    sarif.write_bytes(b'{"runs": []}')
    expected = "sha256:" + hashlib.sha256(b'{"runs": []}').hexdigest()
    assert mod.compute_sarif_digest(str(sarif)) == expected


# --- approval context extraction ------------------------------------------------


def test_approval_from_review_event():
    event = {
        "review": {
            "user": {"login": "sec-reviewer"},
            "submitted_at": "2026-06-09T10:00:00Z",
            "html_url": "https://github.com/o/r/pull/7#pullrequestreview-1",
        },
        "pull_request": {
            "user": {"login": "pr-author"},
            "html_url": "https://github.com/o/r/pull/7",
        },
    }
    ctx = mod.read_approval_context(event)
    assert ctx.approver == "sec-reviewer"
    assert ctx.initiator == "pr-author"
    assert ctx.accepted_at == "2026-06-09T10:00:00Z"
    assert ctx.approval_ref == "https://github.com/o/r/pull/7#pullrequestreview-1"


def test_approval_from_merged_pull_request_event():
    event = {
        "pull_request": {
            "user": {"login": "pr-author"},
            "merged_by": {"login": "merger"},
            "html_url": "https://github.com/o/r/pull/8",
        },
    }
    ctx = mod.read_approval_context(event)
    assert ctx.approver == "merger"
    assert ctx.initiator == "pr-author"
    assert ctx.accepted_at is None
    assert ctx.approval_ref == "https://github.com/o/r/pull/8"


def test_approval_missing_approver_raises():
    with pytest.raises(ValueError):
        mod.read_approval_context({"pull_request": {"user": {"login": "x"}}})


# --- mint sign call with a mocked SDK -------------------------------------------


@dataclass
class _FakeResponse:
    signature_id: str = "sig_ra"
    verification_url: str = "https://asqav.com/v/sig_ra"


def _install_fake_asqav(monkeypatch, captured):
    fake = types.ModuleType("asqav")

    def fake_init(api_key=None, **kwargs):
        captured["api_key"] = api_key

    class FakeAgent:
        def __init__(self, agent_id):
            self.agent_id = agent_id

        @classmethod
        def get(cls, agent_id):
            captured["agent_id"] = agent_id
            return cls(agent_id)

        def sign(self, action_type, **kwargs):
            captured["action_type"] = action_type
            captured["sign_kwargs"] = kwargs
            return _FakeResponse()

    class FakeAPIError(Exception):
        def __init__(self, message, status_code=None):
            super().__init__(message)
            self.status_code = status_code

    fake.init = fake_init
    fake.Agent = FakeAgent
    fake.APIError = FakeAPIError
    monkeypatch.setitem(sys.modules, "asqav", fake)
    return fake


def test_mint_passes_risk_acceptance_fields(monkeypatch):
    captured = {}
    _install_fake_asqav(monkeypatch, captured)
    approval = mod.ApprovalContext(
        approver="sec-reviewer",
        initiator="pr-author",
        accepted_at="2026-06-09T10:00:00Z",
        approval_ref="https://github.com/o/r/pull/7",
    )

    outputs = mod.mint(
        api_key="sk_test",
        agent_id="agent-1",
        reason="compensating control in place",
        finding_id="py/sql-injection:fp-123",
        sarif_digest="sha256:" + "a" * 64,
        valid_seconds=90 * 86400,
        supersedes="sig_old",
        risk_snapshot={"snapshot_at": "2026-06-09T09:00:00Z"},
        approval=approval,
    )

    kwargs = captured["sign_kwargs"]
    assert kwargs["receipt_type"] == mod.RECEIPT_TYPE
    assert kwargs["compliance_mode"] is True
    assert kwargs["policy_decision"] == "none"
    assert kwargs["approver_id"] == "sec-reviewer"
    assert kwargs["initiator_id"] == "pr-author"
    assert kwargs["acceptance_reason"] == "compensating control in place"
    assert kwargs["accepted_at"] == "2026-06-09T10:00:00Z"
    assert kwargs["supersedes"] == "sig_old"
    assert kwargs["sarif_digest"] == "sha256:" + "a" * 64
    assert kwargs["finding_ref"] == "py/sql-injection:fp-123"
    assert kwargs["approval_ref"] == "https://github.com/o/r/pull/7"
    assert kwargs["risk_snapshot"] == {"snapshot_at": "2026-06-09T09:00:00Z"}
    assert kwargs["valid_seconds"] == 90 * 86400
    assert outputs["signature-id"] == "sig_ra"
    assert outputs["verification-url"] == "https://asqav.com/v/sig_ra"


# --- manifest parsing ------------------------------------------------------------


def test_parse_manifest_plain_lines():
    text = "# suppressions\nsig_a\n\nsig_b\n"
    assert mod.parse_manifest(text) == ["sig_a", "sig_b"]


def test_parse_manifest_json_array():
    assert mod.parse_manifest(json.dumps(["sig_a", "sig_b"])) == ["sig_a", "sig_b"]


def test_parse_manifest_json_object():
    assert mod.parse_manifest(json.dumps({"receipts": ["sig_a"]})) == ["sig_a"]


def test_parse_manifest_empty_raises():
    with pytest.raises(ValueError):
        mod.parse_manifest("\n# nothing\n")


# --- client-side audit filter -----------------------------------------------------


def _ok_receipt(sig_id, **payload_extra):
    payload = {"receipt_type": mod.RECEIPT_TYPE, **payload_extra}
    return {
        "signature_id": sig_id,
        "verified": True,
        "validation_label": "fully_verified",
        "payload": payload,
    }


def test_audit_passes_valid_receipts():
    receipts = {"sig_a": _ok_receipt("sig_a"), "sig_b": _ok_receipt("sig_b")}
    assert mod.evaluate_audit(receipts) == []


def test_audit_fails_expired_receipt():
    rec = _ok_receipt("sig_a")
    rec["validation_label"] = mod.EXPIRED_LABEL
    rec["verified"] = False
    failures = mod.evaluate_audit({"sig_a": rec})
    assert len(failures) == 1
    assert "expired" in failures[0]


def test_audit_fails_superseded_receipt():
    receipts = {
        "sig_old": _ok_receipt("sig_old"),
        "sig_new": _ok_receipt("sig_new", supersedes="sig_old"),
    }
    failures = mod.evaluate_audit(receipts)
    assert failures == ["sig_old: superseded by sig_new."]


def test_audit_fails_missing_receipt():
    failures = mod.evaluate_audit({"sig_gone": None})
    assert failures == ["sig_gone: receipt not found."]


def test_audit_fails_wrong_receipt_type():
    rec = _ok_receipt("sig_a")
    rec["payload"]["receipt_type"] = "protectmcp:lifecycle:code_authorship"
    failures = mod.evaluate_audit({"sig_a": rec})
    assert len(failures) == 1
    assert "receipt_type" in failures[0]


def test_audit_unverified_receipt_fails():
    rec = _ok_receipt("sig_a")
    rec["verified"] = False
    rec["validation_label"] = "signature_invalid"
    failures = mod.evaluate_audit({"sig_a": rec})
    assert len(failures) == 1
    assert "failed verification" in failures[0]
