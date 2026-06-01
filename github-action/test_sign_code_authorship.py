"""Pure unit tests for the code-authorship signing action.

No network, no real Asqav API. The SDK sign call is mocked. These tests cover
the three load-bearing pieces of the action:

1. change_digest computation (wire form + diff hashing).
2. authored_by attestation_source enforcement (always present).
3. the in-toto Statement v1 wrapper shape (valid JSON, _type, predicateType,
   subject).
"""

import hashlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sign_code_authorship as mod  # noqa: E402


# --- change_digest computation ------------------------------------------------


def test_change_digest_hashes_supplied_diff_text():
    diff = "diff --git a/x b/x\n+hello\n"
    expected = "sha256:" + hashlib.sha256(diff.encode("utf-8")).hexdigest()
    assert mod.compute_change_digest("base", "head", diff_text=diff) == expected


def test_change_digest_is_sha256_wire_form():
    digest = mod.compute_change_digest("base", "head", diff_text="anything")
    assert digest.startswith("sha256:")
    hex_part = digest.split(":", 1)[1]
    assert len(hex_part) == 64
    assert all(c in "0123456789abcdef" for c in hex_part)


def test_change_digest_falls_back_to_head_sha_without_base():
    head = "a" * 40
    expected = "sha256:" + hashlib.sha256(head.encode("utf-8")).hexdigest()
    assert mod.compute_change_digest(None, head, diff_text=None) == expected


# --- authored_by attestation_source enforcement -------------------------------


def test_authored_by_always_has_attestation_source():
    ab = mod.build_authored_by(
        agent_id="agent-1", model_id=None, model_version=None, tool=None
    )
    assert ab["attestation_source"] == "github-actions-input"


def test_authored_by_model_fields_carry_attestation_source():
    ab = mod.build_authored_by(
        agent_id="agent-1",
        model_id="claude-opus-4-8",
        model_version="2026-05",
        tool="github-actions",
    )
    # Model claim present -> attestation_source MUST be present (SDK requires it).
    assert ab["model_id"] == "claude-opus-4-8"
    assert ab["model_version"] == "2026-05"
    assert ab["attestation_source"] == "github-actions-input"
    assert ab["tool"] == "github-actions"


def test_authored_by_omits_empty_optional_fields():
    ab = mod.build_authored_by(
        agent_id=None, model_id=None, model_version=None, tool=None
    )
    assert ab == {"attestation_source": "github-actions-input"}


# --- in-toto Statement wrapper shape ------------------------------------------


def _sample_receipt():
    return {
        "signature_id": "sig_123",
        "algorithm": "ML-DSA-65",
        "signature": {"alg": "ML-DSA-65", "kid": "key_1", "sig": "AAAA"},
        "receipt_type": mod.RECEIPT_TYPE,
    }


def test_intoto_statement_shape():
    commit = "f" * 40
    stmt = mod.build_intoto_statement(
        repo_ref="owner/repo",
        commit_sha=commit,
        change_digest="sha256:" + "0" * 64,
        signed_receipt=_sample_receipt(),
    )
    # Valid JSON round-trip.
    reparsed = json.loads(json.dumps(stmt))
    assert reparsed["_type"] == "https://in-toto.io/Statement/v1"
    assert reparsed["predicateType"] == "https://asqav.com/CodeAuthorship/v1"
    assert reparsed["subject"] == [
        {"name": f"owner/repo@{commit}", "digest": {"sha256": commit}}
    ]
    # Predicate carries the signed receipt incl ML-DSA-65 envelope.
    assert reparsed["predicate"]["algorithm"] == "ML-DSA-65"
    assert reparsed["predicate"]["signature"]["alg"] == "ML-DSA-65"


# --- end-to-end with a mocked SDK ---------------------------------------------


@dataclass
class _FakeResponse:
    signature_id: str = "sig_abc"
    action_id: str = "act_abc"
    receipt_type: str = mod.RECEIPT_TYPE
    algorithm: str = "ML-DSA-65"
    signature: object = None
    verification_url: str = "https://asqav.com/v/sig_abc"
    timestamp: float = 1234.0
    payload: object = None
    anchors: object = None
    previous_receipt_hash: object = None

    def __post_init__(self):
        if self.signature is None:
            self.signature = {"alg": "ML-DSA-65", "kid": "key_1", "sig": "AAAA"}

    def signature_envelope(self):
        if isinstance(self.signature, dict):
            return self.signature
        return None


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

    fake.init = fake_init
    fake.Agent = FakeAgent
    monkeypatch.setitem(sys.modules, "asqav", fake)


def test_sign_and_export_passes_honest_fields(monkeypatch, tmp_path):
    captured = {}
    _install_fake_asqav(monkeypatch, captured)

    ctx = mod.GitContext(
        repo_ref="owner/repo",
        commit_sha="c" * 40,
        base_sha="b" * 40,
        change_ref="https://github.com/owner/repo/pull/7",
        change_digest="sha256:" + "1" * 64,
    )
    intoto_path = tmp_path / "out.intoto.jsonl"

    outputs = mod.sign_and_export(
        api_key="sk_test",
        agent_id="agent-1",
        change_class="write",
        model_id="claude-opus-4-8",
        model_version="2026-05",
        tool="github-actions",
        intoto_path=str(intoto_path),
        git_ctx=ctx,
    )

    kwargs = captured["sign_kwargs"]
    # Honest-scope invariants on the sign call.
    assert kwargs["receipt_type"] == mod.RECEIPT_TYPE
    assert kwargs["compliance_mode"] is True
    assert kwargs["policy_decision"] == "none"
    assert kwargs["repo_ref"] == "owner/repo"
    assert kwargs["commit_sha"] == "c" * 40
    assert kwargs["base_sha"] == "b" * 40
    assert kwargs["change_digest"] == "sha256:" + "1" * 64
    assert kwargs["change_class"] == "write"
    # Model claim is always producer-asserted.
    assert kwargs["authored_by"]["attestation_source"] == "github-actions-input"
    assert kwargs["authored_by"]["model_id"] == "claude-opus-4-8"

    # Outputs.
    assert outputs["signature-id"] == "sig_abc"
    assert outputs["verification-url"] == "https://asqav.com/v/sig_abc"
    assert outputs["intoto-statement-path"] == str(intoto_path)

    # in-toto file is valid JSON with the right shape.
    stmt = json.loads(intoto_path.read_text())
    assert stmt["_type"] == "https://in-toto.io/Statement/v1"
    assert stmt["predicateType"] == "https://asqav.com/CodeAuthorship/v1"
    assert stmt["subject"][0]["name"] == "owner/repo@" + "c" * 40
    assert stmt["subject"][0]["digest"]["sha256"] == "c" * 40
    assert stmt["predicate"]["algorithm"] == "ML-DSA-65"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
