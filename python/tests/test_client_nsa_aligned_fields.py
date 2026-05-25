"""SDK parity for the cycle-30 NSA-aligned receipt extensions.

Five new wire fields land on the cloud `SignRequest` per NSA CSI
U/OO/6030316-26: ``result_digest``, ``expires_at``, ``nonce``,
``tool_fingerprint``, ``config_manifest_digest`` and
``cve_inventory_digest``. The SDK forwards the caller-supplied digests
verbatim, computes ``tool_fingerprint`` from ``tool_name`` + a JSON
schema, and auto-generates a 24-hex-char ``nonce`` when the caller omits
one. The cross-field validator mirrors cloud rule 9: a
``protectmcp:lifecycle:configuration_change`` receipt without
``config_manifest_digest`` fails before the HTTP roundtrip.
"""

from __future__ import annotations

import os
import re
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import (
    Agent,
    _compute_tool_fingerprint,
    _generate_nonce,
)


def _agent() -> Agent:
    return Agent(
        agent_id="agent_nsa",
        name="nsa-agent",
        public_key="pk_xxx",
        key_id="key_nsa",
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


# === nonce auto-generation ===


def test_nonce_auto_generated_when_omitted() -> None:
    """The SDK supplies a 24-hex-char nonce when the caller omits one."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign("api:call", {"k": "v"}, compliance_mode=True)

    nonce = captured["body"]["nonce"]
    assert isinstance(nonce, str)
    assert re.fullmatch(r"[0-9a-f]{24}", nonce)


def test_nonce_caller_supplied_value_wins() -> None:
    """An explicit nonce is forwarded verbatim instead of regenerated."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            nonce="0123456789abcdef01234567",
        )
    assert captured["body"]["nonce"] == "0123456789abcdef01234567"


def test_generate_nonce_is_24_hex_chars() -> None:
    """The helper returns a stable 24-hex-char string."""
    nonce = _generate_nonce()
    assert re.fullmatch(r"[0-9a-f]{24}", nonce)


# === tool_fingerprint compute + passthrough ===


def test_tool_fingerprint_helper_is_deterministic() -> None:
    """JCS-canonical input -> stable sha256:<hex>."""
    fp1 = _compute_tool_fingerprint("search", {"a": 1, "b": 2})
    fp2 = _compute_tool_fingerprint("search", {"b": 2, "a": 1})
    assert fp1 == fp2
    assert fp1.startswith("sha256:")


def test_tool_fingerprint_auto_computed_when_schema_present() -> None:
    """The SDK derives ``tool_fingerprint`` from ``tool_name`` + schema."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            tool_name="search",
            tool_schema={"args": {"q": "string"}},
            compliance_mode=True,
        )

    fp = captured["body"]["tool_fingerprint"]
    assert fp.startswith("sha256:")
    assert len(fp) == len("sha256:") + 64


def test_tool_fingerprint_caller_value_wins() -> None:
    """Explicit ``tool_fingerprint`` overrides auto-derivation."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    explicit = "sha256:" + "a" * 64
    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            tool_name="search",
            tool_schema={"args": {"q": "string"}},
            tool_fingerprint=explicit,
            compliance_mode=True,
        )
    assert captured["body"]["tool_fingerprint"] == explicit


def test_tool_fingerprint_absent_when_no_schema_provided() -> None:
    """No auto-fingerprint when ``tool_schema`` is omitted."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            tool_name="search",
            compliance_mode=True,
        )
    assert "tool_fingerprint" not in captured["body"]


# === caller-forwarded digest fields ===


@pytest.mark.parametrize(
    "field, value",
    [
        ("result_digest", "sha256:" + "b" * 64),
        ("expires_at", "2026-06-01T00:00:00Z"),
        ("config_manifest_digest", "sha256:" + "c" * 64),
        ("cve_inventory_digest", "sha256:" + "d" * 64),
    ],
)
def test_caller_digest_fields_forwarded_verbatim(field: str, value: str) -> None:
    """The SDK passes ``result_digest``, ``expires_at``,
    ``config_manifest_digest`` and ``cve_inventory_digest`` verbatim."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            **{field: value},
        )
    assert captured["body"][field] == value


@pytest.mark.parametrize(
    "field",
    [
        "result_digest",
        "expires_at",
        "tool_fingerprint",
        "config_manifest_digest",
        "cve_inventory_digest",
    ],
)
def test_nsa_aligned_fields_absent_when_not_set(field: str) -> None:
    """No body key when the caller does not pass the kwarg."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign("api:call", {"k": "v"}, compliance_mode=True)
    assert field not in captured["body"]


# === Rule 9: configuration_change requires config_manifest_digest ===


def test_configuration_change_requires_config_manifest_digest() -> None:
    """The cross-field validator rejects ``configuration_change`` without
    ``config_manifest_digest``. Verbatim message kept byte-for-byte in
    lockstep with the cloud cross-field validator."""
    with patch("asqav.client._post") as p:
        with pytest.raises(ValueError) as exc_info:
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                receipt_type="protectmcp:lifecycle:configuration_change",
                policy_decision="none",
            )
        msg = str(exc_info.value)
        assert msg == (
            "false_attestation_guard: receipt_type="
            "protectmcp:lifecycle:configuration_change "
            "requires config_manifest_digest (rule 9)"
        )
        p.assert_not_called()


def test_configuration_change_accepts_when_digest_present() -> None:
    """The matched pair (configuration_change + config_manifest_digest)
    passes the SDK gate."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    digest = "sha256:" + "e" * 64
    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            receipt_type="protectmcp:lifecycle:configuration_change",
            policy_decision="none",
            config_manifest_digest=digest,
        )
    assert captured["body"]["config_manifest_digest"] == digest
    assert captured["body"]["receipt_type"] == (
        "protectmcp:lifecycle:configuration_change"
    )


def test_lifecycle_without_configuration_change_does_not_require_digest() -> None:
    """The rule-9 guard is scoped to the ``:configuration_change`` suffix.
    Plain ``protectmcp:lifecycle`` receipts remain digest-optional."""
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
            policy_decision="none",
        )
    assert captured["body"]["receipt_type"] == "protectmcp:lifecycle"
    assert "config_manifest_digest" not in captured["body"]
