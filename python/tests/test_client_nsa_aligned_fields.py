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
    RECEIPT_TYPE_NAMESPACE,
    Agent,
    _compute_config_manifest_digest,
    _compute_cve_inventory_digest,
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
    """JCS-canonical input -> stable 32 bare hex chars."""
    fp1 = _compute_tool_fingerprint("search", {"a": 1, "b": 2})
    fp2 = _compute_tool_fingerprint("search", {"b": 2, "a": 1})
    assert fp1 == fp2
    assert re.fullmatch(r"[0-9a-f]{32}", fp1)


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
    assert re.fullmatch(r"[0-9a-f]{32}", fp)


def test_tool_fingerprint_caller_value_wins() -> None:
    """Explicit ``tool_fingerprint`` overrides auto-derivation."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    explicit = "a" * 32
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
        ("config_manifest_digest", "sha256:" + "c" * 64),
        ("cve_inventory_digest", "sha256:" + "d" * 64),
    ],
)
def test_caller_digest_fields_forwarded_verbatim(field: str, value: str) -> None:
    """The SDK passes ``result_digest``,
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
            "configuration_change_missing_config_manifest_digest: "
            "receipt_type=protectmcp:lifecycle:configuration_change "
            "requires config_manifest_digest (sha256:<64 hex>)."
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


# === Rule 10: expiry_collision_guard (valid_seconds vs expires_at) ===


class TestExpiryCollisionGuard:
    """Rule 10: ``valid_seconds`` and ``expires_at`` are mutually exclusive."""

    def test_both_set_raises_verbatim(self) -> None:
        """Passing both knobs raises ``expiry_collision_guard`` before the
        HTTP roundtrip with a verbatim message in lockstep with the cloud."""
        with patch("asqav.client._post") as p:
            with pytest.raises(ValueError) as exc_info:
                _agent().sign(
                    "api:call",
                    {"k": "v"},
                    compliance_mode=True,
                    valid_seconds=3600,
                    expires_at="2026-06-01T00:00:00Z",
                )
            assert str(exc_info.value) == (
                "expiry_collision_guard: pass either valid_seconds or "
                "expires_at, not both (rule 10)"
            )
            p.assert_not_called()

    def test_neither_set_uses_server_default(self) -> None:
        """Passing neither knob is fine; the cloud applies its default."""
        captured: dict = {}

        def fake_post(path: str, body: dict) -> dict:
            captured["body"] = body
            return _ok_response()

        with patch("asqav.client._post", side_effect=fake_post):
            _agent().sign("api:call", {"k": "v"}, compliance_mode=True)
        assert "valid_seconds" not in captured["body"]
        assert "expires_at" not in captured["body"]

    def test_only_valid_seconds_set_passes(self) -> None:
        """``valid_seconds`` alone is forwarded verbatim."""
        captured: dict = {}

        def fake_post(path: str, body: dict) -> dict:
            captured["body"] = body
            return _ok_response()

        with patch("asqav.client._post", side_effect=fake_post):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                valid_seconds=7200,
            )
        assert captured["body"]["valid_seconds"] == 7200
        assert "expires_at" not in captured["body"]

    def test_only_expires_at_set_converts_to_valid_seconds(self) -> None:
        """``expires_at`` alone is converted client-side to ``valid_seconds``
        and never sent on the wire. The cloud reverted its ``expires_at``
        field (extra=ignore), so the SDK owns the absolute->duration
        conversion to avoid a silent-drop footgun."""
        captured: dict = {}

        def fake_post(path: str, body: dict) -> dict:
            captured["body"] = body
            return _ok_response()

        # Far-future ISO horizon -> a large positive duration.
        with patch("asqav.client._post", side_effect=fake_post):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                expires_at="2099-06-01T00:00:00Z",
            )
        assert "expires_at" not in captured["body"]
        assert captured["body"]["valid_seconds"] > 0

    def test_expires_at_epoch_float_converts_to_valid_seconds(self) -> None:
        """A POSIX-epoch ``expires_at`` converts to a duration too."""
        import time as _time

        captured: dict = {}

        def fake_post(path: str, body: dict) -> dict:
            captured["body"] = body
            return _ok_response()

        with patch("asqav.client._post", side_effect=fake_post):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                expires_at=_time.time() + 3600,
            )
        assert "expires_at" not in captured["body"]
        # ~3600s horizon, allow a small clock delta.
        assert 3590 <= captured["body"]["valid_seconds"] <= 3601


# === Rule 11: digest_format_guard (sha256:<64-hex>) ===


_VALID_SHA = "sha256:" + "f" * 64


class TestDigestFormatGuard:
    """Rule 11: every caller-supplied digest MUST match ``sha256:<64-hex>``."""

    @pytest.mark.parametrize(
        "field",
        ["config_manifest_digest", "cve_inventory_digest"],
    )
    def test_valid_format_passes(self, field: str) -> None:
        """A correctly-formatted ``sha256:<64-hex>`` digest is forwarded
        verbatim. The cross-field validator for configuration_change still
        requires the digest to be present, so we send a plain decision
        receipt here."""
        captured: dict = {}

        def fake_post(path: str, body: dict) -> dict:
            captured["body"] = body
            return _ok_response()

        with patch("asqav.client._post", side_effect=fake_post):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                **{field: _VALID_SHA},
            )
        assert captured["body"][field] == _VALID_SHA

    def test_tool_fingerprint_valid_32_hex_passes(self) -> None:
        """A 32 bare lowercase hex ``tool_fingerprint`` is forwarded
        verbatim. The cloud validates this field with the bare form
        (SHA-256[:32]), not the self-describing ``sha256:<hex>`` form."""
        captured: dict = {}

        def fake_post(path: str, body: dict) -> dict:
            captured["body"] = body
            return _ok_response()

        fp = "a" * 32
        with patch("asqav.client._post", side_effect=fake_post):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                tool_fingerprint=fp,
            )
        assert captured["body"]["tool_fingerprint"] == fp

    @pytest.mark.parametrize(
        "field",
        ["config_manifest_digest", "cve_inventory_digest"],
    )
    def test_wrong_prefix_rejected(self, field: str) -> None:
        """Anything other than the ``sha256:`` prefix raises the verbatim cloud token."""
        with patch("asqav.client._post") as p:
            with pytest.raises(ValueError) as exc_info:
                _agent().sign(
                    "api:call",
                    {"k": "v"},
                    compliance_mode=True,
                    **{field: "md5:" + "0" * 64},
                )
            assert str(exc_info.value) == (
                f"{field}_not_sha256_wire_form: must look like "
                "'sha256:<64 lowercase hex>'."
            )
            p.assert_not_called()

    @pytest.mark.parametrize(
        "field",
        ["config_manifest_digest", "cve_inventory_digest"],
    )
    def test_wrong_length_rejected(self, field: str) -> None:
        """A short hex tail fails the regex."""
        with patch("asqav.client._post") as p:
            with pytest.raises(ValueError) as exc_info:
                _agent().sign(
                    "api:call",
                    {"k": "v"},
                    compliance_mode=True,
                    **{field: "sha256:abc123"},
                )
            assert str(exc_info.value) == (
                f"{field}_not_sha256_wire_form: must look like "
                "'sha256:<64 lowercase hex>'."
            )
            p.assert_not_called()

    @pytest.mark.parametrize(
        "field",
        ["config_manifest_digest", "cve_inventory_digest"],
    )
    def test_non_hex_rejected(self, field: str) -> None:
        """Uppercase / non-hex characters fail the regex."""
        with patch("asqav.client._post") as p:
            with pytest.raises(ValueError) as exc_info:
                _agent().sign(
                    "api:call",
                    {"k": "v"},
                    compliance_mode=True,
                    **{field: "sha256:" + "Z" * 64},
                )
            assert str(exc_info.value) == (
                f"{field}_not_sha256_wire_form: must look like "
                "'sha256:<64 lowercase hex>'."
            )
            p.assert_not_called()

    @pytest.mark.parametrize(
        "bad",
        [
            "sha256:" + "a" * 64,  # self-describing form is now rejected
            "a" * 31,  # too short
            "a" * 33,  # too long
            "Z" * 32,  # non-hex / uppercase
        ],
    )
    def test_tool_fingerprint_non_32_hex_rejected(self, bad: str) -> None:
        """tool_fingerprint must be 32 bare lowercase hex chars
        (SHA-256[:32]) to match the cloud wire form. The verbatim token
        mirrors the cloud message ``tool_fingerprint_not_32_hex_chars``."""
        with patch("asqav.client._post") as p:
            with pytest.raises(ValueError) as exc_info:
                _agent().sign(
                    "api:call",
                    {"k": "v"},
                    compliance_mode=True,
                    tool_fingerprint=bad,
                )
            assert str(exc_info.value) == (
                "tool_fingerprint_not_32_hex_chars: must be 32 lowercase "
                "hex chars (SHA-256[:32])."
            )
            p.assert_not_called()


class TestDigestHelpers:
    """Helpers always emit a digest that passes the rule-11 regex."""

    def test_compute_tool_fingerprint_matches_regex(self) -> None:
        fp = _compute_tool_fingerprint("search", {"q": "string"})
        assert re.fullmatch(r"[0-9a-f]{32}", fp)

    def test_compute_config_manifest_digest_is_deterministic(self) -> None:
        d1 = _compute_config_manifest_digest({"a": 1, "b": [2, 3]})
        d2 = _compute_config_manifest_digest({"b": [2, 3], "a": 1})
        assert d1 == d2
        assert re.fullmatch(r"sha256:[a-f0-9]{64}", d1)

    def test_compute_cve_inventory_digest_is_deterministic(self) -> None:
        cves = [{"id": "CVE-2026-0001", "severity": "high"}]
        d1 = _compute_cve_inventory_digest(cves)
        d2 = _compute_cve_inventory_digest([dict(cves[0])])
        assert d1 == d2
        assert re.fullmatch(r"sha256:[a-f0-9]{64}", d1)


# === BLOCKER 5: protectmcp:observation:result_bound receipt type ===


class TestResultBoundReceiptType:
    """The new ``:result_bound`` suffix is a first-class receipt type."""

    def test_result_bound_in_namespace(self) -> None:
        assert "protectmcp:observation:result_bound" in RECEIPT_TYPE_NAMESPACE

    @pytest.mark.parametrize(
        "receipt_type",
        sorted(RECEIPT_TYPE_NAMESPACE),
    )
    def test_every_namespace_member_passes_validation(
        self, receipt_type: str
    ) -> None:
        """Every receipt_type in the namespace must round-trip through
        sign() without tripping ``invalid_receipt_type``."""
        captured: dict = {}

        def fake_post(path: str, body: dict) -> dict:
            captured["body"] = body
            return _ok_response()

        kwargs: dict = {
            "compliance_mode": True,
            "receipt_type": receipt_type,
        }
        # The rule-9 guard requires config_manifest_digest for the
        # configuration_change receipt; supply a deterministic digest.
        if receipt_type == "protectmcp:lifecycle:configuration_change":
            kwargs["config_manifest_digest"] = (
                _compute_config_manifest_digest({"mode": "test"})
            )
        # result_bound requires result_digest, lockstep with cloud rule 9.
        if receipt_type == "protectmcp:observation:result_bound":
            kwargs["result_digest"] = "sha256:" + "0" * 64
        # Risk-acceptance requires approver_id + acceptance_reason.
        if receipt_type == "protectmcp:lifecycle:risk_acceptance":
            kwargs["approver_id"] = "approver:alice@example.com"
            kwargs["acceptance_reason"] = "accepted"
        # Lifecycle receipts use policy_decision=none.
        if receipt_type.startswith("protectmcp:lifecycle") or (
            receipt_type.startswith("protectmcp:observation")
        ):
            kwargs["policy_decision"] = "none"

        with patch("asqav.client._post", side_effect=fake_post):
            _agent().sign("api:call", {"k": "v"}, **kwargs)
        assert captured["body"]["receipt_type"] == receipt_type

    def test_result_bound_carries_result_digest(self) -> None:
        """The headline use case: an observation receipt that binds the
        tool output via ``result_digest``."""
        captured: dict = {}

        def fake_post(path: str, body: dict) -> dict:
            captured["body"] = body
            return _ok_response()

        result_digest = "sha256:" + "9" * 64
        with patch("asqav.client._post", side_effect=fake_post):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                receipt_type="protectmcp:observation:result_bound",
                policy_decision="none",
                result_digest=result_digest,
            )
        assert captured["body"]["receipt_type"] == (
            "protectmcp:observation:result_bound"
        )
        assert captured["body"]["result_digest"] == result_digest
