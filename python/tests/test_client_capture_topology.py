"""SDK parity for `capture_topology` + `observation` decision vocabulary.

Covers the cycle-12/13 cloud + IETF -04 deliverables: the wire `decision`
vocabulary gains `observation` (paired with internal `policy_decision="none"`
on `protectmcp:lifecycle` receipts) and the producer-side `capture_topology`
metadata travels on `SignRequest` as a closed Literal of five values.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav.client import (
    CAPTURE_TOPOLOGY_NAMESPACE,
    DECISION_MAP,
    RECEIPT_TYPE_NAMESPACE,
    Agent,
    _map_policy_decision_to_decision,
)


def _agent() -> Agent:
    return Agent(
        agent_id="agent_capture",
        name="capture-agent",
        public_key="pk_xxx",
        key_id="key_capture",
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


# === observation decision + policy_decision="none" ===


def test_decision_map_includes_none_to_observation() -> None:
    """The lifecycle opt-out token maps to the wire `observation` decision."""
    assert DECISION_MAP["none"] == "observation"


def test_map_policy_decision_translates_none() -> None:
    """`_map_policy_decision_to_decision('none')` returns `observation`."""
    assert _map_policy_decision_to_decision("none") == "observation"


def test_decision_vocabulary_includes_observation() -> None:
    """A lifecycle receipt with `policy_decision='none'` ships
    `decision='observation'` on the outgoing JSON body."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response({
            "compliance_mode": True,
            "policy_decision": "none",
            "decision": "observation",
        })

    with patch("asqav.client._post", side_effect=fake_post):
        resp = _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            receipt_type="protectmcp:lifecycle",
            policy_decision="none",
        )

    body = captured["body"]
    assert body["policy_decision"] == "none"
    assert body["receipt_type"] == "protectmcp:lifecycle"
    assert resp.decision == "observation"


def test_observation_round_trips_on_response_without_cloud_field() -> None:
    """An older cloud may omit `decision`; the SDK maps `none` locally."""
    extra = {
        "compliance_mode": True,
        "policy_decision": "none",
        "receipt_type": "protectmcp:lifecycle",
    }
    with patch("asqav.client._post", return_value=_ok_response(extra)):
        resp = _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            receipt_type="protectmcp:lifecycle",
            policy_decision="none",
        )
    assert resp.decision == "observation"


# === capture_topology vocabulary ===


def test_capture_topology_namespace_matches_cloud_literal() -> None:
    """SDK namespace mirrors the cloud SignRequest Literal exactly."""
    assert CAPTURE_TOPOLOGY_NAMESPACE == frozenset(
        {
            "in_process_sdk",
            "network_proxy",
            "browser_extension",
            "ebpf_observer",
            "mcp_proxy",
            "passive_telemetry",
        }
    )


def test_capture_topology_is_re_exported_at_top_level() -> None:
    """`asqav.CAPTURE_TOPOLOGY_NAMESPACE` is part of the public surface."""
    assert asqav.CAPTURE_TOPOLOGY_NAMESPACE == CAPTURE_TOPOLOGY_NAMESPACE


@pytest.mark.parametrize("value", sorted(CAPTURE_TOPOLOGY_NAMESPACE))
def test_capture_topology_passes_through(value: str) -> None:
    """All five canonical values are forwarded verbatim on the body."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            capture_topology=value,
        )
    assert captured["body"]["capture_topology"] == value


def test_capture_topology_invalid_value_rejected() -> None:
    """A value outside the closed Literal is rejected before the HTTP call."""
    with patch("asqav.client._post") as p:
        with pytest.raises(ValueError, match="invalid_capture_topology"):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                capture_topology="rootkit",
            )
        p.assert_not_called()


def test_capture_topology_absent_when_not_set() -> None:
    """No body key when the caller does not pass the kwarg."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
        )
    assert "capture_topology" not in captured["body"]


def test_capture_topology_dropped_outside_compliance_mode() -> None:
    """Compliance-mode-off requests omit `capture_topology` to preserve
    the non-compliance body shape used in lightweight self-hosted setups."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=False,
            capture_topology="mcp_proxy",
        )
    assert "capture_topology" not in captured["body"]


# === parametrised vocabulary parity (capture_topology + receipt_type) ===


@pytest.mark.parametrize("value", sorted(CAPTURE_TOPOLOGY_NAMESPACE))
def test_capture_topology_parametrised_accepts_all_values(value: str) -> None:
    """Every value in the 6-token vocabulary is forwarded on the body."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            capture_topology=value,
            receipt_type=(
                "protectmcp:observation"
                if value == "passive_telemetry"
                else "protectmcp:decision"
            ),
        )
    assert captured["body"]["capture_topology"] == value


@pytest.mark.parametrize(
    "bad_value", ["bogus", "in-process-sdk", "PASSIVE_TELEMETRY", " "]
)
def test_capture_topology_parametrised_rejects_unknown(bad_value: str) -> None:
    """Any value outside the closed vocabulary is rejected client-side."""
    with patch("asqav.client._post") as p:
        with pytest.raises(ValueError, match="invalid_capture_topology"):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                capture_topology=bad_value,
            )
        p.assert_not_called()


@pytest.mark.parametrize("value", sorted(RECEIPT_TYPE_NAMESPACE))
def test_receipt_type_parametrised_accepts_all_values(value: str) -> None:
    """Every token in the receipt_type vocabulary is accepted client-side."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    kwargs: dict = {
        "compliance_mode": True,
        "receipt_type": value,
    }
    # Lifecycle namespace pairs with `none` to exercise the opt-out path.
    if value.startswith("protectmcp:lifecycle"):
        kwargs["policy_decision"] = "none"
    # Rule 9 lockstep: configuration_change receipts MUST carry config_manifest_digest.
    if value == "protectmcp:lifecycle:configuration_change":
        kwargs["config_manifest_digest"] = "sha256:" + "0" * 64
    # Rule 9 lockstep: result_bound receipts MUST carry result_digest.
    if value == "protectmcp:observation:result_bound":
        kwargs["result_digest"] = "sha256:" + "0" * 64
    # Risk-acceptance receipts MUST carry approver_id + acceptance_reason.
    if value == "protectmcp:lifecycle:risk_acceptance":
        kwargs["approver_id"] = "approver:alice@example.com"
        kwargs["acceptance_reason"] = "accepted"

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign("api:call", {"k": "v"}, **kwargs)
    assert captured["body"]["receipt_type"] == value


@pytest.mark.parametrize(
    "bad_value",
    ["protectmcp:bogus", "protectmcp:Observation", "decision", "observation"],
)
def test_receipt_type_parametrised_rejects_unknown(bad_value: str) -> None:
    """Any token outside the closed receipt-type vocabulary is rejected."""
    with patch("asqav.client._post") as p:
        with pytest.raises(ValueError, match="invalid_receipt_type"):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                receipt_type=bad_value,
            )
        p.assert_not_called()


def test_passive_telemetry_with_decision_receipt_rejected() -> None:
    """Client-side mirror of cloud rule 8: passive_telemetry +
    protectmcp:decision fails before the HTTP roundtrip."""
    with patch("asqav.client._post") as p:
        with pytest.raises(ValueError, match="false_attestation_guard"):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                capture_topology="passive_telemetry",
                receipt_type="protectmcp:decision",
            )
        p.assert_not_called()


class TestFalseAttestationGuardWidened:
    """Rule 8: passive_telemetry pairs ONLY with protectmcp:observation.

    Pins the widened guard (PR #219) against silent regression for every
    rejected receipt_type. The single-case decision test above is retained
    for symmetry with the original cycle-26 fix.
    """

    @pytest.mark.parametrize(
        "receipt_type",
        [
            "protectmcp:decision",
            "protectmcp:restraint",
            "protectmcp:lifecycle",
            "protectmcp:lifecycle:configuration_change",
            "protectmcp:acknowledgment",
        ],
    )
    def test_passive_telemetry_rejects_non_observation(
        self, receipt_type: str
    ) -> None:
        """Each non-observation receipt_type raises ValueError carrying
        the verbatim false_attestation_guard prefix and the (rule 8)
        suffix. The HTTP layer must not be reached."""
        offending = receipt_type.split(":", 1)[-1]
        with patch("asqav.client._post") as p:
            with pytest.raises(ValueError) as exc_info:
                _agent().sign(
                    "api:call",
                    {"k": "v"},
                    compliance_mode=True,
                    capture_topology="passive_telemetry",
                    receipt_type=receipt_type,
                )
            msg = str(exc_info.value)
            assert msg.startswith("false_attestation_guard:")
            assert "(rule 8)" in msg
            assert f"not :{offending}" in msg
            p.assert_not_called()

    def test_passive_telemetry_accepts_observation(self) -> None:
        """The matched pair passes the SDK gate (no exception before
        the HTTP roundtrip)."""
        captured: dict = {}

        def fake_post(path: str, body: dict) -> dict:
            captured["body"] = body
            return _ok_response()

        with patch("asqav.client._post", side_effect=fake_post):
            _agent().sign(
                "api:call",
                {"k": "v"},
                compliance_mode=True,
                capture_topology="passive_telemetry",
                receipt_type="protectmcp:observation",
            )
        assert captured["body"]["capture_topology"] == "passive_telemetry"
        assert captured["body"]["receipt_type"] == "protectmcp:observation"


def test_passive_telemetry_with_observation_receipt_accepted() -> None:
    """The matched pair (passive_telemetry + :observation) passes through."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            capture_topology="passive_telemetry",
            receipt_type="protectmcp:observation",
        )
    assert captured["body"]["capture_topology"] == "passive_telemetry"
    assert captured["body"]["receipt_type"] == "protectmcp:observation"
