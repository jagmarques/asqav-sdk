"""Tests for pluggable detector framework (criterion 331).

Non-vacuity mapping (mutant → what breaks it):
  - Remove DetectorBlockedError raise in run_detectors → deny/fail-closed tests fail.
  - Remove _detectors stamp into context → allow/receipt-records tests fail.
  - Remove fail_open branch in run_detectors → fail_open tests fail.
  - Remove _registry guard → additive test fails (unexpected _detectors key).

Each test asserts on the specific mechanism, not just the outcome.
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav._detectors import (
    DetectorBlockedError,
    DetectorPlugin,
    DetectorResult,
    clear_detectors,
    register_detector,
    run_detectors,
)
from asqav.client import Agent


# ─── Shared helpers ───────────────────────────────────────────────────────────


def _agent() -> Agent:
    return Agent(
        agent_id="agent_det_test",
        name="detector-test-agent",
        public_key="pk_xxx",
        key_id="key_det",
        algorithm="ML-DSA-65",
        capabilities=["api:*"],
        created_at=1700000000.0,
    )


def _ok_response(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    base: dict[str, Any] = {
        "signature": "sig_b64",
        "signature_id": "sig_det",
        "action_id": "act_det",
        "timestamp": 1700000000.0,
        "verification_url": "https://asqav.com/verify/sig_det",
        "algorithm": "ML-DSA-65",
        "chain_hash": "deadbeef",
    }
    if extra:
        base.update(extra)
    return base


class AllowDetector:
    name = "allow-detector"

    def inspect(self, action_type: str, context: dict) -> DetectorResult:
        return DetectorResult(allow=True, confidence=0.9, labels=["ok"], reason="all clear")


class DenyDetector:
    name = "deny-detector"

    def inspect(self, action_type: str, context: dict) -> DetectorResult:
        return DetectorResult(
            allow=False,
            confidence=1.0,
            labels=["pii", "ssn"],
            reason="SSN detected",
        )


class RaisingDetector:
    name = "raising-detector"

    def inspect(self, action_type: str, context: dict) -> DetectorResult:
        raise RuntimeError("detector_internal_error")


@pytest.fixture(autouse=True)
def _reset_detectors():
    """Ensure each test starts with a clean detector registry."""
    clear_detectors()
    yield
    clear_detectors()


# ─── Unit: run_detectors ──────────────────────────────────────────────────────


def test_no_detectors_returns_empty_list() -> None:
    records = run_detectors("api:call", {"x": 1})
    assert records == []


def test_allow_detector_returns_record() -> None:
    register_detector(AllowDetector())
    records = run_detectors("api:call", {"x": 1})
    assert len(records) == 1
    assert records[0]["allow"] is True
    assert records[0]["detector"] == "allow-detector"


def test_deny_detector_raises_detector_blocked_error() -> None:
    register_detector(DenyDetector())
    with pytest.raises(DetectorBlockedError) as exc_info:
        run_detectors("api:call", {"ssn": "123-45-6789"})
    err = exc_info.value
    assert err.detector_name == "deny-detector"
    assert err.result.allow is False
    assert "pii" in err.result.labels
    assert "detector_blocked" in str(err)


def test_failing_detector_fail_closed_by_default() -> None:
    register_detector(RaisingDetector())
    with pytest.raises(DetectorBlockedError) as exc_info:
        run_detectors("api:call", {})
    err = exc_info.value
    assert err.detector_name == "raising-detector"
    assert "detector_error_blocked" in str(err)


def test_failing_detector_fail_open_proceeds() -> None:
    register_detector(RaisingDetector(), fail_open=True)
    records = run_detectors("api:call", {})
    assert len(records) == 1
    assert records[0]["allow"] is True  # synthetic allow=True error record
    assert "fail-open" in records[0]["reason"]


def test_multiple_detectors_all_run_on_allow() -> None:
    register_detector(AllowDetector())
    register_detector(AllowDetector())
    records = run_detectors("api:call", {})
    assert len(records) == 2


def test_deny_stops_after_first_deny() -> None:
    """Second detector must NOT run after first returns deny."""
    called: list[str] = []

    class TrackingAllow:
        name = "track-allow"

        def inspect(self, action_type: str, context: dict) -> DetectorResult:
            called.append("allow")
            return DetectorResult(allow=True)

    class TrackingDeny:
        name = "track-deny"

        def inspect(self, action_type: str, context: dict) -> DetectorResult:
            called.append("deny")
            return DetectorResult(allow=False, reason="blocked")

    register_detector(TrackingDeny())
    register_detector(TrackingAllow())

    with pytest.raises(DetectorBlockedError):
        run_detectors("api:call", {})

    assert called == ["deny"]  # allow never ran


def test_none_context_is_treated_as_empty() -> None:
    register_detector(AllowDetector())
    records = run_detectors("api:call", None)
    assert len(records) == 1


# ─── Integration: Agent.sign path ─────────────────────────────────────────────


def test_allow_detector_lets_sign_proceed() -> None:
    """Allow detector: sign POST is made AND _detectors appears in context."""
    captured: dict[str, Any] = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    register_detector(AllowDetector())
    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign("api:call", {"model": "gpt-4"}, compliance_mode=False)

    assert captured, "sign() POST was never called: allow detector blocked sign"
    # _detectors must appear in the signed body context.
    body = captured["body"]
    ctx = body.get("context") or {}
    assert "_detectors" in ctx, "_detectors key missing from signed body context"
    records = ctx["_detectors"]
    assert len(records) == 1
    assert records[0]["allow"] is True
    assert records[0]["detector"] == "allow-detector"


def test_deny_detector_blocks_sign_no_post() -> None:
    """Deny detector: sign raises DetectorBlockedError; no POST is made."""
    post_called = False

    def fake_post(path: str, body: dict) -> dict:
        nonlocal post_called
        post_called = True
        return _ok_response()

    register_detector(DenyDetector())
    with patch("asqav.client._post", side_effect=fake_post):
        with pytest.raises(DetectorBlockedError) as exc_info:
            _agent().sign("api:call", {"ssn": "123-45-6789"}, compliance_mode=False)

    assert not post_called, "sign() POST must NOT be called when a detector denies"
    assert exc_info.value.detector_name == "deny-detector"


def test_fail_closed_detector_error_blocks_sign() -> None:
    """A detector that raises blocks sign by default (fail-closed)."""
    post_called = False

    def fake_post(path: str, body: dict) -> dict:
        nonlocal post_called
        post_called = True
        return _ok_response()

    register_detector(RaisingDetector())
    with patch("asqav.client._post", side_effect=fake_post):
        with pytest.raises(DetectorBlockedError):
            _agent().sign("api:call", {}, compliance_mode=False)

    assert not post_called


def test_fail_open_detector_error_allows_sign() -> None:
    """A detector that raises with fail_open=True lets sign proceed."""
    captured: dict[str, Any] = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    register_detector(RaisingDetector(), fail_open=True)
    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign("api:call", {"x": 1}, compliance_mode=False)

    assert captured, "sign() POST should proceed with fail_open=True"
    body = captured["body"]
    ctx = body.get("context") or {}
    # Error record should appear in context under _detectors.
    assert "_detectors" in ctx
    assert ctx["_detectors"][0]["labels"] == ["error"]


def test_additive_no_detector_body_unchanged() -> None:
    """With no detectors, the signed body must NOT contain _detectors key."""
    captured: dict[str, Any] = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign("api:call", {"model": "gpt-4"}, compliance_mode=False)

    assert captured
    body = captured["body"]
    ctx = body.get("context") or {}
    assert "_detectors" not in ctx, (
        "_detectors key must not appear when no detectors are registered"
    )


# ─── Unit: DetectorPlugin protocol ────────────────────────────────────────────


def test_detector_plugin_protocol_structural() -> None:
    """Any object with name + inspect satisfies DetectorPlugin (structural)."""
    assert isinstance(AllowDetector(), DetectorPlugin)


def test_detector_blocked_error_attributes() -> None:
    result = DetectorResult(
        allow=False, labels=["pii"], detector="test-det", reason="blocked"
    )
    err = DetectorBlockedError("blocked", result)
    assert err.detector_name == "test-det"
    assert err.result is result
    assert err.docs_url.startswith("https://")


# ─── Reference detector: PresidioDetector ────────────────────────────────────


def test_presidio_detector_labels_on_deny() -> None:
    """PresidioDetector returns deny + labels when Presidio finds PII."""
    from asqav.extras.presidio_detector import PresidioDetector

    class FakeResult:
        def __init__(self, entity_type: str, score: float) -> None:
            self.entity_type = entity_type
            self.score = score

    class FakeEngine:
        def analyze(self, **kwargs: Any) -> list:
            return [FakeResult("US_SSN", 0.9), FakeResult("PERSON", 0.85)]

    det = PresidioDetector(threshold=0.5)
    det._engine = FakeEngine()  # bypass lazy import

    result = det.inspect("api:call", {"data": "SSN: 123-45-6789"})
    assert result.allow is False
    assert "US_SSN" in result.labels
    assert result.confidence >= 0.85


def test_presidio_detector_allow_when_no_pii() -> None:
    from asqav.extras.presidio_detector import PresidioDetector

    class FakeEngine:
        def analyze(self, **kwargs: Any) -> list:
            return []  # no PII found

    det = PresidioDetector()
    det._engine = FakeEngine()

    result = det.inspect("api:call", {"data": "hello world"})
    assert result.allow is True


def test_presidio_detector_import_error_without_extra() -> None:
    """Without presidio-analyzer installed, PresidioDetector raises ImportError."""
    from asqav.extras.presidio_detector import PresidioDetector, _load_analyzer

    with patch.dict("sys.modules", {"presidio_analyzer": None}):
        with pytest.raises(ImportError, match="presidio-analyzer"):
            _load_analyzer()


# ─── Reference detector: OpaDetector ─────────────────────────────────────────


def test_opa_detector_allow_on_true_result() -> None:
    from asqav.extras.opa_detector import OpaDetector

    det = OpaDetector(opa_url="http://fake-opa:8181", policy_path="test/allow")

    def fake_post_json(payload: dict) -> dict:
        return {"result": True}

    det._post_json = fake_post_json  # type: ignore[method-assign]
    result = det.inspect("api:call", {"model": "gpt-4"})
    assert result.allow is True


def test_opa_detector_deny_on_false_result() -> None:
    from asqav.extras.opa_detector import OpaDetector

    det = OpaDetector(opa_url="http://fake-opa:8181", policy_path="test/allow")

    def fake_post_json(payload: dict) -> dict:
        return {"result": False}

    det._post_json = fake_post_json  # type: ignore[method-assign]
    result = det.inspect("api:call", {"model": "gpt-4"})
    assert result.allow is False
    assert "opa_deny" in result.labels


def test_opa_detector_deny_on_missing_result() -> None:
    """Missing 'result' key in OPA response → fail-closed deny."""
    from asqav.extras.opa_detector import OpaDetector

    det = OpaDetector()

    def fake_post_json(payload: dict) -> dict:
        return {}  # no 'result' key

    det._post_json = fake_post_json  # type: ignore[method-assign]
    result = det.inspect("api:call", {})
    assert result.allow is False


def test_opa_detector_raises_on_http_error() -> None:
    """HTTP failure from OPA propagates as RuntimeError (caller handles)."""
    from asqav.extras.opa_detector import OpaDetector

    det = OpaDetector()

    def fake_post_json(payload: dict) -> dict:
        raise ConnectionError("refused")

    det._post_json = fake_post_json  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="opa_request_failed"):
        det.inspect("api:call", {})


def test_opa_sends_correct_payload() -> None:
    """OpaDetector sends {input: {action_type, context}} to OPA."""
    from asqav.extras.opa_detector import OpaDetector

    sent: dict[str, Any] = {}
    det = OpaDetector(opa_url="http://opa:8181", policy_path="my/policy")

    def fake_post_json(payload: dict) -> dict:
        sent.update(payload)
        return {"result": True}

    det._post_json = fake_post_json  # type: ignore[method-assign]
    det.inspect("tool:call", {"model": "x"})

    assert sent["input"]["action_type"] == "tool:call"
    assert sent["input"]["context"] == {"model": "x"}
