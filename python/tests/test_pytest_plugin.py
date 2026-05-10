"""Tests for the asqav pytest plugin."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from asqav.pytest_plugin import (
    CapturedOutcome,
    _get_state_for_tests,
    _reset_state_for_tests,
    make_bundle_from_report,
    pytest_configure,
    pytest_runtest_makereport,
    pytest_sessionfinish,
)

assert CapturedOutcome  # imported for the public surface


@pytest.fixture(autouse=True)
def _reset() -> None:
    _reset_state_for_tests()


def _config(**opts) -> MagicMock:  # type: ignore[no-untyped-def]
    cfg = MagicMock()
    defaults = {
        "--asqav": False,
        "--asqav-agent": "test-runner",
        "--asqav-output": "audit-bundle.json",
        "--asqav-framework": "eu_ai_act",
    }
    defaults.update(opts)
    cfg.getoption.side_effect = lambda key: defaults[key]
    return cfg


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_disabled_by_default() -> None:
    """The plugin is a no-op unless --asqav is passed."""
    pytest_configure(_config(**{"--asqav": False}))
    assert _get_state_for_tests().enabled is False


@patch("asqav.Agent.create")
@patch("asqav.init")
def test_configure_enables_plugin(mock_init: MagicMock, mock_create: MagicMock) -> None:
    """--asqav with an API key initializes the agent and enables the plugin."""
    mock_create.return_value = MagicMock(agent_id="agt_test")
    with patch.dict(os.environ, {"ASQAV_API_KEY": "sk_test"}, clear=False):
        pytest_configure(_config(**{
            "--asqav": True,
            "--asqav-agent": "ci-runner",
            "--asqav-output": "out.json",
            "--asqav-framework": "eu_ai_act",
        }))

    state = _get_state_for_tests()
    assert state.enabled is True
    assert state.agent_name == "ci-runner"
    assert state.output_path == "out.json"
    assert state.framework == "eu_ai_act"
    mock_create.assert_called_once_with("ci-runner")


def test_configure_without_api_key_raises() -> None:
    """--asqav without ASQAV_API_KEY exits with a usage error."""
    with patch.dict(os.environ, {"ASQAV_API_KEY": ""}, clear=False):
        with pytest.raises(pytest.UsageError):
            pytest_configure(_config(**{"--asqav": True}))


# ---------------------------------------------------------------------------
# Result capture
# ---------------------------------------------------------------------------


def test_makereport_captures_call_phase() -> None:
    """Only the 'call' phase produces a TestResult; setup/teardown are ignored."""
    state = _get_state_for_tests()
    state.enabled = True
    state.agent = MagicMock()
    state.agent.sign.return_value = MagicMock(signature_id="sig_1")

    report = MagicMock(when="call", nodeid="test.py::test_a",
                      outcome="passed", duration=0.1, longrepr=None,
                      sections=[])

    # Drive the hookwrapper generator with an outcome that yields the report.
    gen = pytest_runtest_makereport(MagicMock(), MagicMock())
    next(gen)
    try:
        gen.send(MagicMock(get_result=lambda: report))
    except StopIteration:
        pass

    assert len(state.results) == 1
    assert state.results[0].nodeid == "test.py::test_a"
    assert state.results[0].outcome == "passed"
    assert len(state.signatures) == 1


def test_makereport_skips_non_call_phases() -> None:
    """Setup and teardown phases are ignored."""
    state = _get_state_for_tests()
    state.enabled = True
    state.agent = MagicMock()

    for phase in ("setup", "teardown"):
        report = MagicMock(when=phase, nodeid="test.py::test_a",
                          outcome="passed", duration=0.0, longrepr=None,
                          sections=[])
        gen = pytest_runtest_makereport(MagicMock(), MagicMock())
        next(gen)
        try:
            gen.send(MagicMock(get_result=lambda r=report: r))
        except StopIteration:
            pass

    assert len(state.results) == 0


def test_makereport_disabled_does_nothing() -> None:
    """When the plugin is disabled the hook is a pass-through."""
    state = _get_state_for_tests()
    state.enabled = False

    report = MagicMock(when="call", nodeid="test.py::test_a",
                      outcome="passed", duration=0.1, longrepr=None,
                      sections=[])
    gen = pytest_runtest_makereport(MagicMock(), MagicMock())
    next(gen)
    try:
        gen.send(MagicMock(get_result=lambda: report))
    except StopIteration:
        pass

    assert len(state.results) == 0
    assert len(state.signatures) == 0


def test_sign_failure_does_not_mask_test() -> None:
    """If sign() raises, the report.sections records the error but no exception bubbles."""
    state = _get_state_for_tests()
    state.enabled = True
    state.agent = MagicMock()
    state.agent.sign.side_effect = RuntimeError("api down")

    report = MagicMock(when="call", nodeid="test.py::test_a",
                      outcome="failed", duration=0.5, longrepr="boom",
                      sections=[])

    gen = pytest_runtest_makereport(MagicMock(), MagicMock())
    next(gen)
    try:
        gen.send(MagicMock(get_result=lambda: report))
    except StopIteration:
        pass

    # No signatures were appended (sign failed) but the section was added.
    assert len(state.signatures) == 0
    assert any("sign failed" in s[1] for s in report.sections)


# ---------------------------------------------------------------------------
# Session finish
# ---------------------------------------------------------------------------


@patch("asqav.compliance.export_bundle")
def test_sessionfinish_writes_bundle(mock_export: MagicMock, tmp_path) -> None:
    """At end of run, a ComplianceBundle is exported to the configured path."""
    mock_bundle = MagicMock(receipt_count=2, merkle_root="root123")
    mock_export.return_value = mock_bundle

    state = _get_state_for_tests()
    state.enabled = True
    state.signatures = [MagicMock(), MagicMock()]
    state.output_path = str(tmp_path / "out.json")
    state.framework = "eu_ai_act"

    pytest_sessionfinish(MagicMock(), 0)

    mock_export.assert_called_once_with(state.signatures, framework="eu_ai_act")
    mock_bundle.to_file.assert_called_once_with(state.output_path)


def test_sessionfinish_noop_when_no_signatures() -> None:
    """Empty signature list is a no-op so the runner does not write garbage."""
    state = _get_state_for_tests()
    state.enabled = True
    state.signatures = []

    # Should not raise even though nothing is patched.
    pytest_sessionfinish(MagicMock(), 0)


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------


@patch("asqav.compliance.export_bundle")
def test_make_bundle_from_report(mock_export: MagicMock) -> None:
    """make_bundle_from_report wraps export_bundle and forwards the framework."""
    mock_export.return_value = MagicMock(receipt_count=3)
    sigs = [MagicMock(), MagicMock(), MagicMock()]
    bundle = make_bundle_from_report(sigs, framework="dora")
    mock_export.assert_called_once_with(sigs, framework="dora")
    assert bundle.receipt_count == 3
