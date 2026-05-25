"""Pytest plugin that signs each test outcome and emits a ComplianceBundle.

Activate with ``pytest --asqav --asqav-agent=test-runner``; ``ASQAV_API_KEY``
must be set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class CapturedOutcome:
    """One test outcome captured for the bundle."""

    nodeid: str
    outcome: str  # "passed" | "failed" | "skipped"
    duration: float
    longrepr: str | None = None


@dataclass
class _PluginState:
    """In-memory state for one pytest invocation."""

    enabled: bool = False
    agent_name: str = "test-runner"
    output_path: str = "audit-bundle.json"
    framework: str = "eu_ai_act"
    results: list[CapturedOutcome] = field(default_factory=list)
    signatures: list[Any] = field(default_factory=list)
    agent: Any = None


_state = _PluginState()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("asqav", "Asqav governance plugin")
    group.addoption(
        "--asqav",
        action="store_true",
        default=False,
        help="Enable asqav: sign each test result and write a compliance bundle.",
    )
    group.addoption(
        "--asqav-agent",
        action="store",
        default="test-runner",
        help="Agent name to use when signing test results (default: test-runner).",
    )
    group.addoption(
        "--asqav-output",
        action="store",
        default="audit-bundle.json",
        help="Where to write the compliance bundle (default: audit-bundle.json).",
    )
    group.addoption(
        "--asqav-framework",
        action="store",
        default="eu_ai_act",
        help="Compliance framework key (default: eu_ai_act). See `asqav compliance frameworks`.",
    )


def pytest_configure(config: pytest.Config) -> None:
    if not config.getoption("--asqav"):
        return

    import os

    if not os.environ.get("ASQAV_API_KEY"):
        # Configuration error, not a test failure. Exit early so the rest of
        # the run does not partially sign with a half-set-up agent.
        raise pytest.UsageError(
            "--asqav requires ASQAV_API_KEY in the environment. "
            "Get one at https://asqav.com."
        )

    import asqav

    asqav.init()

    _state.enabled = True
    _state.agent_name = config.getoption("--asqav-agent")
    _state.output_path = config.getoption("--asqav-output")
    _state.framework = config.getoption("--asqav-framework")
    _state.agent = asqav.Agent.create(_state.agent_name)
    _state.results = []
    _state.signatures = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # type: ignore[no-untyped-def]
    """Capture each test phase outcome. We only sign the 'call' phase."""
    outcome = yield
    if not _state.enabled:
        return

    report = outcome.get_result()
    if report.when != "call":
        return

    longrepr: str | None = None
    if report.longrepr is not None:
        longrepr = str(report.longrepr)[:2000]  # cap to keep payloads sane

    result = CapturedOutcome(
        nodeid=report.nodeid,
        outcome=report.outcome,
        duration=report.duration,
        longrepr=longrepr,
    )
    _state.results.append(result)

    if _state.agent is None:
        return

    try:
        sig = _state.agent.sign(
            "test:result",
            context={
                "nodeid": result.nodeid,
                "outcome": result.outcome,
                "duration": result.duration,
                "framework": _state.framework,
            },
        )
        _state.signatures.append(sig)
    except Exception as exc:
        # A failed signature should not mask the test failure.
        report.sections.append(("asqav", f"sign failed: {exc}"))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _state.enabled or not _state.signatures:
        return

    from asqav.compliance import export_bundle

    bundle = export_bundle(_state.signatures, framework=_state.framework)
    bundle.to_file(_state.output_path)
    print(
        f"\n[asqav] wrote {bundle.receipt_count} signed test results to "
        f"{_state.output_path} (merkle_root={bundle.merkle_root})"
    )


# === Programmatic API ===


def make_bundle_from_report(
    signatures: list[Any], *, framework: str = "eu_ai_act"
) -> Any:
    """Build a :class:`ComplianceBundle` from pre-signed test results."""
    from asqav.compliance import export_bundle

    return export_bundle(signatures, framework=framework)


# Test-only escape hatches so test_pytest_plugin.py can manipulate plugin state
# without spawning a real subprocess.
def _reset_state_for_tests() -> None:
    global _state
    _state = _PluginState()


def _get_state_for_tests() -> _PluginState:
    return _state
