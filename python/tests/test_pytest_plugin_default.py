"""Pin the pytest plugin's default framework key to one that exists in FRAMEWORKS."""

from __future__ import annotations

import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav import pytest_plugin
from asqav.compliance import FRAMEWORKS
from asqav.pytest_plugin import _PluginState, make_bundle_from_report

EXPECTED_DEFAULT = "eu_ai_act"


def test_plugin_state_default_framework_is_eu_ai_act() -> None:
    """The dataclass default applies before any pytest_configure hook fires."""
    state = _PluginState()
    assert state.framework == EXPECTED_DEFAULT


def test_plugin_state_default_is_a_known_framework() -> None:
    """Whatever the default is, it must resolve in FRAMEWORKS."""
    state = _PluginState()
    assert state.framework in FRAMEWORKS


def test_make_bundle_from_report_default_framework() -> None:
    """The programmatic helper mirrors the plugin's default."""
    sig = inspect.signature(make_bundle_from_report)
    assert sig.parameters["framework"].default == EXPECTED_DEFAULT


def test_addoption_default_matches_state_default() -> None:
    """`pytest --asqav` with no --asqav-framework gets the same key."""

    captured: dict[str, str] = {}

    class _Group:
        def addoption(self, *args, **kwargs):
            for arg in args:
                if isinstance(arg, str) and arg.startswith("--asqav"):
                    captured[arg] = kwargs.get("default")
                    break

    class _Parser:
        def getgroup(self, *args, **kwargs):
            return _Group()

    pytest_plugin.pytest_addoption(_Parser())  # type: ignore[arg-type]

    assert captured["--asqav-framework"] == EXPECTED_DEFAULT
