"""Tests for extras packaging and framework stub imports."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _module_installed(name: str) -> bool:
    """True if ``name`` is reachable on the filesystem.

    Pops any pre-existing ``sys.modules`` entry before probing so a mock
    injected by another test (with ``__spec__ = None``) does not mask
    a real on-disk install. Restores the entry afterwards so we do not
    perturb the rest of the test session.
    """
    saved = sys.modules.pop(name, None)
    try:
        spec = importlib.util.find_spec(name)
    except (ValueError, ModuleNotFoundError, ImportError):
        spec = None
    finally:
        if saved is not None:
            sys.modules[name] = saved
    return spec is not None

# -- Packaging tests (no framework deps needed) --


def _read_pyproject() -> dict:
    """Parse pyproject.toml and return optional-dependencies."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml not found"

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(pyproject_path, "rb") as f:
        return tomllib.load(f)


def test_extras_defined_in_pyproject():
    """All 6 framework extras are defined in pyproject.toml optional-dependencies."""
    data = _read_pyproject()
    opt_deps = data["project"]["optional-dependencies"]

    expected = {"langchain", "crewai", "litellm", "haystack", "openai-agents", "cli"}
    for extra in expected:
        assert extra in opt_deps, f"Missing extra: {extra}"


def test_all_extra_includes_everything():
    """The 'all' extra includes deps from all framework extras."""
    data = _read_pyproject()
    opt_deps = data["project"]["optional-dependencies"]

    all_deps = opt_deps["all"]

    # Each framework extra's deps should appear in 'all'
    framework_extras = ["httpx", "cli", "langchain", "crewai", "litellm", "haystack", "openai-agents"]  # noqa: E501
    for extra in framework_extras:
        for dep in opt_deps[extra]:
            # Strip version specifiers for matching
            dep_name = dep.split(">=")[0].split("<=")[0].split("==")[0].strip()
            all_dep_names = [d.split(">=")[0].split("<=")[0].split("==")[0].strip() for d in all_deps]  # noqa: E501
            assert dep_name in all_dep_names, (
                f"Dep '{dep_name}' from extra '{extra}' not found in 'all'"
            )


def test_base_import_no_extras():
    """Base asqav import works without any extras installed."""
    import asqav

    assert hasattr(asqav, "__version__")
    assert hasattr(asqav, "init")


def test_extras_package_importable():
    """The extras package itself is importable."""
    from asqav import extras

    assert extras.__doc__ is not None
    assert "pip install asqav" in extras.__doc__


# -- Stub import tests (framework not installed) --


def _purge(*module_names: str) -> None:
    """Drop modules from sys.modules so a fresh import re-runs the guards.

    Other integration tests inject framework mocks (sys.modules["crewai"] = ...).
    Without this purge the stub re-import sees the mock and skips the
    ImportError path, so the stub-import test silently passes when the user
    actually has the framework missing. Fail closed by clearing both.
    """
    for name in module_names:
        sys.modules.pop(name, None)


@pytest.mark.skipif(_module_installed("langchain_core"), reason="langchain-core installed; stub error path is unreachable in this env")
def test_langchain_stub_import_error():
    """Importing langchain stub raises ImportError when langchain-core is not installed."""
    _purge("asqav.extras.langchain", "langchain_core", "langchain_core.callbacks")
    with pytest.raises(ImportError, match="pip install asqav"):
        import asqav.extras.langchain  # noqa: F401


@pytest.mark.skipif(_module_installed("crewai"), reason="crewai installed; stub error path is unreachable in this env")
def test_crewai_stub_import_error():
    """Importing crewai stub raises ImportError when crewai is not installed."""
    _purge("asqav.extras.crewai", "crewai")
    with pytest.raises(ImportError, match="pip install asqav"):
        import asqav.extras.crewai  # noqa: F401


@pytest.mark.skipif(_module_installed("litellm"), reason="litellm installed; stub error path is unreachable in this env")
def test_litellm_stub_import_error():
    """Importing litellm stub raises ImportError when litellm is not installed."""
    _purge("asqav.extras.litellm", "litellm")
    with pytest.raises(ImportError, match="pip install asqav"):
        import asqav.extras.litellm  # noqa: F401


@pytest.mark.skipif(_module_installed("haystack"), reason="haystack-ai installed; stub error path is unreachable in this env")
def test_haystack_stub_import_error():
    """Importing haystack stub raises ImportError when haystack-ai is not installed."""
    _purge("asqav.extras.haystack", "haystack", "haystack.dataclasses")
    with pytest.raises(ImportError, match="pip install asqav"):
        import asqav.extras.haystack  # noqa: F401


@pytest.mark.skipif(_module_installed("agents"), reason="openai-agents installed; stub error path is unreachable in this env")
def test_openai_agents_stub_import_error():
    """Importing openai_agents stub raises ImportError when openai-agents is not installed."""
    _purge("asqav.extras.openai_agents", "agents", "agents.tracing")
    with pytest.raises(ImportError, match="pip install asqav"):
        import asqav.extras.openai_agents  # noqa: F401
