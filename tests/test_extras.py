"""Tests for extras packaging and framework stub imports."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


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
    framework_extras = ["httpx", "cli", "langchain", "crewai", "litellm", "haystack", "openai-agents"]
    for extra in framework_extras:
        for dep in opt_deps[extra]:
            # Strip version specifiers for matching
            dep_name = dep.split(">=")[0].split("<=")[0].split("==")[0].strip()
            all_dep_names = [d.split(">=")[0].split("<=")[0].split("==")[0].strip() for d in all_deps]
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


def test_langchain_stub_import_error():
    """Importing langchain stub raises ImportError when langchain-core is not installed."""
    # Ensure the module is not cached from a previous import
    sys.modules.pop("asqav.extras.langchain", None)

    with pytest.raises(ImportError, match="pip install asqav"):
        import asqav.extras.langchain  # noqa: F401


def test_crewai_stub_import_error():
    """Importing crewai stub raises ImportError when crewai is not installed."""
    sys.modules.pop("asqav.extras.crewai", None)

    with pytest.raises(ImportError, match="pip install asqav"):
        import asqav.extras.crewai  # noqa: F401


def test_litellm_stub_import_error():
    """Importing litellm stub raises ImportError when litellm is not installed."""
    sys.modules.pop("asqav.extras.litellm", None)

    with pytest.raises(ImportError, match="pip install asqav"):
        import asqav.extras.litellm  # noqa: F401


def test_haystack_stub_import_error():
    """Importing haystack stub raises ImportError when haystack-ai is not installed."""
    sys.modules.pop("asqav.extras.haystack", None)

    with pytest.raises(ImportError, match="pip install asqav"):
        import asqav.extras.haystack  # noqa: F401


def test_openai_agents_stub_import_error():
    """Importing openai_agents stub raises ImportError when openai-agents is not installed."""
    sys.modules.pop("asqav.extras.openai_agents", None)

    with pytest.raises(ImportError, match="pip install asqav"):
        import asqav.extras.openai_agents  # noqa: F401
