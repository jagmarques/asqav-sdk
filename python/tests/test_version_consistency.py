"""Version consistency: __version__ in asqav/__init__.py must match version in pyproject.toml.

Prevents the drift bug that yanked 0.3.7 from PyPI (where pyproject.toml shipped a
version different from asqav.__version__).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[import-not-found, no-redef]
    except ImportError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]


PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.mark.skipif(tomllib is None, reason="No TOML parser available (Python 3.10 without tomli)")
def test_dunder_version_matches_pyproject() -> None:
    from asqav import __version__

    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)

    pyproject_version = data["project"]["version"]

    assert __version__ == pyproject_version, (
        f"asqav.__version__ ({__version__!r}) does not match pyproject.toml "
        f"version ({pyproject_version!r}). This is the same drift bug that "
        f"yanked 0.3.7 from PyPI - bump both together."
    )
