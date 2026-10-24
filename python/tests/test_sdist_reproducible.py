"""check_sdist_reproducible.py builds from clean archives of HEAD. The NEGCTL
edits a tracked file before the second build, so the digests must differ."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_sdist_reproducible.py"
_spec = importlib.util.spec_from_file_location("check_sdist_reproducible", _SCRIPT)
sdist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sdist)


def _check(**kwargs):
    try:
        return sdist.check(sdist.ROOT, **kwargs)
    except sdist.BuilderUnavailable as exc:
        pytest.skip(str(exc))


def _edit_tracked(tree: Path) -> None:
    target = tree / "python" / "src" / "asqav" / "__init__.py"
    target.write_text(target.read_text() + "\n# edited between builds\n")


def test_sdist_twice_same_sha256():
    result = _check()
    assert result.digests[0] == result.digests[1]
    assert result.ok is True
    assert result.names[0] == result.names[1]


def test_negctl_tracked_file_edited_between_builds():
    result = _check(between=_edit_tracked)
    assert result.digests[0] != result.digests[1]
    assert result.ok is False
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        code = sdist.main([], between=_edit_tracked)
    assert code == 1
    assert "digests differ" in err.getvalue()
