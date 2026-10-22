# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""The counter holds the python tests to the docstring and comment budgets.

Every test here drives the real counter over a file it writes, so a bad input
is measured the same way the python tests are.
"""

from __future__ import annotations

from pathlib import Path

from tests._verbosity import COMMENT_MAX_CHARS, DOCSTRING_MAX_LINES, scan

# NEGCTL: the planted inputs are a docstring one line over the budget and a
# comment one character over it.


def _write(path: Path, text: str) -> Path:
    """Write a source file and return its path."""
    path.write_text(text, encoding="utf-8")
    return path


def test_six_line_docstring_passes(tmp_path: Path) -> None:
    """A docstring of exactly the budget passes the counter."""
    lines = "\n".join(f"    line {n}" for n in range(1, DOCSTRING_MAX_LINES))
    _write(tmp_path / "six.py", f'def test_x():\n    """one\n{lines}"""\n')
    assert scan(tmp_path) == []


def test_seven_line_docstring_fails(tmp_path: Path) -> None:
    """NEGCTL: a docstring one line over the budget is reported."""
    lines = "\n".join(f"    line {n}" for n in range(1, DOCSTRING_MAX_LINES + 1))
    path = _write(tmp_path / "seven.py", f'def test_x():\n    """one\n{lines}"""\n')
    assert scan(tmp_path) == [(path, 2, "docstring")]


def test_long_comment_fails(tmp_path: Path) -> None:
    """NEGCTL: a comment past the budget is reported, one at it is not."""
    _write(tmp_path / "at.py", "#" + "x" * (COMMENT_MAX_CHARS - 1) + "\n")
    assert scan(tmp_path) == []
    over = _write(tmp_path / "over.py", "#" + "x" * COMMENT_MAX_CHARS + "\n")
    assert scan(tmp_path) == [(over, 1, "comment")]
