# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""The counter holds the python tests to the docstring and comment budgets.

Every test here drives the real counter over a file it writes, so a bad input
is measured the same way the python tests are.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests._verbosity import (
    COMMENT_MAX_CHARS,
    DOCSTRING_MAX_LINES,
    python_files,
    scan,
)

TESTS = Path(__file__).resolve().parent
_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

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


def test_unparseable_file_raises(tmp_path: Path) -> None:
    """A file that cannot be parsed raises, and an empty tree is not clean."""
    _write(tmp_path / "bad.py", "def (\n")
    with pytest.raises(SyntaxError):
        scan(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        scan(empty)


def test_tests_tree_has_zero_hits() -> None:
    """The python tests hold the docstring and the comment budget."""
    assert scan(TESTS) == []
    assert len(python_files(TESTS)) > 0


def test_docstrings_name_behaviour() -> None:
    """Every docstring in the python tests names the behaviour under test."""
    for path in python_files(TESTS):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, _OWNERS):
                continue
            body = getattr(node, "body", None)
            if not body or not isinstance(body[0], ast.Expr):
                continue
            value = body[0].value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            first = value.value.splitlines()[0].strip()
            assert first, f"{path}:{body[0].lineno} docstring has a blank first line"
