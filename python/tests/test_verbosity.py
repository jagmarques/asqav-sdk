# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""The counter holds the python tests to the docstring and comment budgets."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests._verbosity import (
    COMMENT_MAX_CHARS,
    DOCSTRING_MAX_LINES,
    DOCSTRING_MAX_WIDTH,
    python_files,
    scan,
)

TESTS = Path(__file__).resolve().parent
_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

# NEGCTL plants: a docstring, a docstring line and a comment one over budget,
# an indented comment over it and a file the parser cannot read.


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


def test_indented_comment_is_measured_by_its_physical_line(tmp_path: Path) -> None:
    """NEGCTL: a 79-char comment indented four spaces is an 83-char line."""
    comment = "#" + "x" * (COMMENT_MAX_CHARS - 2)
    assert len(comment) == COMMENT_MAX_CHARS - 1
    path = _write(tmp_path / "indented.py", f"def f():\n    {comment}\n    return 1\n")
    assert scan(tmp_path) == [(path, 2, "comment")]


def test_trailing_comment_is_measured_by_its_physical_line(tmp_path: Path) -> None:
    """NEGCTL: a short trailing comment on a long code line is reported."""
    code = "value = " + "1 + " * 18 + "1"
    path = _write(tmp_path / "trailing.py", f"{code}  # short\n")
    assert scan(tmp_path) == [(path, 1, "comment")]


def test_pragma_is_exempt_and_prose_beside_it_is_not(tmp_path: Path) -> None:
    """NEGCTL: a long noqa or type pragma passes; a long prose comment fails."""
    pad = "x" * 60
    pragmas = {"pragma.py": "noqa: E501 and more words", "typed.py": "type: ignore[x]"}
    for name, pragma in pragmas.items():
        _write(tmp_path / name, f"{pad} = 1  # {pragma}\n")
    assert scan(tmp_path) == []
    prose = _write(tmp_path / "prose.py", f"{pad} = 1  # prose past the budget\n")
    assert scan(tmp_path) == [(prose, 1, "comment")]


def test_wide_docstring_line_fails(tmp_path: Path) -> None:
    """NEGCTL: a docstring line past the width budget is reported."""
    at = '"""' + "x" * (DOCSTRING_MAX_WIDTH - 6) + '"""\n'
    _write(tmp_path / "at.py", at)
    assert scan(tmp_path) == []
    over = _write(tmp_path / "over.py", '"""' + "x" * (DOCSTRING_MAX_WIDTH - 5) + '"""\n')
    assert scan(tmp_path) == [(over, 1, "docstring-width")]


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
