# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""The verbosity counter the python tests are held to.

Two budgets: a docstring block spans at most six physical lines and a comment
line at most eighty characters. The counter is stdlib only, and the tests
drive it over a file they write (the NEGCTL cases).
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

DOCSTRING_MAX_LINES = 6
COMMENT_MAX_CHARS = 80

__all__ = ["COMMENT_MAX_CHARS", "DOCSTRING_MAX_LINES", "python_files", "scan"]

_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def python_files(root: Path) -> list[Path]:
    """Every Python file under a path, or the file itself."""
    path = Path(root)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.py") if p.is_file())


def _docstring_hits(source: str, path: Path) -> list[tuple[Path, int, str]]:
    """The docstrings of one source that span more than the budget."""
    hits: list[tuple[Path, int, str]] = []
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, _OWNERS):
            continue
        body = getattr(node, "body", None)
        if not body or not isinstance(body[0], ast.Expr):
            continue
        value = body[0].value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        span = body[0].end_lineno - body[0].lineno + 1
        if span > DOCSTRING_MAX_LINES:
            hits.append((path, body[0].lineno, "docstring"))
    return hits


def _comment_hits(source: str, path: Path) -> list[tuple[Path, int, str]]:
    """The comment lines of one source that run past the budget."""
    hits: list[tuple[Path, int, str]] = []
    reader = io.StringIO(source).readline
    for token in tokenize.generate_tokens(reader):
        if token.type == tokenize.COMMENT and len(token.string) > COMMENT_MAX_CHARS:
            hits.append((path, token.start[0], "comment"))
    return hits


def scan(root: Path) -> list[tuple[Path, int, str]]:
    """Every budget hit under a path, as a (path, line, kind) triple.

    A file that cannot be read, parsed or tokenized raises, and a path with no
    Python file raises too: an empty result means the tree was measured and
    holds no hit.
    """
    files = python_files(root)
    if not files:
        raise ValueError(f"no Python file under {root}")
    hits: list[tuple[Path, int, str]] = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        hits.extend(_docstring_hits(source, path))
        hits.extend(_comment_hits(source, path))
    return hits
