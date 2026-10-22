#!/usr/bin/env python3
"""Count the over-long Python docstrings and comment lines under a directory.

The SDK's Python tests carry the behaviour they pin in their docstrings and
their comments, so both are held to a measured budget: a docstring block spans
at most six physical lines, and a comment line at most eighty characters.

The check prints the counts it measured over the scanned tree and exits
non-zero when either count is above zero, so a regression fails the gate.
"""

from __future__ import annotations

import argparse
import ast
import io
import sys
import tokenize
from pathlib import Path

MAX_DOCSTRING_LINES = 6
MAX_COMMENT_CHARS = 80


def scan(path: Path) -> tuple[list[str], list[str]]:
    """Return the docstring and comment findings of one Python file."""
    source = path.read_text(encoding="utf-8")
    docstrings: list[str] = []
    comments: list[str] = []
    tree = ast.parse(source, filename=str(path))
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        if ast.get_docstring(node, clean=False) is None:
            continue
        block = node.body[0]
        span = block.end_lineno - block.lineno + 1
        if span > MAX_DOCSTRING_LINES:
            name = getattr(node, "name", "<module>")
            docstrings.append(
                f"{path}:{block.lineno}: docstring of {name} spans {span} lines"
                f" (max {MAX_DOCSTRING_LINES})"
            )
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        if len(token.string) > MAX_COMMENT_CHARS:
            comments.append(
                f"{path}:{token.start[0]}: comment line is {len(token.string)} characters"
                f" (max {MAX_COMMENT_CHARS})"
            )
    return docstrings, comments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count over-long docstrings and comments.")
    parser.add_argument("--path", default="python/tests", help="directory to scan")
    args = parser.parse_args(argv)
    root = Path(args.path)
    if not root.is_dir():
        print(f"counting check: {root} is not a directory", file=sys.stderr)
        return 2
    docstrings: list[str] = []
    comments: list[str] = []
    files = sorted(root.rglob("*.py"))
    for path in files:
        found_docstrings, found_comments = scan(path)
        docstrings.extend(found_docstrings)
        comments.extend(found_comments)
    for finding in docstrings + comments:
        print(finding)
    print(f"counting check: {len(files)} file(s) under {root}")
    print(f"counting check: docstrings over {MAX_DOCSTRING_LINES} lines: {len(docstrings)}")
    print(f"counting check: comment lines over {MAX_COMMENT_CHARS} characters: {len(comments)}")
    return 1 if docstrings or comments else 0


if __name__ == "__main__":
    sys.exit(main())
