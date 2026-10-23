"""Fail when an SDK docstring or comment passes the concision caps.
Exit 0 clean, 1 on a finding, 2 on an unreadable file."""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from pathlib import Path
from typing import NamedTuple

MAX_DOCSTRING_LINES = 6
MAX_COMMENT_COLUMNS = 80
MAX_BLOCK_ROWS = 2

SOURCE_DIR = Path("python/src/asqav")
EXCLUDED_DIR = "verifier"
TOOL_DIRECTIVE = re.compile(r"#\s*(noqa|type:\s*ignore|pragma:|ruff:)")
PARAM_SECTION = re.compile(
    r"^[ \t]*(?:Args:|Parameters:|Parameters[ \t]*\n[ \t]*-{3,})[ \t]*$", re.MULTILINE
)

DocNode = ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


class Finding(NamedTuple):
    path: str
    line: int
    kind: str


class ScanError(Exception):
    """A file could not be read or parsed, so the scan is incomplete."""


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _module_is_public(relative: Path) -> bool:
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return all(_is_public(part) for part in parts)


def _docstring_findings(tree: ast.Module, path: str, public_module: bool) -> list[Finding]:
    found: list[Finding] = []

    def visit(node: DocNode, public: bool) -> None:
        doc = ast.get_docstring(node, clean=False)
        if doc is not None:
            first = node.body[0]
            span = (first.end_lineno or first.lineno) - first.lineno + 1
            documents_params = (
                public
                and not isinstance(node, ast.Module)
                and PARAM_SECTION.search(doc) is not None
            )
            if span > MAX_DOCSTRING_LINES and not documents_params:
                found.append(Finding(path, first.lineno, "docstring"))
        for child in _children(node):
            visit(child, public and _is_public(child.name))

    visit(tree, public_module)
    return found


def _children(node: ast.AST) -> list[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
    """The defs and classes nested in node, looking through if/try/with."""
    found: list[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append(child)
        else:
            found.extend(_children(child))
    return found


def _comment_findings(text: str, path: str) -> list[Finding]:
    lines = text.splitlines()
    found: list[Finding] = []
    own_rows: list[int] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type != tokenize.COMMENT:
                continue
            row, column = token.start
            too_long = len(token.string) + column > MAX_COMMENT_COLUMNS
            if too_long and not TOOL_DIRECTIVE.match(token.string):
                found.append(Finding(path, row, "comment-line"))
            if not lines[row - 1][:column].strip():
                own_rows.append(row)
    except (tokenize.TokenError, SyntaxError) as error:
        raise ScanError(f"{path}: cannot tokenize: {error}") from error
    run: list[int] = []
    for row in [*own_rows, 0]:
        if run and row == run[-1] + 1:
            run.append(row)
            continue
        if len(run) > MAX_BLOCK_ROWS:
            found.append(Finding(path, run[0], "comment-block"))
        run = [row]
    return found


def scan_file(root: Path, file: Path) -> list[Finding]:
    name = file.relative_to(root).as_posix()
    try:
        text = file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ScanError(f"{name}: cannot be read: {error}") from error
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as error:
        raise ScanError(f"{name}: cannot be parsed: {error}") from error
    public = _module_is_public(file.relative_to(root / SOURCE_DIR))
    return _docstring_findings(tree, name, public) + _comment_findings(text, name)


def scan(root: Path) -> list[Finding]:
    """Every finding under root/python/src/asqav, verifier/ excluded."""
    source = root / SOURCE_DIR
    if not source.is_dir():
        raise ScanError(f"{SOURCE_DIR.as_posix()}: no such directory under {root}")
    found: list[Finding] = []
    for file in sorted(source.rglob("*.py")):
        if file.relative_to(source).parts[0] == EXCLUDED_DIR:
            continue
        found.extend(scan_file(root, file))
    return sorted(found)


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        print("usage: check_comment_concision.py [root]", file=sys.stderr)
        return 2
    root = Path(argv[0]) if argv else Path(__file__).resolve().parents[2]
    try:
        findings = scan(root)
    except ScanError as error:
        print(f"check_comment_concision: {error}", file=sys.stderr)
        return 2
    for finding in findings:
        print(f"{finding.path}:{finding.line}:{finding.kind}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
