#!/usr/bin/env python3
"""Count the over-long Python docstrings, comment lines and comment blocks.

The SDK's Python sources are held to three budgets: a docstring block spans at
most six physical lines, a comment line at most eighty characters and a
comment block at most two comment lines.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

MAX_DOCSTRING_LINES = 6
MAX_COMMENT_CHARS = 80
MAX_COMMENT_BLOCK_LINES = 2

# The Apache-2.0 header of the standalone verifier, pinned by the digest of
# its text: the exemption follows the header, not the file name, so a header
# that is edited or a comment that merely mentions the licence is counted.
LICENSE_HEADER_SHA256 = "e6e7c4d87cfb25d4c262f62f06ef57f555ce812f87b2976488d8f02f9f953208"

_STRUCTURAL_TOKENS = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
    }
)


@dataclass
class Counts:
    """The three budgets measured over one scanned tree."""

    docstrings: int = 0
    comment_lines: int = 0
    comment_blocks: int = 0
    files_scanned: int = 0
    findings: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        """The number of findings over all three budgets."""
        return self.docstrings + self.comment_lines + self.comment_blocks


def _is_content(token: tokenize.TokenInfo) -> bool:
    """True when a comment token carries text beyond the comment marker."""
    return bool(token.string.lstrip("#").strip())


def comment_blocks(source: str) -> list[list[tokenize.TokenInfo]]:
    """Group the comment tokens of one file into blocks.

    A block is a run of comment lines with no code line between them; blank
    lines do not end a block, so a long comment cannot be split to reset the
    count.
    """
    blocks: list[list[tokenize.TokenInfo]] = []
    current: list[tokenize.TokenInfo] = []
    last_line = 0
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            current.append(token)
            last_line = token.start[0]
        elif token.type not in _STRUCTURAL_TOKENS and token.start[0] > last_line:
            if current:
                blocks.append(current)
                current = []
            last_line = token.start[0]
    if current:
        blocks.append(current)
    return blocks


def license_header_lines(source: str) -> list[str]:
    """The pinned licence header of a file, or an empty list.

    The header is the run of comment lines that opens the file and ends at its
    SPDX identifier. It is exempt only while its text is the pinned header, so
    an edited header is counted like any other comment.
    """
    blocks = comment_blocks(source)
    if not blocks:
        return []
    prefix: list[str] = []
    for token in blocks[0]:
        prefix.append(token.string)
        if "SPDX-License-Identifier:" in token.string:
            break
    else:
        return []
    digest = hashlib.sha256(("\n".join(prefix) + "\n").encode("utf-8")).hexdigest()
    return prefix if digest == LICENSE_HEADER_SHA256 else []


def scan_file(path: Path) -> Counts:
    """Measure the three budgets over one Python file."""
    counts = Counts()
    source = path.read_text(encoding="utf-8")
    counts.files_scanned = 1
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
            counts.docstrings += 1
            counts.findings.append(
                f"{path}:{block.lineno}: docstring of {name} spans {span} lines"
                f" (max {MAX_DOCSTRING_LINES})"
            )
    header = license_header_lines(source)
    blocks = comment_blocks(source)
    exempt = set()
    if header:
        exempt = {token.start[0] for token in blocks[0][: len(header)]}
    for block in blocks:
        counted = [token for token in block if token.start[0] not in exempt]
        for token in counted:
            if len(token.string) > MAX_COMMENT_CHARS:
                counts.comment_lines += 1
                counts.findings.append(
                    f"{path}:{token.start[0]}: comment line is {len(token.string)} characters"
                    f" (max {MAX_COMMENT_CHARS})"
                )
        content = [token for token in counted if _is_content(token)]
        if len(content) > MAX_COMMENT_BLOCK_LINES:
            counts.comment_blocks += 1
            counts.findings.append(
                f"{path}:{content[0].start[0]}: comment block spans {len(content)} comment lines"
                f" (max {MAX_COMMENT_BLOCK_LINES})"
            )
    return counts


def scan_tree(root: Path) -> Counts:
    """Measure the three budgets over one file or over a directory tree."""
    if root.is_dir():
        files = sorted(root.rglob("*.py"))
    elif root.is_file():
        files = [root]
    else:
        raise FileNotFoundError(root)
    total = Counts()
    for path in files:
        counts = scan_file(path)
        total.docstrings += counts.docstrings
        total.comment_lines += counts.comment_lines
        total.comment_blocks += counts.comment_blocks
        total.files_scanned += counts.files_scanned
        total.findings.extend(counts.findings)
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Count over-long docstrings, comment lines and comment blocks."
    )
    parser.add_argument("--path", nargs="+", default=["python/tests"], help="files or directories to scan")
    args = parser.parse_args(argv)
    total = Counts()
    for raw in args.path:
        root = Path(raw)
        if not root.exists():
            print(f"counting check: {root} does not exist", file=sys.stderr)
            return 2
        try:
            counts = scan_tree(root)
        except (OSError, SyntaxError, UnicodeDecodeError) as error:
            print(f"counting check: {root}: {error}", file=sys.stderr)
            return 2
        total.docstrings += counts.docstrings
        total.comment_lines += counts.comment_lines
        total.comment_blocks += counts.comment_blocks
        total.files_scanned += counts.files_scanned
        total.findings.extend(counts.findings)
    for finding in total.findings:
        print(finding)
    print(f"counting check: {total.files_scanned} file(s) under {', '.join(args.path)}")
    print(f"counting check: docstrings over {MAX_DOCSTRING_LINES} lines: {total.docstrings}")
    print(f"counting check: comment lines over {MAX_COMMENT_CHARS} characters: {total.comment_lines}")
    print(f"counting check: comment blocks over {MAX_COMMENT_BLOCK_LINES} lines: {total.comment_blocks}")
    return 1 if total.total else 0


if __name__ == "__main__":
    sys.exit(main())
