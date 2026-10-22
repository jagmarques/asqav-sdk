"""The counting check holds the verifier sources to the conciseness budget.

The check in tools/ measures three budgets over a scanned tree: a docstring
block of at most six lines, a comment line of at most eighty characters and a
comment block of at most two comment lines. These tests run the real check
over the verifier and plant the bad inputs it has to reject.
"""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_python_conciseness as conciseness  # noqa: E402

VERIFIER_DIR = REPO_ROOT / "python" / "src" / "asqav" / "verifier"
VERIFY_RECEIPT = VERIFIER_DIR / "verify_receipt.py"


def _copy_of_verify_receipt(tmp_path: Path) -> Path:
    """A copy of the verifier source in tmp_path, for the planted bad inputs."""
    copy = tmp_path / "verify_receipt.py"
    shutil.copy(VERIFY_RECEIPT, copy)
    return copy


def _last_top_level_import_line(source: str) -> int:
    """The line of the last top-level import statement of a source file."""
    tree = ast.parse(source)
    last = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last = max(last, node.end_lineno or 0)
    return last


def _first_statement_line(source: str) -> tuple[int, int]:
    """The line and the column of the first statement of the first function."""
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    statement = function.body[0]
    return statement.lineno - 1, statement.col_offset


def test_verifier_scope_counts_zero():
    counts = conciseness.scan_tree(VERIFIER_DIR)
    assert counts.files_scanned > 0
    assert counts.docstrings == 0, counts.findings
    assert counts.comment_lines == 0, counts.findings
    assert counts.comment_blocks == 0, counts.findings


def test_verifier_license_header_kept():
    source = VERIFY_RECEIPT.read_text(encoding="utf-8")
    header = conciseness.license_header_lines(source)
    assert header, "verify_receipt.py must open with the pinned licence header"
    text = "\n".join(header)
    assert "SPDX-License-Identifier: Apache-2.0" in text
    assert "Licensed under the Apache License, Version 2.0" in text
    counts = conciseness.scan_tree(VERIFY_RECEIPT)
    assert counts.comment_lines == 0, counts.findings
    assert counts.comment_blocks == 0, counts.findings


def test_license_exemption_deny(tmp_path):
    copy = _copy_of_verify_receipt(tmp_path)
    lines = copy.read_text(encoding="utf-8").splitlines()
    planted = [
        "# SPDX-License-Identifier: Apache-2.0",
        "# a second licence line below the imports",
        "# a third licence line below the imports",
    ]
    index = _last_top_level_import_line("\n".join(lines))
    lines[index:index] = planted
    copy.write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = conciseness.scan_tree(copy)
    assert counts.comment_blocks == 1, counts.findings
    assert any(f":{index + 1}:" in finding for finding in counts.findings)
    assert conciseness.main(["--path", str(copy)]) != 0


def test_license_exemption_covers_the_header_only(tmp_path):
    copy = _copy_of_verify_receipt(tmp_path)
    lines = copy.read_text(encoding="utf-8").splitlines()
    header = conciseness.license_header_lines("\n".join(lines))
    assert header
    index = len(header)
    lines[index:index] = ["# a", "# b", "# c"]
    copy.write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = conciseness.scan_tree(copy)
    assert counts.comment_blocks == 1, counts.findings
    assert any(f":{index + 1}:" in finding for finding in counts.findings)


def test_license_header_must_open_the_file(tmp_path):
    copy = _copy_of_verify_receipt(tmp_path)
    source = copy.read_text(encoding="utf-8")
    copy.write_text("VALUE = 1\n" + source, encoding="utf-8")
    assert conciseness.license_header_lines(copy.read_text(encoding="utf-8")) == []
    counts = conciseness.scan_tree(copy)
    assert counts.comment_blocks == 1, counts.findings


def test_negctl_three_line_block(tmp_path):
    copy = _copy_of_verify_receipt(tmp_path)
    source = copy.read_text(encoding="utf-8")
    index, column = _first_statement_line(source)
    indent = " " * column
    lines = source.splitlines()
    lines[index:index] = [f"{indent}# a", f"{indent}# b", f"{indent}# c"]
    copy.write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = conciseness.scan_tree(copy)
    assert counts.comment_blocks == 1, counts.findings
    assert any(f":{index + 1}:" in finding for finding in counts.findings)
    assert conciseness.main(["--path", str(copy)]) != 0


def test_negctl_long_docstring(tmp_path):
    copy = _copy_of_verify_receipt(tmp_path)
    planted = "\n".join(
        [
            "",
            "",
            "def _planted_docstring():",
            '    """one',
            "    two",
            "    three",
            "    four",
            "    five",
            "    six",
            '    """',
        ]
    )
    with copy.open("a", encoding="utf-8") as handle:
        handle.write(planted + "\n")
    counts = conciseness.scan_tree(copy)
    assert counts.docstrings == 1, counts.findings
    assert conciseness.main(["--path", str(copy)]) != 0


def test_negctl_long_comment_line(tmp_path):
    copy = _copy_of_verify_receipt(tmp_path)
    planted = "\n".join(["", "", "# " + "x" * 79])
    with copy.open("a", encoding="utf-8") as handle:
        handle.write(planted + "\n")
    counts = conciseness.scan_tree(copy)
    assert counts.comment_lines == 1, counts.findings
    assert conciseness.main(["--path", str(copy)]) != 0


def test_comment_line_at_the_budget_is_accepted(tmp_path):
    copy = _copy_of_verify_receipt(tmp_path)
    planted = "\n".join(["", "", "# " + "x" * (conciseness.MAX_COMMENT_CHARS - 2)])
    with copy.open("a", encoding="utf-8") as handle:
        handle.write(planted + "\n")
    counts = conciseness.scan_tree(copy)
    assert counts.comment_lines == 0, counts.findings


def test_scan_reports_a_broken_file(tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("def oops(:\n", encoding="utf-8")
    try:
        conciseness.scan_tree(broken)
    except SyntaxError:
        pass
    else:
        raise AssertionError("a file that does not parse must raise, never scan as empty")
    assert conciseness.main(["--path", str(broken)]) == 2


def test_scan_reports_an_unreadable_file(tmp_path):
    unreadable = tmp_path / "unreadable.py"
    unreadable.write_bytes(b"\xff\xfe\x00not utf-8\n")
    try:
        conciseness.scan_tree(unreadable)
    except UnicodeDecodeError:
        pass
    else:
        raise AssertionError("a file that cannot be decoded must raise, never scan as empty")
    assert conciseness.main(["--path", str(unreadable)]) == 2


def test_scan_reports_a_missing_path(tmp_path):
    assert conciseness.main(["--path", str(tmp_path / "absent.py")]) == 2


def test_scan_reports_an_empty_tree(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert conciseness.main(["--path", str(empty)]) == 2


def test_scan_counts_a_misplaced_string_statement(tmp_path):
    copy = tmp_path / "misplaced.py"
    body = "\n".join(f"    line {number}" for number in range(6))
    copy.write_text(f'def f():\n    value = 1\n    """one\n{body}\n    """\n', encoding="utf-8")
    counts = conciseness.scan_tree(copy)
    assert counts.docstrings == 1, counts.findings


def test_scan_counts_an_f_string_statement(tmp_path):
    copy = tmp_path / "fstring.py"
    body = "\n".join(f"    line {number}" for number in range(6))
    copy.write_text(f'def f():\n    value = 1\n    f"""one\n{body}\n    """\n', encoding="utf-8")
    counts = conciseness.scan_tree(copy)
    assert counts.docstrings == 1, counts.findings


def test_scan_reads_a_case_variant_file(tmp_path):
    (tmp_path / "UPPER.PY").write_text("# " + "x" * 79 + "\n", encoding="utf-8")
    counts = conciseness.scan_tree(tmp_path)
    assert counts.files_scanned == 1
    assert counts.comment_lines == 1, counts.findings


def test_scan_follows_a_symlinked_directory(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    (target / "linked.py").write_text("# " + "x" * 79 + "\n", encoding="utf-8")
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "link").symlink_to(target, target_is_directory=True)
    counts = conciseness.scan_tree(tree)
    assert counts.files_scanned == 1
    assert counts.comment_lines == 1, counts.findings


def test_verify_receipt_stdlib_only():
    tree = ast.parse(VERIFY_RECEIPT.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert roots
    for name in sorted(roots):
        assert name in sys.stdlib_module_names, name
