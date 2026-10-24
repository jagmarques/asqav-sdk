"""The counting check holds python/tests to the docstring and comment caps.
Each case runs the real scan; the planted files are the negative controls."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_python_conciseness import conciseness

TESTS_DIR = Path(__file__).resolve().parent
MAX_DOCSTRING_LINES = conciseness.MAX_DOCSTRING_LINES
MAX_COMMENT_CHARS = conciseness.MAX_COMMENT_CHARS


def _test_with_docstring(tmp_path: Path, lines: int) -> Path:
    """A test file whose test_x docstring spans exactly `lines` lines."""
    body = [f"    line {number}" for number in range(2, lines)]
    source = "\n".join(['def test_x():', '    """line 1', *body, '    """', ""])
    target = tmp_path / "test_planted.py"
    target.write_text(source, encoding="utf-8")
    return target


def test_tests_tree_has_zero_hits():
    counts = conciseness.scan_tree(TESTS_DIR)
    assert counts.files_scanned > 0
    assert counts.findings == []
    assert counts.total == 0


def test_six_line_docstring_passes(tmp_path):
    _test_with_docstring(tmp_path, MAX_DOCSTRING_LINES)
    counts = conciseness.scan_tree(tmp_path)
    assert counts.files_scanned == 1
    assert counts.findings == []


# NEGCTL: one line over the cap, through the same scan as the tree test.
def test_seven_line_docstring_fails(tmp_path):
    path = _test_with_docstring(tmp_path, MAX_DOCSTRING_LINES + 1)
    counts = conciseness.scan_tree(tmp_path)
    assert counts.docstrings == 1
    assert len(counts.findings) == 1
    assert counts.findings[0].startswith(f"{path}:2: docstring of test_x spans")
    assert conciseness.main(["--path", str(tmp_path)]) == 1


# NEGCTL: an 81-char comment is one hit; an 80-char one is none.
def test_long_comment_fails(tmp_path):
    path = tmp_path / "test_comment.py"
    path.write_text("# " + "x" * (MAX_COMMENT_CHARS - 1) + "\n", encoding="utf-8")
    counts = conciseness.scan_tree(tmp_path)
    assert counts.comment_lines == 1
    assert len(counts.findings) == 1
    assert counts.findings[0].startswith(f"{path}:1: comment line is")
    path.write_text("# " + "x" * (MAX_COMMENT_CHARS - 2) + "\n", encoding="utf-8")
    assert conciseness.scan_tree(tmp_path).findings == []


def test_unparseable_file_raises(tmp_path):
    (tmp_path / "test_broken.py").write_text("def (\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        conciseness.scan_tree(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        conciseness.scan_tree(empty)
    assert conciseness.main(["--path", str(empty)]) == 2


def test_docstrings_name_behaviour():
    counts = conciseness.scan_tree(TESTS_DIR)
    assert counts.files_scanned > 0
    assert counts.blank_first_lines == 0, counts.findings


# NEGCTL: a docstring that opens on a blank line is reported.
def test_blank_first_docstring_line_fails(tmp_path):
    path = tmp_path / "test_blank.py"
    path.write_text('def test_x():\n    """\n    Summary.\n    """\n', encoding="utf-8")
    counts = conciseness.scan_tree(tmp_path)
    assert counts.blank_first_lines == 1
    assert len(counts.findings) == 1
    assert counts.findings[0].startswith(f"{path}:2: docstring of test_x opens")
    assert conciseness.main(["--path", str(tmp_path)]) == 1
