"""The counting check holds python/tests to the docstring and comment caps.
Each case runs the real scan; the planted files are the negative controls."""

from __future__ import annotations

from pathlib import Path

from tests.test_python_conciseness import conciseness

TESTS_DIR = Path(__file__).resolve().parent


def test_tests_tree_has_zero_hits():
    counts = conciseness.scan_tree(TESTS_DIR)
    assert counts.files_scanned > 0
    assert counts.findings == []
    assert counts.total == 0
