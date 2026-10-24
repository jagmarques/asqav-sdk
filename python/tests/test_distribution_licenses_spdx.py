"""SPDX scope gate of check_distribution_licenses.py. Each NEGCTL plants
a header in a tmp copy of the sources and runs the shipped main()."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_distribution_licenses.py"
_spec = importlib.util.spec_from_file_location("check_distribution_licenses", _SCRIPT)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)

REPO = checker.ROOT.parent
HEADER = "SPDX-License-Identifier: Apache-2.0"


def _gate(root: Path) -> tuple[int, str]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        code = checker.main(["--root", str(root)])
    return code, err.getvalue()


def _clean_tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for scan in checker.SPDX_SCAN_DIRS:
        (root / scan).mkdir(parents=True)
    for rel in checker.APACHE_SPDX_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, root / rel)
    return root


def _planted_gate(tmp_path: Path, rel: str, text: str, line: int) -> tuple[int, str]:
    root = _clean_tree(tmp_path)
    assert _gate(root) == (0, "")
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    source = REPO / rel
    lines = source.read_text().splitlines(True) if source.is_file() else ["x = 1\n"] * 4
    lines.insert(line, text + "\n")
    target.write_text("".join(lines))
    return _gate(root)


@pytest.mark.parametrize(
    "rel, text",
    [
        ("typescript/src/index.ts", f"// {HEADER}"),
        ("python/src/asqav/__init__.py", f"# {HEADER}"),
    ],
)
def test_spdx_gate_rejects_second_header(tmp_path, rel, text):
    code, err = _planted_gate(tmp_path, rel, text, 0)
    assert code == 1
    assert rel in err


@pytest.mark.parametrize(
    "rel, text, line",
    [
        ("typescript/src/index.ts", f"    // {HEADER}", 0),
        ("typescript/src/verifier/core.ts", f"/*\n * {HEADER}\n */", 0),
        ("python/src/asqav/client.py", f"# {HEADER}", 2),
        ("typescript/src/jcs.ts", f"// {HEADER.lower()}", 0),
        ("typescript/src/verifier/view.tsx", f"// {HEADER}", 0),
        ("typescript/src/verifier/run.mjs", f"// {HEADER}", 0),
        ("python/src/asqav/verifier/verify_receipt.pyi", f"# {HEADER}", 0),
        ("python/src/asqav/verify_receipt.py", f"# {HEADER}", 0),
    ],
)
def test_negctl_spdx_adversarial_forms(tmp_path, rel, text, line):
    code, err = _planted_gate(tmp_path, rel, text, line)
    assert code == 1
    assert rel in err


def test_negctl_allowlisted_file_without_header(tmp_path):
    root = _clean_tree(tmp_path)
    rel = checker.APACHE_SPDX_FILES[0]
    kept = [ln for ln in (root / rel).read_text().splitlines(True) if HEADER not in ln]
    (root / rel).write_text("".join(kept))
    code, err = _gate(root)
    assert code == 1
    assert rel in err


def test_spdx_gate_passes_at_head():
    assert _gate(REPO) == (0, "")
