"""SPDX scope gate of check_distribution_licenses.py. Each NEGCTL plants
a header in a tmp copy of the sources and runs the shipped main()."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
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


TS_APACHE = [rel for rel in checker.APACHE_SPDX_FILES if rel.startswith("typescript/")]


def test_spdx_allowlist_is_exactly_two_files():
    assert set(checker.APACHE_SPDX_FILES) == set(checker.find_spdx_files(REPO))
    assert len(checker.APACHE_SPDX_FILES) == 2


def test_ts_notice_and_pack_files_name_the_apache_file():
    notice = (REPO / "typescript" / "NOTICE").read_text()
    files = json.loads((REPO / "typescript" / "package.json").read_text())["files"]
    assert TS_APACHE
    for rel in TS_APACHE:
        packed = rel.split("/", 1)[1]
        assert packed in notice
        assert packed in files
    assert {"dist", "README.md", "LICENSE", "NOTICE"} <= set(files)


@pytest.mark.parametrize(
    "text",
    [
        'import { canonicalize } from "../jcs";',
        'import "@noble/post-quantum";',
        'import { request } from "node:https";',
        'const r = await fetch("https://api.asqav.com");',
        "const key = process.env.ASQAV_API_KEY;",
        'const m = require("node:http");',
        'const m = await import("node:https");',
    ],
)
def test_negctl_ts_verifier_reaches_beyond_node_builtins(tmp_path, text):
    code, err = _planted_gate(tmp_path, TS_APACHE[0], text, 3)
    assert code == 1
    assert TS_APACHE[0] in err


CARRIERS = ("README.md", "python/README.md", "python/NOTICE")


def _unnamed(text: str) -> list[str]:
    names = [rel.removeprefix("python/src/") for rel in checker.APACHE_SPDX_FILES]
    return [name for name in names if name not in text]


@pytest.mark.parametrize("carrier", CARRIERS)
def test_license_carriers_name_every_apache_file(carrier):
    assert _unnamed((REPO / carrier).read_text()) == []


def test_negctl_carrier_missing_the_ts_file_is_caught():
    text = (REPO / "README.md").read_text().replace(TS_APACHE[0], "typescript/src/verifier/")
    assert _unnamed(text) == [TS_APACHE[0]]


PINNED = (
    "SKEW_BOUND_SECONDS",
    "MAX_NESTING_DEPTH",
    "FIRST_RECEIPT_SEED",
    "REQUIRED_FIELDS",
    "ALLOWED_TYPES",
)
_TS_DUMP = (
    "const m = await import(process.argv[1]); const out = {};"
    "for (const k of process.argv.slice(2)) out[k] = m[k] instanceof Set ? [...m[k]].sort() : m[k];"
    "console.log(JSON.stringify(out));"
)


def _python_constants() -> dict:
    rel = next(rel for rel in checker.APACHE_SPDX_FILES if rel.endswith(".py"))
    spec = importlib.util.spec_from_file_location("standalone_verify_receipt", REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = {name: getattr(module, name) for name in PINNED}
    out["REQUIRED_FIELDS"] = list(out["REQUIRED_FIELDS"])
    out["ALLOWED_TYPES"] = sorted(out["ALLOWED_TYPES"])
    return out


def _ts_constants(path: Path) -> dict:
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    run = subprocess.run(
        ["node", "--no-warnings", "--input-type=module", "-e", _TS_DUMP, str(path), *PINNED],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(run.stdout)


def test_ts_verifier_constants_match_the_python_verifier():
    assert _ts_constants(REPO / TS_APACHE[0]) == _python_constants()


def test_negctl_ts_constant_drift_is_caught(tmp_path):
    want = _python_constants()
    pinned = f"SKEW_BOUND_SECONDS = {want['SKEW_BOUND_SECONDS']}"
    text = (REPO / TS_APACHE[0]).read_text()
    assert pinned in text
    copy = tmp_path / "verify-receipt.ts"
    copy.write_text(text.replace(pinned, f"SKEW_BOUND_SECONDS = {want['SKEW_BOUND_SECONDS'] + 1}"))
    assert _ts_constants(copy) != want
