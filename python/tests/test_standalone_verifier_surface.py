"""The standalone verifier artifact carries zero asqav producer dependency (421).

verify_receipt.py is the exit-artifact tool: one file, stdlib plus one optional
signature dependency, no asqav producer module. The AST pins the import
surface, and a subprocess runs the copied file in a bare directory with the
asqav package refused at import time, against the published ML-DSA-65 receipt.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from asqav.verifier import verify_receipt as vr

VERIFIER_SOURCE = Path(vr.__file__)
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "verifier" / "docs" / "fixtures"

try:
    from dilithium_py.ml_dsa import ML_DSA_65 as _ML_DSA_CHECK  # noqa: F401

    _DILITHIUM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep, always present in CI
    _DILITHIUM_AVAILABLE = False


def _module_names() -> list[tuple[str, bool]]:
    """Every imported module root with whether it sits inside a function body.

    Returns (root, lazy) pairs from the AST, lazy=True when the import appears
    only inside a function, mirroring the optional-dep treatment.
    """
    tree = ast.parse(VERIFIER_SOURCE.read_text())
    found: list[tuple[str, bool]] = []

    class Walk(ast.NodeVisitor):
        def __init__(self) -> None:
            self.depth = 0

        def visit_FunctionDef(self, node):
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node):
            for alias in node.names:
                found.append((alias.name.split(".")[0], self.depth > 0))

        def visit_ImportFrom(self, node):
            if node.module and not node.level:
                found.append((node.module.split(".")[0], self.depth > 0))

    Walk().visit(tree)
    return found


def test_import_surface_is_stdlib_plus_optional_dilithium_only() -> None:
    allowed = set(sys.stdlib_module_names) | {"dilithium_py"}
    for root, _lazy in _module_names():
        assert root != "asqav", "the standalone verifier must never import producer code"
        assert root in allowed, f"non-stdlib import {root!r} in verify_receipt.py"


def test_dilithium_is_the_only_non_stdlib_dep_and_imports_lazily() -> None:
    roots = {root for root, _lazy in _module_names()}
    assert "dilithium_py" in roots, "the ML-DSA-65 axis must stay in the tool"
    for root, lazy in _module_names():
        if root == "dilithium_py":
            assert lazy, "dilithium-py must import inside the check, not at top level"


def test_the_file_level_apache_exception_stays_documented() -> None:
    # The standalone license exception is the contract the exit artifact ships
    text = VERIFIER_SOURCE.read_text()
    assert "SPDX-License-Identifier: Apache-2.0" in text
    assert "file-level" in text and "Apache-2.0" in text


#: Refuses outbound sockets and any asqav import in the child, so a verifier
#: that reached for the network or the producer package fails loudly
SITECUSTOMIZE = """
import socket
import sys

_BLOCKED = OSError("outbound network blocked by test_standalone_verifier_surface")


def _refuse(*_a, **_k):
    raise _BLOCKED


class _NoConnect(socket.socket):
    connect = _refuse
    connect_ex = _refuse


socket.socket = _NoConnect
socket.create_connection = _refuse


class _RefuseAsqav:
    def find_spec(self, name, path=None, target=None):
        if name == "asqav" or name.startswith("asqav."):
            raise ImportError(f"import of {name!r} refused: standalone check")
        return None


sys.meta_path.insert(0, _RefuseAsqav())
"""


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_standalone_file_verifies_the_published_receipt_without_asqav(tmp_path) -> None:
    shutil.copy(VERIFIER_SOURCE, tmp_path / "verify_receipt.py")
    shutil.copy(FIXTURES / "published-receipt.json", tmp_path / "receipt.json")
    shutil.copy(FIXTURES / "published-jwks.json", tmp_path / "jwks.json")
    (tmp_path / "sitecustomize.py").write_text(SITECUSTOMIZE)

    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    proc = subprocess.run(
        [
            sys.executable,
            "verify_receipt.py",
            "--receipt",
            "receipt.json",
            "--jwks",
            "jwks.json",
            "--offline",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "=> verified" in proc.stdout, proc.stdout
    # The lapsed signed expiry reports on its own axis and never folds the verdict
    assert "[FAIL] expiry" in proc.stdout, proc.stdout

    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["signature"]["alg"] == "ML-DSA-65"


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_standalone_file_rejects_a_tampered_published_receipt(tmp_path) -> None:
    shutil.copy(VERIFIER_SOURCE, tmp_path / "verify_receipt.py")
    receipt = json.loads((FIXTURES / "published-receipt.json").read_text())
    receipt["payload"]["decision"] = "deny"
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    shutil.copy(FIXTURES / "published-jwks.json", tmp_path / "jwks.json")
    (tmp_path / "sitecustomize.py").write_text(SITECUSTOMIZE)

    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    proc = subprocess.run(
        [
            sys.executable,
            "verify_receipt.py",
            "--receipt",
            "receipt.json",
            "--jwks",
            "jwks.json",
            "--offline",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 1, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "=> unverified" in proc.stdout, proc.stdout
