"""The PreToolUse gate when the cli extra is missing (criterion 578).

Exit 1 is a non-blocking error to the harness, so a gate that cannot import typer must
exit 2 or every tool call runs ungated. The import is blocked in a child process.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
_SNIPPET = (
    "import sys\n"
    "sys.modules['typer'] = None\n"  # None makes `import typer` raise
    "sys.argv = ['asqav'] + sys.argv[1:]\n"
    "import importlib\n"
    "importlib.import_module({module!r})\n"
)


def _run(module: str, *argv: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": os.path.abspath(_SRC)}
    code = _SNIPPET.format(module=module)
    return subprocess.run(
        [sys.executable, "-c", code, *argv],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


@pytest.mark.parametrize("module", ["asqav.cli", "asqav.cli_hook"])
def test_the_pretool_gate_blocks_when_typer_is_missing(module) -> None:
    result = _run(module, "hook", "pretool")
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert 'pip install "asqav[cli]"' in result.stderr


@pytest.mark.parametrize("module", ["asqav.cli", "asqav.cli_hook"])
def test_other_commands_keep_exit_one_when_typer_is_missing(module) -> None:
    result = _run(module, "doctor")
    assert result.returncode == 1, (result.returncode, result.stderr)
    assert 'pip install "asqav[cli]"' in result.stderr


def test_the_posttool_audit_keeps_exit_one_when_typer_is_missing() -> None:
    result = _run("asqav.cli", "hook", "posttool")
    assert result.returncode == 1, (result.returncode, result.stderr)
