"""Tests for the asqav hook CLI (PostToolUse / PreToolUse dry-run paths)."""

from __future__ import annotations

import json
import os
import sys

from typer.testing import CliRunner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.cli import app  # noqa: E402

runner = CliRunner()

# Sample Claude Code PostToolUse event (offline; no session_id needed for identity).
_POSTTOOL_EVENT = {
    "session_id": "s1",
    "tool_name": "Bash",
    "tool_input": {"command": "ls"},
    "tool_response": "x",
}

_PRETOOL_EVENT = {
    "session_id": "s2",
    "tool_name": "Write",
    "tool_input": {"file_path": "/tmp/f", "content": "hi"},
}


def _invoke_hook(subcommand: str, event: dict, extra_args: list[str] | None = None) -> dict:
    """Run `asqav hook <subcommand> --dry-run`, pipe event JSON, return parsed body."""
    args = ["hook", subcommand, "--dry-run"] + (extra_args or [])
    result = runner.invoke(app, args, input=json.dumps(event))
    assert result.exit_code == 0, f"CLI exited {result.exit_code}: {result.output}"
    return json.loads(result.output)


# === posttool honesty invariant ===


def test_posttool_dry_run_action_type() -> None:
    """action_type is tool:<tool_name> for posttool."""
    body = _invoke_hook("posttool", _POSTTOOL_EVENT)
    assert body["action_type"] == "tool:Bash"


def test_posttool_dry_run_context() -> None:
    """context carries the tool_input dict."""
    body = _invoke_hook("posttool", _POSTTOOL_EVENT)
    assert body["context"] == {"tool_input": {"command": "ls"}}


def test_posttool_dry_run_session_id() -> None:
    """session_id is forwarded from the hook event."""
    body = _invoke_hook("posttool", _POSTTOOL_EVENT)
    assert body["session_id"] == "s1"


def test_posttool_honesty_invariant_capture_topology() -> None:
    """Honesty invariant: posttool must set capture_topology=passive_telemetry."""
    body = _invoke_hook("posttool", _POSTTOOL_EVENT)
    assert body["capture_topology"] == "passive_telemetry"


def test_posttool_honesty_invariant_receipt_type() -> None:
    """Honesty invariant: posttool must set receipt_type=protectmcp:observation."""
    body = _invoke_hook("posttool", _POSTTOOL_EVENT)
    assert body["receipt_type"] == "protectmcp:observation"


def test_posttool_no_decision_receipt_type() -> None:
    """posttool must never emit a :decision receipt (would be a false attestation)."""
    body = _invoke_hook("posttool", _POSTTOOL_EVENT)
    assert body.get("receipt_type") != "protectmcp:decision"


# === pretool honesty invariant ===


def test_pretool_dry_run_receipt_type() -> None:
    """pretool is a gate decision: receipt_type=protectmcp:decision."""
    body = _invoke_hook("pretool", _PRETOOL_EVENT)
    assert body["receipt_type"] == "protectmcp:decision"


def test_pretool_dry_run_capture_topology() -> None:
    """pretool runs in-process: capture_topology=in_process_sdk."""
    body = _invoke_hook("pretool", _PRETOOL_EVENT)
    assert body["capture_topology"] == "in_process_sdk"


# === pretool fails CLOSED (exit 2) on every gate failure ===
# Claude Code treats PreToolUse exit 1 as a non-blocking error and proceeds, only
# exit 2 blocks (code.claude.com/docs/en/hooks). A malformed/empty event or missing
# identity must exit 2, else the gate fails OPEN and the tool runs unsigned.


def test_pretool_blocks_on_empty_stdin() -> None:
    """Empty stdin must exit 2 (block), not exit 1 (proceed unsigned)."""
    result = runner.invoke(app, ["hook", "pretool"], input="")
    assert result.exit_code == 2, f"expected block (2), got {result.exit_code}"


def test_pretool_blocks_on_whitespace_stdin() -> None:
    """Whitespace-only stdin must exit 2 (block)."""
    result = runner.invoke(app, ["hook", "pretool"], input="   \n")
    assert result.exit_code == 2, f"expected block (2), got {result.exit_code}"


def test_pretool_blocks_on_invalid_json() -> None:
    """Garbage (invalid JSON) on stdin must exit 2 (block)."""
    result = runner.invoke(app, ["hook", "pretool"], input="{not json")
    assert result.exit_code == 2, f"expected block (2), got {result.exit_code}"


def test_pretool_blocks_on_non_object_json() -> None:
    """Valid JSON that is not an object (array) must exit 2 (block)."""
    result = runner.invoke(app, ["hook", "pretool"], input="[1, 2, 3]")
    assert result.exit_code == 2, f"expected block (2), got {result.exit_code}"


def test_pretool_blocks_on_non_object_scalar() -> None:
    """Valid JSON scalar (a bare string) must exit 2 (block)."""
    result = runner.invoke(app, ["hook", "pretool"], input='"hello"')
    assert result.exit_code == 2, f"expected block (2), got {result.exit_code}"


def test_pretool_blocks_on_missing_identity(monkeypatch) -> None:
    """A well-formed event but no ASQAV_API_KEY/ASQAV_AGENT_ID must exit 2 (block)."""
    monkeypatch.delenv("ASQAV_API_KEY", raising=False)
    monkeypatch.delenv("ASQAV_AGENT_ID", raising=False)
    result = runner.invoke(app, ["hook", "pretool"], input=json.dumps(_PRETOOL_EVENT))
    assert result.exit_code == 2, f"expected block (2), got {result.exit_code}"


# === posttool audit path stays FAIL-OPEN (exit 1 on bad input is correct) ===


def test_posttool_fails_open_on_empty_stdin() -> None:
    """PostToolUse is audit-only: bad input exits 1 (non-blocking), never 2."""
    result = runner.invoke(app, ["hook", "posttool"], input="")
    assert result.exit_code == 1, f"expected fail-open (1), got {result.exit_code}"


def test_posttool_fails_open_on_invalid_json() -> None:
    """PostToolUse audit fails open on garbage input: exit 1, never the gate's 2."""
    result = runner.invoke(app, ["hook", "posttool"], input="{not json")
    assert result.exit_code == 1, f"expected fail-open (1), got {result.exit_code}"
