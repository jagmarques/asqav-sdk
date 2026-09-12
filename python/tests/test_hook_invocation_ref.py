"""The hook forwards `tool_use_id` as `invocation_ref` on both hook paths.

A pre-action decision receipt and a post-action observation receipt for the
same tool call carry the SAME value, so a reader can SEE duplicate emissions.
Absent or non-string inputs omit the field entirely: nothing is synthesised
and there is no fallback to `session_id` (the trace, not the invocation).
Visibility only: no de-duplication, no rejection keyed on this field.

The `_map_event` kwargs below mirror the real `hook_pretool` / `hook_posttool`
call sites exactly; the CLI controls beneath invoke the real entry points so
the entrypoint semantics cannot drift behind these unit tests.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav import cli_hook  # noqa: E402
from asqav.cli import app  # noqa: E402
from asqav.client import Agent  # noqa: E402

# Exact kwargs of the hook_pretool call site (FAIL-CLOSED gate).
_PRE_FIELDS = dict(
    receipt_type="protectmcp:decision",
    capture_topology="in_process_sdk",
    bind_result=False,
    policy_decision="permit",
)
# Exact kwargs of the hook_posttool call site without --bind-result (audit).
_POST_FIELDS = dict(
    receipt_type="protectmcp:observation",
    capture_topology="passive_telemetry",
    bind_result=False,
    policy_decision=None,
)

_runner = CliRunner()


def _pre_event(tool_use_id: Any = "toolu_abc123") -> dict[str, Any]:
    event: dict[str, Any] = {
        "session_id": "sess_1",
        "tool_name": "Write",
        "tool_input": {"file_path": "/tmp/f"},
    }
    if tool_use_id is not ...:
        event["tool_use_id"] = tool_use_id
    return event


def _post_event(tool_use_id: Any = "toolu_abc123") -> dict[str, Any]:
    event = _pre_event(tool_use_id)
    event["tool_response"] = {"success": True}
    return event


def _dry_run(subcommand: str, event: dict, extra_args: list[str] | None = None) -> dict:
    args = ["hook", subcommand, "--dry-run"] + (extra_args or [])
    result = _runner.invoke(app, args, input=json.dumps(event))
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_pre_receipt_carries_tool_use_id_verbatim() -> None:
    _, _, _, fields = cli_hook._map_event(_pre_event(), **_PRE_FIELDS)
    assert fields["invocation_ref"] == "toolu_abc123"


def test_post_receipt_carries_tool_use_id_verbatim() -> None:
    _, _, _, fields = cli_hook._map_event(_post_event(), **_POST_FIELDS)
    assert fields["invocation_ref"] == "toolu_abc123"


def test_pre_and_post_share_one_invocation_ref() -> None:
    _, _, _, pre = cli_hook._map_event(_pre_event(), **_PRE_FIELDS)
    _, _, _, post = cli_hook._map_event(_post_event(), **_POST_FIELDS)
    assert pre["invocation_ref"] == post["invocation_ref"] == "toolu_abc123"


def test_absent_tool_use_id_omits_the_key() -> None:
    _, _, _, fields = cli_hook._map_event(_pre_event(...), **_PRE_FIELDS)
    assert "invocation_ref" not in fields
    assert fields["trace_id"] == "sess_1"


def test_non_string_tool_use_id_omits_the_key() -> None:
    for bad in (123, ["toolu_x"], {"id": "toolu_x"}, True):
        _, _, _, fields = cli_hook._map_event(_pre_event(bad), **_PRE_FIELDS)
        assert "invocation_ref" not in fields, bad


def test_empty_tool_use_id_omits_the_key() -> None:
    _, _, _, fields = cli_hook._map_event(_pre_event(""), **_PRE_FIELDS)
    assert "invocation_ref" not in fields


def test_cli_pretool_dry_run_forwards_the_reference() -> None:
    body = _dry_run("pretool", _pre_event())
    assert body["invocation_ref"] == "toolu_abc123"
    assert body["receipt_type"] == "protectmcp:decision"


def test_cli_posttool_dry_run_forwards_the_reference() -> None:
    body = _dry_run("posttool", _post_event())
    assert body["invocation_ref"] == "toolu_abc123"
    assert body["receipt_type"] == "protectmcp:observation"


def test_cli_posttool_bind_result_forwards_the_reference() -> None:
    body = _dry_run("posttool", _post_event(), ["--bind-result"])
    assert body["invocation_ref"] == "toolu_abc123"
    assert body["receipt_type"] == "protectmcp:observation:result_bound"
    assert "result_digest" in body


def test_cli_pretool_dry_run_omits_a_missing_reference() -> None:
    body = _dry_run("pretool", _pre_event(...))
    assert "invocation_ref" not in body


def _agent() -> Agent:
    return Agent(
        agent_id="agent_hook",
        name="hook-agent",
        public_key="pk_test",
        key_id="key_hook",
        algorithm="ML-DSA-65",
        capabilities=["tool:*"],
        created_at=1700000000.0,
    )


def _ok_response() -> dict:
    return {
        "signature": "sig_b64",
        "signature_id": "sig_abc",
        "action_id": "act_abc",
        "timestamp": 1700000000.0,
        "verification_url": "https://example.invalid/verify/sig_abc",
    }


def test_sign_projects_invocation_ref_to_wire() -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign("tool:Write", context={"tool_input": {}}, invocation_ref="toolu_abc123")
    assert captured["body"]["invocation_ref"] == "toolu_abc123"


def test_sign_omits_invocation_ref_when_unset() -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign("tool:Write", context={"tool_input": {}})
    assert "invocation_ref" not in captured["body"]
