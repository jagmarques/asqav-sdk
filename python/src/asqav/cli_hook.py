"""asqav hook - sign Claude Code harness hook events via the cloud sign endpoint.

This is the HARNESS-HOOK surface: Claude Code fires `asqav hook posttool`/`pretool`
with the hook event JSON on stdin, and the command signs it through
`asqav.client.sign()`. It is NOT `asqav/hooks.py` (those are in-process sign
callbacks the agent opts into). Honesty invariant: a posttool receipt observed the
action after it ran, so it forces capture_topology=passive_telemetry and a
receipt_type the cloud's rule-8 false_attestation_guard accepts. A posttool path
can never emit a :decision.
"""

from __future__ import annotations

import hashlib
import json as json_mod
import os
import sys
from typing import Any

try:
    import typer
except ImportError:
    print("CLI requires typer. Install with: pip install asqav[cli]")
    sys.exit(1)

hook_app = typer.Typer(
    name="hook",
    help="Sign Claude Code harness hook events (PostToolUse audit, PreToolUse gate).",
    no_args_is_help=True,
)

# rule 8 accepts only these two receipt_types under passive_telemetry. A posttool
# receipt is structurally pinned to this set so it cannot claim a decision.
_OBSERVATION = "protectmcp:observation"
_OBSERVATION_RESULT_BOUND = "protectmcp:observation:result_bound"
_DECISION = "protectmcp:decision"


def _read_event() -> dict[str, Any]:
    """Parse the hook event JSON from stdin. Exit 1 on empty or malformed input."""
    raw = sys.stdin.read()
    if not raw.strip():
        print("Error: no hook event on stdin (expected Claude Code hook JSON).", file=sys.stderr)
        raise typer.Exit(code=1)
    try:
        event = json_mod.loads(raw)
    except json_mod.JSONDecodeError as exc:
        print(f"Error: hook event is not valid JSON: {exc}", file=sys.stderr)
        raise typer.Exit(code=1) from exc
    if not isinstance(event, dict):
        print("Error: hook event must be a JSON object.", file=sys.stderr)
        raise typer.Exit(code=1)
    return event


def _result_digest(tool_response: Any) -> str:
    """sha256:<hex> over the JCS-canonical bytes of the tool response."""
    from asqav.canonicalize import canonicalize

    digest = hashlib.sha256(canonicalize(tool_response)).hexdigest()
    return f"sha256:{digest}"


def _build_body(
    *,
    action_type: str,
    context: dict[str, Any],
    session_id: str,
    agent_id: str,
    compliance_fields: dict[str, Any],
) -> dict[str, Any]:
    """Build the same body sign() POSTs, reusing the SDK builder (no re-impl)."""
    from asqav.client import _build_sign_body

    return _build_sign_body(
        action_type=action_type,
        context=context,
        session_id=session_id or None,
        agent_id=agent_id,
        compliance_fields={k: v for k, v in compliance_fields.items() if v is not None} or None,
    )


def _map_event(
    event: dict[str, Any],
    *,
    receipt_type: str,
    capture_topology: str,
    bind_result: bool,
    policy_decision: str | None,
) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Map a Claude Code hook event to sign() inputs (action_type, context, session, fields).

    Field map (snake_case keys confirmed against code.claude.com/docs/en/hooks):
    action_type=tool:<tool_name>; context={tool_input}; session_id and trace_id
    from session_id. result_digest is added only when binding a tool result.
    """
    tool_name = event.get("tool_name", "unknown")
    tool_input = event.get("tool_input", {})
    session_id = event.get("session_id", "") or ""

    action_type = f"tool:{tool_name}"
    context = {"tool_input": tool_input}
    compliance_fields: dict[str, Any] = {
        "compliance_mode": True,
        "receipt_type": receipt_type,
        "policy_decision": policy_decision or "permit",
        "capture_topology": capture_topology,
        "trace_id": session_id or None,
    }
    if bind_result:
        compliance_fields["result_digest"] = _result_digest(event.get("tool_response"))
    return action_type, context, session_id, compliance_fields


def _require_identity() -> tuple[str, str]:
    """Return (api_key, agent_id) from env, or error clearly. No silent no-op."""
    api_key = os.environ.get("ASQAV_API_KEY")
    agent_id = os.environ.get("ASQAV_AGENT_ID")
    missing = [
        name
        for name, val in (("ASQAV_API_KEY", api_key), ("ASQAV_AGENT_ID", agent_id))
        if not val
    ]
    if missing:
        print(
            "Error: " + " and ".join(missing) + " must be set to sign hook events.",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)
    return api_key, agent_id  # type: ignore[return-value]


def _sign_event(
    *,
    action_type: str,
    context: dict[str, Any],
    session_id: str,
    compliance_fields: dict[str, Any],
    api_key: str,
    agent_id: str,
) -> Any:
    """Init the SDK, fetch the agent, and sign. Raises on any failure."""
    import asqav

    asqav.init(api_key=api_key)
    agent = asqav.Agent.get(agent_id)
    if session_id:
        agent._session_id = session_id  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in compliance_fields.items() if v is not None}
    kwargs.pop("compliance_mode", None)
    return agent.sign(action_type, context=context, compliance_mode=True, **kwargs)


@hook_app.command("posttool")
def hook_posttool(
    bind_result: bool = typer.Option(
        False,
        "--bind-result",
        help="Bind the tool result (observation:result_bound + result_digest).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Build and print the sign() body as JSON. No network call.",
    ),
) -> None:
    """Sign a Claude Code PostToolUse event (FAIL-OPEN audit).

    PostToolUse runs after the tool already executed, so this is best-effort
    evidence, never a gate: a signer error leaves the action proceeding unsigned.
    The receipt is passive_telemetry/observation. It cannot claim a decision.
    """
    event = _read_event()
    receipt_type = _OBSERVATION_RESULT_BOUND if bind_result else _OBSERVATION
    action_type, context, session_id, fields = _map_event(
        event,
        receipt_type=receipt_type,
        capture_topology="passive_telemetry",
        bind_result=bind_result,
        policy_decision=None,
    )

    if dry_run:
        api_key = os.environ.get("ASQAV_API_KEY", "")
        agent_id = os.environ.get("ASQAV_AGENT_ID", "")
        body = _build_body(
            action_type=action_type,
            context=context,
            session_id=session_id,
            agent_id=agent_id,
            compliance_fields=fields,
        )
        print(json_mod.dumps(body, indent=2, default=str))
        return

    api_key, agent_id = _require_identity()
    try:
        sig = _sign_event(
            action_type=action_type,
            context=context,
            session_id=session_id,
            compliance_fields=fields,
            api_key=api_key,
            agent_id=agent_id,
        )
    except Exception as exc:
        # Fail-open: the tool already ran, so audit failure must not block work.
        print(f"asqav hook: sign failed, proceeding unsigned: {exc}", file=sys.stderr)
        return
    print(f"signed {sig.signature_id}", file=sys.stderr)


@hook_app.command("pretool")
def hook_pretool(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Build and print the sign() body as JSON. No network call.",
    ),
) -> None:
    """Sign a Claude Code PreToolUse event (FAIL-CLOSED gate).

    The host shell decided before the tool ran, so this is a real decision:
    in_process_sdk + protectmcp:decision + policy_decision=permit. Exit 2 BLOCKS
    the tool call when the signer is unreachable or signing errors. There is no
    tool result pre-execution, so no result_digest.
    """
    event = _read_event()
    action_type, context, session_id, fields = _map_event(
        event,
        receipt_type=_DECISION,
        capture_topology="in_process_sdk",
        bind_result=False,
        policy_decision="permit",
    )

    if dry_run:
        api_key = os.environ.get("ASQAV_API_KEY", "")
        agent_id = os.environ.get("ASQAV_AGENT_ID", "")
        body = _build_body(
            action_type=action_type,
            context=context,
            session_id=session_id,
            agent_id=agent_id,
            compliance_fields=fields,
        )
        print(json_mod.dumps(body, indent=2, default=str))
        return

    # Fail-closed: any failure to reach the signer or sign blocks the tool (exit 2).
    api_key = os.environ.get("ASQAV_API_KEY")
    agent_id = os.environ.get("ASQAV_AGENT_ID")
    if not api_key or not agent_id:
        print(
            "asqav hook: ASQAV_API_KEY and ASQAV_AGENT_ID required to gate; blocking.",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)
    try:
        sig = _sign_event(
            action_type=action_type,
            context=context,
            session_id=session_id,
            compliance_fields=fields,
            api_key=api_key,
            agent_id=agent_id,
        )
    except Exception as exc:
        print(f"asqav hook: signer unreachable, blocking tool call: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from exc
    print(f"permit {sig.signature_id}", file=sys.stderr)
