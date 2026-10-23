"""asqav hook: sign Claude Code harness hook events via the cloud sign endpoint.
Not `asqav/hooks.py`. A posttool receipt is passive_telemetry, no :decision."""

from __future__ import annotations

import hashlib
import json as json_mod
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any

try:
    import typer
except ImportError:
    print('CLI requires typer. Install with: pip install "asqav[cli]"', file=sys.stderr)
    # Exit 1 does not block the harness, so the PreToolUse gate must exit 2.
    sys.exit(2 if sys.argv[1:3] == ["hook", "pretool"] else 1)

from asqav.code_authorship import CODE_AUTHORSHIP_WRITE_SCOPE
from asqav.credentials import resolve_api_key

hook_app = typer.Typer(
    name="hook",
    help="Sign Claude Code harness hook events (PostToolUse audit, PreToolUse gate).",
    no_args_is_help=True,
)

#: API-key scope the code-authorship command requires on the configured key.
REQUIRED_SCOPE = CODE_AUTHORSHIP_WRITE_SCOPE

# rule 8 accepts only these two receipt_types under passive_telemetry; a
# receipt is pinned to this set so it cannot claim a decision.
_OBSERVATION = "protectmcp:observation"
_OBSERVATION_RESULT_BOUND = "protectmcp:observation:result_bound"
_DECISION = "protectmcp:decision"
_LIFECYCLE = "protectmcp:lifecycle"

#: Seconds the gate waits for the signer before blocking (env overrides).
_DEADLINE_ENV = "ASQAV_HOOK_DEADLINE_SECONDS"
_DEFAULT_DEADLINE_SECONDS = 5.0
#: Budget verification keeps even when the signer spent the whole deadline.
_MIN_VERIFY_BUDGET_SECONDS = 0.5
#: Where the gate caches the JWK Set it verifies against (env overrides).
_JWKS_CACHE_ENV = "ASQAV_HOOK_JWKS_CACHE"
_JWKS_CACHE_MAX_AGE_SECONDS = 24 * 3600
_JWKS_PAGE_CAP = 8
#: Floor on any socket timeout, so a spent budget still makes one attempt.
_MIN_SOCKET_TIMEOUT_SECONDS = 0.05


class HookDeadlineExceeded(RuntimeError):
    """The signer did not answer inside the gate's deadline."""


def _deadline_seconds() -> float:
    raw = os.environ.get(_DEADLINE_ENV, "").strip()
    if not raw:
        return _DEFAULT_DEADLINE_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_DEADLINE_SECONDS
    return value if value > 0 else _DEFAULT_DEADLINE_SECONDS


def _call_with_deadline(fn, deadline: float, *args: Any, **kwargs: Any) -> Any:
    """Run fn on a daemon thread and give up after deadline seconds.
    A daemon thread never blocks exit; a hung signer cannot hold the harness."""
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = exc

    worker = threading.Thread(target=_target, name="asqav-hook-sign", daemon=True)
    worker.start()
    worker.join(deadline)
    if worker.is_alive():
        raise HookDeadlineExceeded(f"signer did not answer within {deadline:g}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _read_event(*, fail_code: int = 1) -> dict[str, Any]:
    """Parse the hook event JSON from stdin. Exit fail_code on empty or malformed input.

    fail_code is 2 on the pretool gate path (a parse failure must fail CLOSED and
    block the tool) and 1 on the posttool audit path (fail-open is correct there).
    """
    raw = sys.stdin.read()
    if not raw.strip():
        print("Error: no hook event on stdin (expected Claude Code hook JSON).", file=sys.stderr)
        raise typer.Exit(code=fail_code)
    try:
        event = json_mod.loads(raw)
    except Exception as exc:  # noqa: BLE001 - deep nesting and huge ints raise past JSONDecodeError
        print(f"Error: hook event is not valid JSON: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise typer.Exit(code=fail_code) from exc
    if not isinstance(event, dict):
        print("Error: hook event must be a JSON object.", file=sys.stderr)
        raise typer.Exit(code=fail_code)
    return event


    # sha256:<hex> over the JCS-canonical bytes of the tool response.
def _result_digest(tool_response: Any) -> str:
    from asqav.canonicalize import canonicalize

    digest = hashlib.sha256(canonicalize(tool_response)).hexdigest()
    return f"sha256:{digest}"


    # Build the body sign() POSTs from the SDK's own context and body builders.
def _build_body(
    *,
    action_type: str,
    context: dict[str, Any],
    session_id: str,
    agent_id: str,
    compliance_fields: dict[str, Any],
    clear_context: bool = False,
) -> dict[str, Any]:
    from asqav import client as _client
    from asqav.client import _build_sign_body, _prepare_sign_context

    # sign() lowercases the action type and files trace_id in the context before
    # hashing, so the digest the receipt must answer is over that form.
    fields = {k: v for k, v in compliance_fields.items() if v is not None}
    wire_action, wire_context = _prepare_sign_context(
        action_type=action_type, context=context, trace_id=fields.pop("trace_id", None)
    )
    # The wire mode is a module global the SDK resolves at init; the hook pins
    # so a dry run shows the bytes the live hook sends.
    previous = _client._mode
    _client._mode = "full-payload" if clear_context else "hash-only"
    try:
        return _build_sign_body(
            action_type=wire_action,
            context=wire_context,
            session_id=session_id or None,
            agent_id=agent_id,
            compliance_fields=fields or None,
        )
    finally:
        _client._mode = previous


def _map_event(
    event: dict[str, Any],
    *,
    receipt_type: str,
    capture_topology: str,
    bind_result: bool,
    policy_decision: str | None,
) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Map a Claude Code hook event to sign() inputs (action_type, context).
    action_type=tool:<tool_name>; invocation_ref is the tool_use_id."""
    tool_name = event.get("tool_name", "unknown")
    tool_input = event.get("tool_input", {})
    session_id = event.get("session_id", "") or ""
    tool_use_id = event.get("tool_use_id")

    action_type = f"tool:{tool_name}"
    context = {"tool_input": tool_input}
    compliance_fields: dict[str, Any] = {
        "compliance_mode": True,
        "receipt_type": receipt_type,
        "policy_decision": policy_decision or "permit",
        "capture_topology": capture_topology,
        "trace_id": session_id or None,
    }
    if isinstance(tool_use_id, str) and tool_use_id != "":
        compliance_fields["invocation_ref"] = tool_use_id
    if bind_result:
        compliance_fields["result_digest"] = _result_digest(event.get("tool_response"))
    return action_type, context, session_id, compliance_fields


    # Return (api_key, agent_id) from the credential chain and env, or error.
def _require_identity() -> tuple[str, str]:
    api_key = resolve_api_key()
    agent_id = os.environ.get("ASQAV_AGENT_ID")
    missing = [
        name
        for name, val in (("ASQAV_API_KEY", api_key), ("ASQAV_AGENT_ID", agent_id))
        if not val
    ]
    if missing:
        print(
            "Error: " + " and ".join(missing) + " must be set to sign hook events "
            "(run `asqav login` for the API key).",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)
    return api_key, agent_id  # type: ignore[return-value]


    # Init the SDK, fetch the agent, sign; return (receipt, digest POSTed).
def _sign_event(
    *,
    action_type: str,
    context: dict[str, Any],
    session_id: str,
    compliance_fields: dict[str, Any],
    api_key: str,
    agent_id: str,
    clear_context: bool = False,
    timeout: float | None = None,
) -> tuple[Any, str]:
    import asqav

    # Hash-only by default: tool arguments are digested here and never sent;
    # --clear-context opts a deployment into sending them.
    asqav.init(
        api_key=api_key,
        mode="full-payload" if clear_context else "hash-only",
        timeout=timeout,
    )
    agent = asqav.Agent.get(agent_id)
    if session_id:
        agent._session_id = session_id  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in compliance_fields.items() if v is not None}
    kwargs.pop("compliance_mode", None)
    # The digest comes from the body sign() hands to the wire, not the response
    # or a second build that would rerun before-hooks and detectors.
    from asqav import client as _client

    sent: dict[str, str] = {}
    token = _client._sign_body_tap.set(lambda body: sent.update(hash=str(body.get("hash", ""))))
    try:
        sig = agent.sign(action_type, context=context, compliance_mode=True, **kwargs)
    finally:
        _client._sign_body_tap.reset(token)
    return sig, sent.get("hash", "")


def _jwks_url() -> str:
    from urllib.parse import urlsplit

    from asqav import client as _client

    parts = urlsplit(_client._api_base)
    return f"{parts.scheme}://{parts.netloc}/.well-known/jwks.json"


def _jwks_cache_path() -> str:
    return os.environ.get(_JWKS_CACHE_ENV) or os.path.join(
        os.path.expanduser("~"), ".asqav", "jwks-cache.json"
    )


def _fetch_jwks(timeout: float) -> dict[str, Any]:
    """Fetch the published JWK Set, spending at most timeout seconds in total.

    timeout is a whole-call budget, not a per-socket one: a socket timeout multiplied by
    the page cap is what let a paginating host run many times past the gate's deadline.
    """
    import urllib.request

    stop = time.monotonic() + timeout
    base = _jwks_url()
    keys: list[dict[str, Any]] = []
    offset = 0
    for _ in range(_JWKS_PAGE_CAP):
        left = stop - time.monotonic()
        if left <= 0:
            raise HookDeadlineExceeded(f"JWK Set fetch outlived its {timeout:g}s budget")
        url = base if not offset else f"{base}?offset={offset}"
        request = urllib.request.Request(url, headers={"User-Agent": "asqav-hook"})
        with urllib.request.urlopen(
            request, timeout=max(min(left, timeout), _MIN_SOCKET_TIMEOUT_SECONDS)
        ) as response:
            page = json_mod.loads(response.read().decode("utf-8"))
        keys.extend(page.get("keys") or [])
        nxt = page.get("next_offset")
        if not isinstance(nxt, int) or nxt <= offset:
            break
        offset = nxt
    return {"keys": keys}


def _read_jwks_cache() -> dict[str, Any] | None:
    path = _jwks_cache_path()
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json_mod.load(fh)
    except (OSError, ValueError):
        return None
    fetched_at = doc.get("fetched_at")
    if not isinstance(fetched_at, (int, float)):
        return None
    if time.time() - fetched_at > _JWKS_CACHE_MAX_AGE_SECONDS:
        return None
    keys = doc.get("keys")
    return {"keys": keys} if isinstance(keys, list) else None


def _write_jwks_cache(jwks: dict[str, Any]) -> None:
    path = _jwks_cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json_mod.dump({"fetched_at": time.time(), "keys": jwks["keys"]}, fh)
        os.replace(tmp, path)
    except OSError:
        pass


def _load_jwks(timeout: float) -> dict[str, Any]:
    cached = _read_jwks_cache()
    if cached is not None:
        return cached
    fresh = _fetch_jwks(timeout)
    _write_jwks_cache(fresh)
    return fresh


def _keys_of_agent(jwks: dict[str, Any], agent_id: str) -> dict[str, Any]:
    """The published keys whose agent_id is the gated agent, and nothing else.

    The verifier picks a key by thumbprint, then kid, then agent bind, all named by the
    receipt, so a set holding another agent's key lets that agent's signature stand in.
    Handing it only this agent's keys fixes the resolved entry's agent_id by construction.
    """
    keys = [
        k for k in (jwks.get("keys") or []) if isinstance(k, dict) and k.get("agent_id") == agent_id
    ]
    return {"keys": keys}


def _key_absent_reason(
    jwks: dict[str, Any], payload: dict[str, Any], signature: Any = None
) -> str:
    """Empty when the set names the receipt's signing key, else why it does not.

    The signed thumbprint names one key, so it decides first; agent_id and kid answer for a
    set, so they only decide when the directory publishes no thumbprint to compare against.
    """
    keys = [k for k in (jwks.get("keys") or []) if isinstance(k, dict)]
    thumb = payload.get("key_thumbprint")
    if thumb and any(k.get("key_thumbprint") for k in keys):
        if any(k.get("key_thumbprint") == thumb for k in keys):
            return ""
        return f"the gated agent has no published key with thumbprint {thumb}"
    agent_id = payload.get("agent_id")
    if agent_id:
        if any(k.get("agent_id") == agent_id for k in keys):
            return ""
        return f"the published set holds no key for agent {agent_id}"
    return _kid_absent_reason(keys, signature)


def _kid_absent_reason(keys: list[dict[str, Any]], signature: Any) -> str:
    kid = signature.get("kid") if isinstance(signature, dict) else None
    if kid:
        if any(kid in (k.get("kid"), k.get("issuer_id")) for k in keys):
            return ""
        return f"the published set holds no key under kid {kid}"
    return "the receipt names no key to match on (no key_thumbprint, agent_id or kid)"


def _key_present(jwks: dict[str, Any], payload: dict[str, Any], signature: Any = None) -> bool:
    # A cached set may predate the agent's key; the caller refreshes once.
    return not _key_absent_reason(jwks, payload, signature)


def _missing_agent_key(
    jwks: dict[str, Any], agent_id: str, payload: dict[str, Any], signature: Any
) -> str:
    # An empty set is its own reason, not "names no key" (blames the receipt)
    if not jwks["keys"]:
        return f"the published set holds no key for the gated agent {agent_id}"
    return _key_absent_reason(jwks, payload, signature)


def _agent_key_set(
    stop: float, payload: dict[str, Any], signature: Any, agent_id: str
) -> tuple[dict[str, Any], str]:
    """Return (the gated agent's published keys, "") or ({}, why the gate must block).

    The directory is multi-tenant, so only entries published for agent_id may verify a
    permit. A same-org sibling, another org's agent, or a kid that names a sibling then
    finds no key of its own to resolve to.
    """
    left = stop - time.monotonic()
    if left <= 0:
        return {}, "the gate's deadline was spent before the JWK Set could be read"
    jwks = _keys_of_agent(_load_jwks(left), agent_id)
    missing = _missing_agent_key(jwks, agent_id, payload, signature)
    if not missing:
        return jwks, ""
    # One refresh answers a rotation; a set still naming no key ends here with
    # the reason, not a re-read of the directory on every tool call.
    left = stop - time.monotonic()
    if left <= 0:
        return {}, f"the gate's deadline was spent before a refresh could answer: {missing}"
    fresh = _fetch_jwks(left)
    _write_jwks_cache(fresh)
    jwks = _keys_of_agent(fresh, agent_id)
    missing = _missing_agent_key(jwks, agent_id, payload, signature)
    if missing:
        return {}, f"signing key not published: {missing}"
    return jwks, ""


#: Axes that establish the receipt is a platform signature over these bytes.
_PERMIT_AXES = ("signature", "issuer_bind", "key_status", "key_binding")


def _request_binding_failure(
    payload: dict[str, Any],
    *,
    sent_hash: str,
    expected_agent_id: str,
    expected_invocation_ref: str,
) -> str:
    """Empty when the receipt answers this agent, digest and tool call, else why not."""
    if payload.get("agent_id") != expected_agent_id:
        return (
            f"agent mismatch: receipt answers {payload.get('agent_id')!r}, "
            f"gate sent {expected_agent_id!r}"
        )
    receipt_digest = payload.get("hash") or payload.get("action_ref")
    # No digest sent (--clear-context) binds nothing, so an empty match blocks
    if not sent_hash or receipt_digest != sent_hash:
        return "receipt digest does not match the digest the gate sent"
    # The cloud signs invocation_ref at the payload top level; a receipt for
    # another call with the same tool, input and session carries another one.
    got = payload.get("invocation_ref")
    if got != expected_invocation_ref:
        return (
            "receipt invocation_ref does not match this tool call "
            f"({'missing' if got is None else 'a different call'})"
        )
    return ""


def _freshness_failure(payload: dict[str, Any]) -> str:
    try:
        issued = datetime.fromisoformat(str(payload.get("issued_at", "")))
        if issued.tzinfo is None:
            issued = issued.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - issued).total_seconds()
    except (TypeError, ValueError):
        return "receipt issued_at is missing or unparseable"
    if not 0 <= age <= _deadline_seconds():
        return f"receipt issued_at outside the gate deadline ({age:.1f}s old)"
    return ""


def _verify_returned_receipt(
    sig: Any,
    timeout: float,
    *,
    sent_hash: str,
    expected_agent_id: str,
    expected_invocation_ref: str,
) -> tuple[bool, str]:
    """Verify the signer receipt against the published JWK Set: (ok, reason).
    Signature axes must pass and the receipt must answer THIS request, fresh."""
    from asqav.verifier import verify_receipt as _vr

    payload = getattr(sig, "payload", None)
    signature = getattr(sig, "signature", None)
    if not isinstance(payload, dict) or not isinstance(signature, dict):
        return False, "signer returned no verifiable receipt (payload or signature missing)"
    anchors = getattr(sig, "anchors", None) or []
    doc = {"payload": payload, "signature": signature, "anchors": anchors}
    jwks, why = _agent_key_set(time.monotonic() + timeout, payload, signature, expected_agent_id)
    if why:
        return False, why
    report = _vr.run_structured(doc, jwks, None)
    axes = {a["name"]: a for a in report.get("axes", [])}
    for name in _PERMIT_AXES:
        axis = axes.get(name)
        if axis is None or axis["result"] != "PASS":
            note = axis["note"] if axis else "axis missing"
            return False, f"{name}: {note}"
    failure = _request_binding_failure(
        payload,
        sent_hash=sent_hash,
        expected_agent_id=expected_agent_id,
        expected_invocation_ref=expected_invocation_ref,
    ) or _freshness_failure(payload)
    if failure:
        return False, failure
    return True, f"signature verified against the published JWK Set (kid {signature.get('kid')})"


#: Wire decisions that deny the call; the gate blocks, never announces success.
_BLOCKING_DECISIONS = ("deny", "rate_limit")


def _wire_decision(sig: Any) -> Any:
    payload = getattr(sig, "payload", None)
    return payload.get("decision") if isinstance(payload, dict) else None


def _decision_label(sig: Any) -> str:
    # The platform records in-process capture as an observation (wire decision
    # "observation"); only a real gate decision earns the word permit.
    return "permit" if _wire_decision(sig) == "allow" else "signed"


@hook_app.command("posttool")
def hook_posttool(
    bind_result: bool = typer.Option(
        True,
        "--bind-result/--no-bind-result",
        help=(
            "Bind the tool result (observation:result_bound + result_digest). On by"
            " default: PostToolUse runs after the tool, so every receipt carries the"
            " outcome."
        ),
    ),
    clear_context: bool = typer.Option(
        False,
        "--clear-context",
        help="Send the tool arguments in clear for the platform to hold (default: digest only).",
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
        api_key = resolve_api_key() or ""
        agent_id = os.environ.get("ASQAV_AGENT_ID", "")
        body = _build_body(
            action_type=action_type,
            context=context,
            session_id=session_id,
            agent_id=agent_id,
            compliance_fields=fields,
            clear_context=clear_context,
        )
        print(json_mod.dumps(body, indent=2, default=str))
        return

    api_key, agent_id = _require_identity()
    deadline = _deadline_seconds()
    try:
        sig, _sent_hash = _call_with_deadline(
            _sign_event,
            deadline,
            action_type=action_type,
            context=context,
            session_id=session_id,
            compliance_fields=fields,
            api_key=api_key,
            agent_id=agent_id,
            clear_context=clear_context,
            timeout=deadline,
        )
    except HookDeadlineExceeded as exc:
        print(f"asqav hook: {exc}; proceeding unsigned", file=sys.stderr)
        return
    except Exception as exc:
        # Fail-open: the tool already ran, so audit failure must not block work.
        print(f"asqav hook: sign failed, proceeding unsigned: {exc}", file=sys.stderr)
        return
    print(f"signed {sig.signature_id}", file=sys.stderr)


def _tool_use_id(event: dict[str, Any]) -> str | None:
    # The id _map_event forwards as invocation_ref; None when the host sent none
    value = event.get("tool_use_id")
    return value if isinstance(value, str) and value != "" else None


def _enforce_receipt(
    sig: Any,
    *,
    sent_hash: str,
    agent_id: str,
    invocation_ref: str,
    deadline: float,
    started: float,
) -> None:
    """Verify the returned receipt and exit 2 unless it is a permit for THIS call."""
    remaining = max(deadline - (time.monotonic() - started), _MIN_VERIFY_BUDGET_SECONDS)
    try:
        ok, reason = _call_with_deadline(
            _verify_returned_receipt, remaining, sig, remaining,
            sent_hash=sent_hash, expected_agent_id=agent_id,
            expected_invocation_ref=invocation_ref,
        )
    except HookDeadlineExceeded:
        ok, reason = False, (
            f"verification did not finish inside the remaining {remaining:g}s of the "
            f"{deadline:g}s deadline"
        )
    except Exception as exc:  # noqa: BLE001 - any failure here is a blocked call
        ok, reason = False, f"could not verify the returned receipt: {exc}"
    if not ok:
        print(
            f"asqav hook: receipt {sig.signature_id} did not verify ({reason}); blocking tool call",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)
    decision = _wire_decision(sig)
    if decision in _BLOCKING_DECISIONS:
        # A verified receipt carrying a denial is still a denial; exit 2 blocks.
        print(f"blocked {sig.signature_id}: {decision}", file=sys.stderr)
        raise typer.Exit(code=2)
    if decision != "allow":
        # Only a permit opens the gate; any other non-blocking decision exits 2.
        print(
            f"asqav hook: non-permit decision {decision!r} on {sig.signature_id}; "
            "blocking tool call",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)
    print(f"{_decision_label(sig)} {sig.signature_id}", file=sys.stderr)


@hook_app.command("pretool")
def hook_pretool(
    clear_context: bool = typer.Option(
        False,
        "--clear-context",
        help="Send the tool arguments in clear for the platform to hold (default: digest only).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Build and print the sign() body as JSON. No network call.",
    ),
) -> None:
    """Sign a Claude Code PreToolUse event (FAIL-CLOSED gate).
    Exit 2 blocks the tool on any signer, deadline, verify or event error."""
    try:
        _pretool_gate(clear_context=clear_context, dry_run=dry_run)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 - exit 1 lets the tool run, so nothing escapes
        print(
            f"asqav hook: unexpected {type(exc).__name__}, blocking tool call: {exc}",
            file=sys.stderr,
        )
        raise typer.Exit(code=2) from exc


def _pretool_gate(*, clear_context: bool, dry_run: bool) -> None:
    started = time.monotonic()
    event = _read_event(fail_code=2)
    action_type, context, session_id, fields = _map_event(
        event,
        receipt_type=_DECISION,
        capture_topology="in_process_sdk",
        bind_result=False,
        policy_decision="permit",
    )

    if dry_run:
        api_key = resolve_api_key() or ""
        agent_id = os.environ.get("ASQAV_AGENT_ID", "")
        body = _build_body(
            action_type=action_type,
            context=context,
            session_id=session_id,
            agent_id=agent_id,
            compliance_fields=fields,
            clear_context=clear_context,
        )
        print(json_mod.dumps(body, indent=2, default=str))
        return

    # Fail-closed: no tool_use_id leaves no way to bind a permit to this call.
    invocation_ref = _tool_use_id(event)
    if invocation_ref is None:
        print(
            "asqav hook: the event carries no tool_use_id, so no receipt can be bound to "
            "this call; blocking tool call",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)
    # Fail-closed: any failure to reach or sign with the signer blocks (exit 2).
    api_key = resolve_api_key()
    agent_id = os.environ.get("ASQAV_AGENT_ID")
    if not api_key or not agent_id:
        print(
            "asqav hook: ASQAV_API_KEY (or `asqav login`) and ASQAV_AGENT_ID required "
            "to gate; blocking.",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)
    deadline = _deadline_seconds()
    try:
        sig, sent_hash = _call_with_deadline(
            _sign_event,
            deadline,
            action_type=action_type,
            context=context,
            session_id=session_id,
            compliance_fields=fields,
            api_key=api_key,
            agent_id=agent_id,
            clear_context=clear_context,
            timeout=deadline,
        )
    except HookDeadlineExceeded as exc:
        print(f"asqav hook: {exc}; blocking tool call", file=sys.stderr)
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        print(f"asqav hook: signer unreachable, blocking tool call: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from exc

    try:
        _enforce_receipt(
            sig,
            sent_hash=sent_hash,
            agent_id=agent_id,
            invocation_ref=invocation_ref,
            deadline=deadline,
            started=started,
        )
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 - nothing after signing may exit open
        print(
            f"asqav hook: could not gate receipt {getattr(sig, 'signature_id', '?')}, "
            f"blocking tool call: {exc}",
            file=sys.stderr,
        )
        raise typer.Exit(code=2) from exc


@hook_app.command("code-authorship")
def hook_code_authorship(
    repo: str = typer.Option(..., "--repo", help="Repository as owner/name."),
    commit_sha: str = typer.Option(
        ..., "--commit-sha", help="Head commit sha the change is bound to."
    ),
    base_sha: str = typer.Option("", "--base-sha", help="Base commit sha for the advisory diff."),
    change_class: str = typer.Option(
        "write", "--change-class", help="read | write | delete | execute | deploy."
    ),
    author: str = typer.Option("", "--author", help="Producer-asserted author identity."),
    anchor: str = typer.Option("", "--anchor", help="Optional anchor reference (PR url, ticket)."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Compute the advisory digest and print the request body. No network call.",
    ),
) -> None:
    """Record code authorship via POST /v1/code-authorship.
    Posts an ADVISORY change digest, which the server recomputes."""
    from asqav.code_authorship import compute_advisory_digest, submit_code_authorship

    advisory = compute_advisory_digest(base_sha or None, commit_sha)

    if dry_run:
        body = {
            "repo": repo,
            "commit_sha": commit_sha,
            "change_digest": advisory,
            "change_class": change_class,
        }
        if base_sha:
            body["base_sha"] = base_sha
        if author:
            body["author"] = author
        if anchor:
            body["anchor"] = anchor
        print(json_mod.dumps(body, indent=2, default=str))
        return

    api_key = resolve_api_key()
    if not api_key:
        print(
            "Error: ASQAV_API_KEY (or `asqav login`) must be set, with the "
            f"{REQUIRED_SCOPE} scope, to record code authorship.",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    import asqav

    asqav.init(api_key=api_key)
    try:
        result = submit_code_authorship(
            repo=repo,
            commit_sha=commit_sha,
            base_sha=base_sha or None,
            change_digest=advisory,
            change_class=change_class,
            author=author or None,
            anchor=anchor or None,
        )
    except Exception as exc:
        print(f"asqav hook: code-authorship recording failed: {exc}", file=sys.stderr)
        raise typer.Exit(code=1) from exc

    print(
        f"capture_layer={result.capture_layer} digest_match={result.digest_match}",
        file=sys.stderr,
    )
    if result.subject_digest:
        print(f"server subject digest: {result.subject_digest}", file=sys.stderr)
