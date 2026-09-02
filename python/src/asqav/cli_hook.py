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
import threading
import time
from typing import Any

try:
    import typer
except ImportError:
    print("CLI requires typer. Install with: pip install asqav[cli]")
    sys.exit(1)

from asqav.code_authorship import CODE_AUTHORSHIP_WRITE_SCOPE
from asqav.credentials import resolve_api_key

hook_app = typer.Typer(
    name="hook",
    help="Sign Claude Code harness hook events (PostToolUse audit, PreToolUse gate).",
    no_args_is_help=True,
)

#: API-key scope the code-authorship command requires on the configured key.
REQUIRED_SCOPE = CODE_AUTHORSHIP_WRITE_SCOPE

# rule 8 accepts only these two receipt_types under passive_telemetry. A posttool
# receipt is structurally pinned to this set so it cannot claim a decision.
_OBSERVATION = "protectmcp:observation"
_OBSERVATION_RESULT_BOUND = "protectmcp:observation:result_bound"
_DECISION = "protectmcp:decision"
_LIFECYCLE = "protectmcp:lifecycle"

#: Seconds the gate waits for the signer before it blocks; ASQAV_HOOK_DEADLINE_SECONDS overrides.
_DEADLINE_ENV = "ASQAV_HOOK_DEADLINE_SECONDS"
_DEFAULT_DEADLINE_SECONDS = 5.0
#: Budget the verification phase keeps even when the signer spent the whole deadline.
_MIN_VERIFY_BUDGET_SECONDS = 0.5
#: Where the gate keeps the JWK Set it verifies permits against; ASQAV_HOOK_JWKS_CACHE overrides.
_JWKS_CACHE_ENV = "ASQAV_HOOK_JWKS_CACHE"
_JWKS_CACHE_MAX_AGE_SECONDS = 24 * 3600
_JWKS_PAGE_CAP = 8
#: Floor on any single socket timeout, so a nearly-spent budget still makes one honest attempt.
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

    A daemon thread never blocks interpreter exit, so a signer that hangs past the
    deadline cannot hold the harness hostage until its own 600-second limit.
    """
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
    except json_mod.JSONDecodeError as exc:
        print(f"Error: hook event is not valid JSON: {exc}", file=sys.stderr)
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


    # Build the same body sign() POSTs, reusing the SDK builder (no re-impl).
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
    from asqav.client import _build_sign_body

    # The wire mode is a module global the SDK resolves at init; the hook pins it
    # explicitly so a dry run shows the bytes the live hook sends.
    previous = _client._mode
    _client._mode = "full-payload" if clear_context else "hash-only"
    try:
        return _build_sign_body(
            action_type=action_type,
            context=context,
            session_id=session_id or None,
            agent_id=agent_id,
            compliance_fields={k: v for k, v in compliance_fields.items() if v is not None}
            or None,
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


    # Return (api_key, agent_id) from the credential chain + env, or error clearly.
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


    # Init the SDK, fetch the agent, and sign. Raises on any failure.
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
) -> Any:
    import asqav

    # Hash-only by default: the tool arguments are digested here and never leave the
    # machine; --clear-context opts a deployment into sending them for the platform to hold.
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
    return agent.sign(action_type, context=context, compliance_mode=True, **kwargs)


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
        return f"the published set holds no key with thumbprint {thumb}"
    agent_id = payload.get("agent_id")
    if agent_id:
        if any(k.get("agent_id") == agent_id for k in keys):
            return ""
        return f"the published set holds no key for agent {agent_id}"
    kid = signature.get("kid") if isinstance(signature, dict) else None
    if kid:
        if any(kid in (k.get("kid"), k.get("issuer_id")) for k in keys):
            return ""
        return f"the published set holds no key under kid {kid}"
    return "the receipt names no key to match on (no key_thumbprint, agent_id or kid)"


def _key_present(jwks: dict[str, Any], payload: dict[str, Any], signature: Any = None) -> bool:
    # A cached set may predate the agent's key; the caller refreshes once on a miss.
    return not _key_absent_reason(jwks, payload, signature)


#: Axes that establish the returned receipt really is a platform signature over these bytes.
_PERMIT_AXES = ("signature", "issuer_bind", "key_status", "key_binding")


def _verify_returned_receipt(sig: Any, timeout: float) -> tuple[bool, str]:
    """Verify the receipt the signer returned against the published JWK Set.

    Returns (ok, reason). The signature-related axes must all PASS; anchors and the
    chain are outside a single receipt's reach and are not required here.
    """
    from asqav.verifier import verify_receipt as _vr

    payload = getattr(sig, "payload", None)
    signature = getattr(sig, "signature", None)
    if not isinstance(payload, dict) or not isinstance(signature, dict):
        return False, "signer returned no verifiable receipt (payload or signature missing)"
    anchors = getattr(sig, "anchors", None) or []
    doc = {"payload": payload, "signature": signature, "anchors": anchors}
    stop = time.monotonic() + timeout
    left = stop - time.monotonic()
    if left <= 0:
        return False, "the gate's deadline was spent before the JWK Set could be read"
    jwks = _load_jwks(left)
    missing = _key_absent_reason(jwks, payload, signature)
    if missing:
        # One refresh answers a rotation; a set that still names no key ends here with the
        # reason, instead of re-reading the whole directory on every tool call.
        left = stop - time.monotonic()
        if left <= 0:
            return False, f"the gate's deadline was spent before a refresh could answer: {missing}"
        jwks = _fetch_jwks(left)
        _write_jwks_cache(jwks)
        missing = _key_absent_reason(jwks, payload, signature)
        if missing:
            return False, f"signing key not published: {missing}"
    report = _vr.run_structured(doc, jwks, None)
    axes = {a["name"]: a for a in report.get("axes", [])}
    for name in _PERMIT_AXES:
        axis = axes.get(name)
        if axis is None or axis["result"] != "PASS":
            note = axis["note"] if axis else "axis missing"
            return False, f"{name}: {note}"
    return True, f"signature verified against the published JWK Set (kid {signature.get('kid')})"


#: Wire decisions that deny the call; the gate must block on them, never announce a success.
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
        False,
        "--bind-result",
        help="Bind the tool result (observation:result_bound + result_digest).",
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
        sig = _call_with_deadline(
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

    The host shell decided before the tool ran, so this is a real decision:
    in_process_sdk + protectmcp:decision + policy_decision=permit. Exit 2 BLOCKS
    the tool call when the signer is unreachable, signing errors, the signer does
    not answer inside the deadline, or the returned receipt does not verify
    against the published JWK Set. There is no tool result pre-execution, so no
    result_digest. A malformed/empty event also blocks (exit 2): the gate fails
    closed on every failure path, never open.
    """
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

    # Fail-closed: any failure to reach the signer or sign blocks the tool (exit 2).
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
        sig = _call_with_deadline(
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

    # The permit is only as good as its signature; verify it before letting the tool run.
    remaining = max(deadline - (time.monotonic() - started), _MIN_VERIFY_BUDGET_SECONDS)
    try:
        ok, reason = _call_with_deadline(_verify_returned_receipt, remaining, sig, remaining)
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
        # A verified receipt that carries a denial is still a denial; exit 2 blocks the tool.
        print(f"blocked {sig.signature_id}: {decision}", file=sys.stderr)
        raise typer.Exit(code=2)
    print(f"{_decision_label(sig)} {sig.signature_id}", file=sys.stderr)


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

    Computes an ADVISORY change digest (sha256 of ``git diff base..head``) and
    posts it with the repo and commit. The server re-fetches the commit,
    recomputes the canonical diff, and signs the authoritative in-toto
    Statement. ``subject[0].digest.sha256`` is the server digest, and
    ``digest_match`` reports whether the advisory value agreed. Requires a key
    with the ``code_authorship:write`` scope.
    """
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
