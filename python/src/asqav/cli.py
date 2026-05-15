"""
asqav CLI - AI agent governance from the terminal.

Every command wraps a public Python API; both surfaces accept the same
arguments and return the same data, so anything callable here is also
callable programmatically.

Requires: pip install asqav[cli]

Commands:
    asqav verify <signature_id>                - Verify a signature (public)
    asqav agents list / create <name> / revoke - Manage agents
    asqav sessions list / end <session_id>     - Manage sessions
    asqav policies list / create / delete      - Manage org policies (Pro)
    asqav webhooks list / create / delete      - Manage webhooks (Pro)
    asqav replay <agent_id> <session_id>       - Replay a session's audit trail
    asqav preflight <agent_id> <action>        - Pre-flight (revocation + policy)
    asqav budget check / record                - Budget gate / signed spend record
    asqav approve <session_id> <entity_id>     - Sign off a multi-party session
    asqav compliance frameworks / export       - List frameworks / export bundle
    asqav sync                                 - Sync local queue to API
    asqav queue list / count / clear           - Manage local queue
    asqav demo / quickstart / doctor           - Onboarding + diagnostics
    asqav --version                            - Show version
"""

from __future__ import annotations

import sys

try:
    import typer
except ImportError:
    print("CLI requires typer. Install with: pip install asqav[cli]")
    sys.exit(1)

from asqav import __version__

app = typer.Typer(
    name="asqav",
    help="AI agent governance - audit trails, policy enforcement, compliance.",
    no_args_is_help=True,
    add_completion=False,
)

agents_app = typer.Typer(
    name="agents",
    help="Manage agents.",
    no_args_is_help=True,
)
app.add_typer(agents_app, name="agents")

queue_app = typer.Typer(
    name="queue",
    help="Manage local offline queue.",
    no_args_is_help=True,
)
app.add_typer(queue_app, name="queue")

budget_app = typer.Typer(
    name="budget",
    help="Check or record agent spend (wraps asqav.BudgetTracker).",
    no_args_is_help=True,
)
app.add_typer(budget_app, name="budget")

compliance_app = typer.Typer(
    name="compliance",
    help="Compliance bundle export (wraps asqav.compliance.export_bundle).",
    no_args_is_help=True,
)
app.add_typer(compliance_app, name="compliance")


def _version_callback(value: bool) -> None:
    if value:
        print(f"asqav {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """AI agent governance - audit trails, policy enforcement, compliance."""


@app.command()
def verify(
    signature_id: str = typer.Argument(help="Signature ID to verify."),
    output: str = typer.Option(
        "text", "--output", "-o", help="Output format: text or json."
    ),
) -> None:
    """Verify a signature by ID (public, no auth needed).

    With ``--output json`` returns the full ``VerificationResponse`` plus
    the IETF profile sub-axes (`chain_valid`, `anchor_status_ots`,
    `anchor_status_rfc3161`, `signed_at_skew_seconds`, `missing_fields`)
    when the cloud emitted them.
    """
    import json as json_mod

    from asqav import APIError, verify_signature

    try:
        result = verify_signature(signature_id)
    except APIError as exc:
        if exc.status_code == 404:
            print(f"Signature not found: {signature_id}")
            raise typer.Exit(code=1)
        print(f"Error: {exc}")
        raise typer.Exit(code=1)

    if output == "json":
        detail = result.verification_detail
        payload = {
            "signature_id": result.signature_id,
            "agent_id": result.agent_id,
            "agent_name": result.agent_name,
            "action_id": result.action_id,
            "action_type": result.action_type,
            "algorithm": result.algorithm,
            "signed_at": result.signed_at,
            "verified": result.verified,
            "verification_url": result.verification_url,
            "verification_detail": (
                {
                    "signer_key_match": detail.signer_key_match,
                    "signature_valid": detail.signature_valid,
                    "algorithm_match": detail.algorithm_match,
                    "agent_active": detail.agent_active,
                    "validation_label": detail.validation_label,
                    "signed_at_skew_seconds": detail.signed_at_skew_seconds,
                    "chain_valid": detail.chain_valid,
                    "anchor_status_ots": detail.anchor_status_ots,
                    "anchor_status_rfc3161": detail.anchor_status_rfc3161,
                    "missing_fields": detail.missing_fields,
                }
                if detail
                else None
            ),
        }
        print(json_mod.dumps(payload, indent=2, default=str))
        return

    status = "valid" if result.verified else "invalid"
    print(f"Signature: {status}")
    print(f"Agent: {result.agent_name} ({result.agent_id})")
    print(f"Action: {result.action_type}")
    print(f"Algorithm: {result.algorithm}")
    detail = result.verification_detail
    if detail is not None:
        print(f"Validation: {detail.validation_label}")
        if detail.chain_valid is not None:
            print(f"Chain valid: {detail.chain_valid}")
        if detail.anchor_status_ots is not None:
            print(f"Anchor (OTS): {detail.anchor_status_ots}")
        if detail.anchor_status_rfc3161 is not None:
            print(f"Anchor (RFC 3161): {detail.anchor_status_rfc3161}")
        if detail.signed_at_skew_seconds is not None:
            print(f"signed_at skew: {detail.signed_at_skew_seconds}s")
        if detail.missing_fields:
            print(f"Missing fields: {', '.join(detail.missing_fields)}")


@app.command()
def sign(
    action_type: str = typer.Option(..., "--action-type", help="Action type, e.g. payment.wire_transfer."),
    action_id: str = typer.Option("", "--action-id", help="Stable client-side action id (the cloud allocates one when empty)."),
    agent_id: str = typer.Option(..., "--agent-id", help="Agent that signs the receipt."),
    compliance_mode: bool = typer.Option(
        False,
        "--compliance-mode/--no-compliance-mode",
        help="Emit the receipt under the IETF Compliance Receipts profile.",
    ),
    action_ref: str = typer.Option("", "--action-ref", help="`sha256:<hex>` over the canonical Action object. Auto-computed when omitted with --action-json."),
    action_json: str = typer.Option(
        "",
        "--action-json",
        help="Path to a JSON file or '-' for stdin. The SDK derives `action_ref` from the JCS bytes when --action-ref is empty.",
    ),
    sandbox_state: str = typer.Option("", "--sandbox-state", help="enabled|disabled|unavailable."),
    iteration_id: str = typer.Option("", "--iteration-id", help="Logical task iteration id (distinct from session_id)."),
    risk_class: str = typer.Option("", "--risk-class", help="low|medium|high|unknown."),
    incident_class: str = typer.Option("", "--incident-class", help="DORA RTS JC 2024-33 Annex II field 3.23: cybersecurity_related|process_failure|system_failure|external_event|payment_related|other."),
    issuer_id: str = typer.Option("", "--issuer-id", help="Legal entity issuing this receipt; overrides server resolution."),
    receipt_type: str = typer.Option(
        "protectmcp:decision",
        "--receipt-type",
        help="protectmcp:decision|protectmcp:restraint|protectmcp:lifecycle.",
    ),
    reason: str = typer.Option("", "--reason", help="Required when --policy-decision is deny|rate_limit."),
    policy_decision: str = typer.Option(
        "permit",
        "--policy-decision",
        help="permit|deny|rate_limit.",
    ),
    algorithm: str = typer.Option(
        "ml-dsa-65",
        "--algorithm",
        help="Signing algorithm: ml-dsa-65|ed25519|es256.",
    ),
    session_id: str = typer.Option("", "--session-id", help="Bind the record to an existing session."),
    valid_seconds: int = typer.Option(0, "--valid-seconds", help="Validity window in seconds (0 = no expiry)."),
    policy_artefact: str = typer.Option(
        "",
        "--policy-artefact",
        help="Path to a JSON file or '-' for stdin. The cloud retains it; the receipt's `policy_digest` is its content hash.",
    ),
    output: str = typer.Option("text", "--output", "-o", help="Output format: text or json."),
) -> None:
    """Issue a Compliance Receipt via the cloud `/agents/{id}/sign` route.

    Mirrors :py:meth:`asqav.Agent.sign` end-to-end so any flag here maps
    1:1 to a kwarg on the SDK call. Reads ``--action-json`` from a path
    or stdin when ``-`` is passed; the SDK then computes ``action_ref``
    from the JCS bytes if ``--action-ref`` was not supplied.
    """
    import json as json_mod

    _init_sdk()
    from asqav import Agent, APIError

    # 1. Load action JSON (file, stdin, or empty).
    action_obj: dict = {}
    if action_json:
        try:
            if action_json == "-":
                action_obj = json_mod.loads(sys.stdin.read())
            else:
                with open(action_json) as f:
                    action_obj = json_mod.load(f)
        except (OSError, json_mod.JSONDecodeError) as exc:
            print(f"Error reading action JSON: {exc}")
            raise typer.Exit(code=1)

    # 2. Load policy artefact (sent server-side; the cloud stores it and
    # stamps `policy_digest` on the receipt).
    policy_artefact_obj: dict | None = None
    if policy_artefact:
        try:
            if policy_artefact == "-":
                policy_artefact_obj = json_mod.loads(sys.stdin.read())
            else:
                with open(policy_artefact) as f:
                    policy_artefact_obj = json_mod.load(f)
        except (OSError, json_mod.JSONDecodeError) as exc:
            print(f"Error reading policy artefact: {exc}")
            raise typer.Exit(code=1)

    # 3. Validate policy_decision/reason coupling client-side. The SDK
    # also validates; this gives a clearer CLI error.
    if policy_decision in {"deny", "rate_limit"} and not reason:
        print(
            "Error: --reason is required when --policy-decision is deny or rate_limit."
        )
        raise typer.Exit(code=1)

    try:
        agent = Agent.get(agent_id)
    except APIError as exc:
        print(f"Error fetching agent: {exc}")
        raise typer.Exit(code=1)

    # 4. Build kwargs. Empty strings -> None so the SDK applies defaults.
    sign_kwargs: dict = {
        "compliance_mode": compliance_mode,
        "policy_decision": policy_decision,
    }
    if action_ref:
        sign_kwargs["action_ref"] = action_ref
    if sandbox_state:
        sign_kwargs["sandbox_state"] = sandbox_state
    if iteration_id:
        sign_kwargs["iteration_id"] = iteration_id
    if risk_class:
        sign_kwargs["risk_class"] = risk_class
    if incident_class:
        sign_kwargs["incident_class"] = incident_class
    if issuer_id:
        sign_kwargs["issuer_id"] = issuer_id
    if receipt_type:
        sign_kwargs["receipt_type"] = receipt_type
    if reason:
        sign_kwargs["reason"] = reason
    if valid_seconds > 0:
        sign_kwargs["valid_seconds"] = valid_seconds

    context = action_obj or None
    if action_id and context is not None:
        context = {**context, "_action_id": action_id}
    elif action_id:
        context = {"_action_id": action_id}

    if policy_artefact_obj is not None and context is not None:
        context = {**context, "_policy_artefact": policy_artefact_obj}
    elif policy_artefact_obj is not None:
        context = {"_policy_artefact": policy_artefact_obj}

    if session_id:
        # Internal attribute the SDK consults when building the body.
        agent._session_id = session_id  # type: ignore[attr-defined]

    if algorithm and algorithm != getattr(agent, "algorithm", None):
        # Algorithm is bound server-side; surface mismatch so callers see intent vs actual.
        print(
            f"Warning: --algorithm={algorithm} differs from agent.algorithm="
            f"{agent.algorithm}. The cloud signs under the agent's algorithm."
        )

    try:
        sig = agent.sign(action_type, context=context, **sign_kwargs)
    except (APIError, ValueError) as exc:
        print(f"Error signing: {exc}")
        raise typer.Exit(code=1)

    if output == "json":
        payload = {
            "signature_id": sig.signature_id,
            "action_id": sig.action_id,
            "algorithm": sig.algorithm,
            "verification_url": sig.verification_url,
            "signed_at": sig.timestamp,
            "policy_digest": sig.policy_digest,
            "policy_decision": sig.policy_decision,
            "compliance_mode": sig.compliance_mode,
            "receipt_type": sig.receipt_type,
            "action_ref": sig.action_ref,
            "issuer_id": sig.issuer_id,
            "previous_receipt_hash": sig.previous_receipt_hash,
            "anchor_status": (
                sig.bitcoin_anchor.status if sig.bitcoin_anchor else None
            ),
            "rfc3161_tsa": sig.rfc3161_tsa,
            "rfc3161_serial": sig.rfc3161_serial,
        }
        print(json_mod.dumps(payload, indent=2, default=str))
        return

    print(f"signature_id: {sig.signature_id}")
    print(f"action_id:    {sig.action_id}")
    print(f"algorithm:    {sig.algorithm}")
    print(f"signed_at:    {sig.timestamp}")
    if sig.policy_digest:
        print(f"policy_digest: {sig.policy_digest}")
    if sig.compliance_mode:
        print("compliance_mode: True")
        if sig.receipt_type:
            print(f"receipt_type:  {sig.receipt_type}")
        if sig.action_ref:
            print(f"action_ref:    {sig.action_ref}")
        if sig.previous_receipt_hash:
            print(f"previous_receipt_hash: {sig.previous_receipt_hash}")
    if sig.bitcoin_anchor:
        print(f"anchor:       {sig.bitcoin_anchor.status}")
    print(f"verify_url:   {sig.verification_url}")


@app.command()
def replay(
    agent_id: str = typer.Argument(
        default="",
        help="Agent ID. Omit when using --bundle.",
    ),
    session_id: str = typer.Argument(
        default="",
        help="Session ID to replay. Omit when using --bundle.",
    ),
    bundle: str = typer.Option(
        "",
        "--bundle",
        "-b",
        help="Replay from a compliance bundle JSON file (offline).",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Print the timeline as JSON instead of a human-readable summary.",
    ),
    output: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Write the timeline to FILE (JSON). Implies --json.",
    ),
) -> None:
    """Replay an agent's audit trail. Wraps asqav.replay() / asqav.replay_from_bundle().

    Online:
        asqav replay agt_x7y8z9 sess_abc123
    Offline (from a bundle):
        asqav replay --bundle compliance-bundle.json
    """
    import json as json_mod

    if bundle:
        from asqav import ComplianceBundle, replay_from_bundle

        try:
            with open(bundle) as f:
                data = json_mod.load(f)
        except OSError as exc:
            print(f"Error reading bundle: {exc}")
            raise typer.Exit(code=1)

        try:
            cb = ComplianceBundle(
                framework=data.get("framework", "unknown"),
                framework_metadata=data.get("framework_metadata", {}),
                generated_at=data.get("generated_at", ""),
                merkle_root=data.get("merkle_root", ""),
                receipt_count=data.get("receipt_count", 0),
                receipts=data.get("receipts", []),
                verification=data.get("verification", {}),
            )
            timeline = replay_from_bundle(cb)
        except Exception as exc:
            print(f"Error replaying bundle: {exc}")
            raise typer.Exit(code=1)
    else:
        if not agent_id or not session_id:
            print(
                "Error: agent_id and session_id are required for online replay. "
                "Use --bundle to replay from a compliance bundle file."
            )
            raise typer.Exit(code=1)

        _init_sdk()
        _require_tier("pro")

        from asqav import APIError
        from asqav import replay as replay_api

        try:
            timeline = replay_api(agent_id, session_id)
        except APIError as exc:
            print(f"Error: {exc}")
            raise typer.Exit(code=1)

    if output:
        timeline.to_file(output)
        print(f"Wrote timeline to {output}")
        return

    if json_out:
        print(timeline.to_json())
        return

    # Human-readable: summary then color-coded steps.
    print(timeline.summary())

    if not timeline.chain_integrity:
        raise typer.Exit(code=1)


@agents_app.command("list")
def agents_list() -> None:
    """List all agents for your organization."""
    import os

    api_key = os.environ.get("ASQAV_API_KEY")
    if not api_key:
        print("Error: ASQAV_API_KEY environment variable required.")
        raise typer.Exit(code=1)

    import asqav

    asqav.init(api_key=api_key)

    from asqav.client import _get

    try:
        data = _get("/agents")
    except Exception as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)

    agents = data if isinstance(data, list) else data.get("agents", [])
    if not agents:
        print("No agents found.")
        return

    for agent in agents:
        name = agent.get("name", "unknown")
        agent_id = agent.get("agent_id", "unknown")
        algorithm = agent.get("algorithm", "unknown")
        print(f"  {name} ({agent_id}) [{algorithm}]")


@agents_app.command("create")
def agents_create(name: str = typer.Argument(help="Name for the new agent.")) -> None:
    """Create a new agent."""
    import os

    api_key = os.environ.get("ASQAV_API_KEY")
    if not api_key:
        print("Error: ASQAV_API_KEY environment variable required.")
        raise typer.Exit(code=1)

    import asqav

    asqav.init(api_key=api_key)

    try:
        agent = asqav.Agent.create(name)
    except Exception as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)

    print(f"Agent created: {agent.name} ({agent.agent_id})")
    print(f"Algorithm: {agent.algorithm}")
    print(f"Key ID: {agent.key_id}")


@agents_app.command("revoke")
def agents_revoke(
    agent_id: str = typer.Argument(help="Agent ID to revoke."),
    reason: str = typer.Option("manual", "--reason", help="Reason recorded in the audit trail."),
) -> None:
    """Revoke an agent's credentials. Irreversible."""
    _init_sdk()
    from asqav import Agent, APIError

    try:
        agent = Agent.get(agent_id)
        agent.revoke(reason=reason)
    except APIError as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)
    print(f"Revoked {agent_id}")


# === sessions commands ===


sessions_app = typer.Typer(
    name="sessions",
    help="List and end sessions.",
    no_args_is_help=True,
)
app.add_typer(sessions_app, name="sessions")


@sessions_app.command("list")
def sessions_list(
    limit: int = typer.Option(50, "--limit", help="Max sessions to return."),
    status: str = typer.Option("", "--status", help="Filter by session status."),
    agent_id: str = typer.Option("", "--agent", help="Filter by agent_id."),
) -> None:
    """List sessions for the current org."""
    _init_sdk()
    from asqav import APIError, list_sessions

    try:
        kwargs: dict = {"limit": limit}
        if status:
            kwargs["status"] = status
        if agent_id:
            kwargs["agent_id"] = agent_id
        rows = list_sessions(**kwargs)
    except APIError as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)

    if not rows:
        print("No sessions found.")
        return
    for s in rows:
        print(
            f"  {s.session_id}  agent={s.agent_id}  status={s.status}  "
            f"started={s.started_at}"
        )


@sessions_app.command("end")
def sessions_end(
    session_id: str = typer.Argument(help="Session to end."),
    status: str = typer.Option("completed", "--status", help="Final status."),
) -> None:
    """End a session by setting its status."""
    _init_sdk()
    from asqav.client import _patch

    try:
        _patch(f"/sessions/{session_id}", {"status": status})
    except Exception as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)
    print(f"Session {session_id} -> {status}")


# === policies commands ===


policies_app = typer.Typer(
    name="policies",
    help="Manage organization policies.",
    no_args_is_help=True,
)
app.add_typer(policies_app, name="policies")


@policies_app.command("list")
def policies_list() -> None:
    """List all org policies."""
    _init_sdk()
    _require_tier("pro")
    from asqav.client import _get

    try:
        data = _get("/policies")
    except Exception as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)
    rows = data if isinstance(data, list) else []
    if not rows:
        print("No policies defined.")
        return
    for p in rows:
        active = "active" if p.get("is_active") else "inactive"
        print(
            f"  {p.get('id', '?')}  {p.get('name', '?')}  "
            f"pattern={p.get('action_pattern', '*')}  "
            f"action={p.get('action', '?')}  [{active}]"
        )


@policies_app.command("create")
def policies_create(
    name: str = typer.Option(..., "--name", help="Policy name."),
    pattern: str = typer.Option("*", "--pattern", help="Action pattern (e.g. data:*)."),
    action: str = typer.Option(
        "block", "--action", help="One of: log, block, block_and_alert."
    ),
) -> None:
    """Create a new policy."""
    _init_sdk()
    _require_tier("pro")
    from asqav.client import _post

    try:
        out = _post(
            "/policies",
            {"name": name, "action_pattern": pattern, "action": action, "is_active": True},
        )
    except Exception as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)
    print(f"Created policy {out.get('id', '?')}: {name}")


@policies_app.command("delete")
def policies_delete(policy_id: str = typer.Argument(help="Policy ID to delete.")) -> None:
    """Delete a policy."""
    _init_sdk()
    _require_tier("pro")
    from asqav.client import _delete

    try:
        _delete(f"/policies/{policy_id}")
    except Exception as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)
    print(f"Deleted policy {policy_id}")


# === webhooks commands ===


webhooks_app = typer.Typer(
    name="webhooks",
    help="Manage outbound webhooks.",
    no_args_is_help=True,
)
app.add_typer(webhooks_app, name="webhooks")


@webhooks_app.command("list")
def webhooks_list() -> None:
    """List configured webhooks."""
    _init_sdk()
    _require_tier("pro")
    from asqav.client import _get

    try:
        data = _get("/webhooks")
    except Exception as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)
    rows = data if isinstance(data, list) else []
    if not rows:
        print("No webhooks configured.")
        return
    for w in rows:
        events = ",".join(w.get("events", []) or [])
        print(f"  {w.get('id', '?')}  {w.get('url', '?')}  events=[{events}]")


@webhooks_app.command("create")
def webhooks_create(
    url: str = typer.Option(..., "--url", help="Webhook target URL (HTTPS)."),
    events: str = typer.Option(
        "signature.created", "--events", help="Comma-separated event names."
    ),
) -> None:
    """Create a webhook subscription."""
    _init_sdk()
    _require_tier("pro")
    from asqav.client import _post

    event_list = [e.strip() for e in events.split(",") if e.strip()]
    try:
        out = _post("/webhooks", {"url": url, "events": event_list})
    except Exception as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)
    print(f"Created webhook {out.get('id', '?')}: {url}")


@webhooks_app.command("delete")
def webhooks_delete(webhook_id: str = typer.Argument(help="Webhook ID.")) -> None:
    """Delete a webhook."""
    _init_sdk()
    _require_tier("pro")
    from asqav.client import _delete

    try:
        _delete(f"/webhooks/{webhook_id}")
    except Exception as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)
    print(f"Deleted webhook {webhook_id}")


# === sync command ===


def _init_sdk() -> None:
    """Initialize SDK with API key from environment."""
    import os

    import asqav

    api_key = os.environ.get("ASQAV_API_KEY")
    if not api_key:
        print("Error: ASQAV_API_KEY environment variable required.")
        raise typer.Exit(code=1)
    asqav.init(api_key=api_key)


_TIER_RANK = {"free": 0, "pro": 1, "business": 2, "enterprise": 3}


def _fetch_account_tier() -> str | None:
    """Best-effort lookup of the calling org's subscription tier.

    Returns the lowercase tier string, or None when the server does not
    expose the endpoint (older deployments) or the call fails. Callers
    treat None as "skip the gate" - the server still enforces.
    """
    try:
        from asqav.client import _get
    except Exception:
        return None
    try:
        data = _get("/account")
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    tier = data.get("tier")
    if not isinstance(tier, str) or not tier:
        return None
    return tier.lower()


def _require_tier(min_tier: str) -> None:
    """Block the command when the calling org is below ``min_tier``.

    Skips the gate when the /account endpoint is unreachable so older
    self-hosted deployments stay functional. The server is still the
    source of truth - this only gives a clean client-side message.
    """
    tier = _fetch_account_tier()
    if tier is None:
        return
    have = _TIER_RANK.get(tier, 0)
    need = _TIER_RANK.get(min_tier.lower(), 0)
    if have >= need:
        return
    print(
        f"Error: this command requires the {min_tier.title()} tier "
        f"(current: {tier}). Upgrade at https://asqav.com/pricing.",
        file=sys.stderr,
    )
    raise typer.Exit(code=2)


@app.command()
def sync() -> None:
    """Sync local queue to the Asqav API."""
    _init_sdk()

    from asqav.local import LocalQueue

    queue = LocalQueue()
    pending = queue.count()

    if pending == 0:
        print("Queue is empty, nothing to sync.")
        return

    print(f"Syncing {pending} pending items...")

    def on_progress(item_id: str, success: bool, error: str | None) -> None:
        if success:
            print(f"  [ok] {item_id}")
        else:
            print(f"  [fail] {item_id}: {error}")

    result = queue.sync(on_progress=on_progress)

    synced = result["synced"]
    total = result["total"]
    print(f"\nSynced {synced}/{total} items.")

    if result["errors"]:
        print(f"\n{result['failed']} items failed and remain in queue.")
        for err in result["errors"]:
            print(f"  {err['id']}: {err['error']}")


# === queue subcommands ===


@queue_app.command("list")
def queue_list() -> None:
    """List pending items in the local queue."""
    from asqav.local import LocalQueue

    queue = LocalQueue()
    items = queue.list_pending()

    if not items:
        print("Queue is empty.")
        return

    for item in items:
        item_id = item.get("id", "unknown")
        action = item.get("action_type", "unknown")
        agent = item.get("agent_id", "unknown")
        queued = item.get("queued_at", "unknown")
        print(f"  {item_id}  {action}  agent={agent}  queued={queued}")


@queue_app.command("count")
def queue_count() -> None:
    """Show number of pending items in the local queue."""
    from asqav.local import LocalQueue

    queue = LocalQueue()
    n = queue.count()
    print(f"{n} pending items.")


# === quickstart command ===


@app.command()
def demo(
    port: int = typer.Option(3030, help="Port for the local dashboard."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not auto-open a browser."),
) -> None:
    """Run the local governance demo dashboard.

    Opens http://localhost:3030 with 4 pre-loaded scenarios (rm -rf,
    fintech wire transfer, k8s scale-to-zero, clinical lab order). No
    signup, no Docker, no API key required.
    """
    from asqav.demo import serve

    serve(port=port, open_browser=not no_browser)


@app.command()
def quickstart() -> None:
    """Get started with asqav in 60 seconds."""
    import os

    api_key = os.environ.get("ASQAV_API_KEY")
    if api_key:
        typer.echo(
            typer.style("API key detected.", fg=typer.colors.GREEN, bold=True)
        )
    else:
        typer.echo(
            typer.style(
                "ASQAV_API_KEY not set. Get one at https://cloud.asqav.com",
                fg=typer.colors.YELLOW,
                bold=True,
            )
        )

    typer.echo("\n--- MCP config (.mcp.json) ---\n")
    typer.echo(
        '{\n'
        '  "mcpServers": {\n'
        '    "asqav": {\n'
        '      "command": "uvx",\n'
        '      "args": ["asqav-mcp"],\n'
        '      "env": {\n'
        '        "ASQAV_API_KEY": "<your-key>"\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '}'
    )

    typer.echo("\n--- Default policy example ---\n")
    typer.echo(
        "from asqav import Policy\n"
        "\n"
        'policy = Policy(\n'
        '    name="default",\n'
        '    rules=[\n'
        '        {"action": "api:call", "effect": "allow"},\n'
        '        {"action": "write:data", "effect": "deny"},\n'
        '    ],\n'
        ')'
    )

    typer.echo("\n--- Next steps ---\n")
    typer.echo("1. Create an agent:  asqav agents create my-agent")
    typer.echo("2. Sign actions:     agent.sign('api:call', metadata={...})")
    typer.echo("3. Verify anywhere:  asqav verify <signature_id>")
    typer.echo("4. Run diagnostics:  asqav doctor")


# === doctor command ===


@app.command()
def doctor() -> None:
    """Validate your governance setup."""
    import os

    all_ok = True

    # Check 1: API key
    api_key = os.environ.get("ASQAV_API_KEY")
    if api_key:
        typer.echo(
            typer.style("PASS", fg=typer.colors.GREEN, bold=True) + "  API key set"
        )
    else:
        typer.echo(
            typer.style("FAIL", fg=typer.colors.RED, bold=True)
            + "  ASQAV_API_KEY not set"
        )
        all_ok = False

    # Check 2: API reachable
    if api_key:
        import asqav

        asqav.init(api_key=api_key)
        try:
            from asqav.client import health_check

            health_check()
            typer.echo(
                typer.style("PASS", fg=typer.colors.GREEN, bold=True)
                + "  API reachable"
            )
        except Exception as exc:
            typer.echo(
                typer.style("FAIL", fg=typer.colors.RED, bold=True)
                + f"  API unreachable: {exc}"
            )
            all_ok = False
    else:
        typer.echo(
            typer.style("SKIP", fg=typer.colors.YELLOW, bold=True)
            + "  API reachable (no key)"
        )

    # Check 3: SDK version
    typer.echo(
        typer.style("INFO", fg=typer.colors.BLUE, bold=True)
        + f"  SDK version: {__version__}"
    )

    if all_ok:
        typer.echo(
            "\n" + typer.style("All checks passed.", fg=typer.colors.GREEN)
        )
    else:
        typer.echo(
            "\n"
            + typer.style("Some checks failed. See above.", fg=typer.colors.RED)
        )
        raise typer.Exit(code=1)


@queue_app.command("clear")
def queue_clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Clear all pending items from the local queue."""
    from asqav.local import LocalQueue

    queue = LocalQueue()
    n = queue.count()

    if n == 0:
        print("Queue is already empty.")
        return

    if not yes:
        confirm = typer.confirm(f"Delete {n} pending items?")
        if not confirm:
            print("Cancelled.")
            raise typer.Exit()

    deleted = queue.clear()
    print(f"Cleared {deleted} items.")


# === preflight command (wraps Agent.preflight) ===


@app.command()
def preflight(
    agent_id: str = typer.Argument(help="Agent ID to check."),
    action_type: str = typer.Argument(help="Action to evaluate (e.g. data:read)."),
    json_out: bool = typer.Option(False, "--json", help="Print result as JSON."),
) -> None:
    """Run a pre-flight check (revocation + suspension + policy) for an action.

    Wraps Agent.get(agent_id).preflight(action_type). Exits non-zero when the
    action is blocked, so it works as a gate in shell pipelines.
    """
    import json as json_mod

    _init_sdk()
    _require_tier("pro")
    from asqav import Agent, APIError

    try:
        agent = Agent.get(agent_id)
        result = agent.preflight(action_type)
    except APIError as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)

    if json_out:
        print(json_mod.dumps({
            "cleared": result.cleared,
            "agent_active": result.agent_active,
            "policy_allowed": result.policy_allowed,
            "reasons": result.reasons,
            "explanation": result.explanation,
        }, indent=2))
    else:
        verdict = (
            typer.style("CLEARED", fg=typer.colors.GREEN, bold=True)
            if result.cleared
            else typer.style("BLOCKED", fg=typer.colors.RED, bold=True)
        )
        typer.echo(f"{verdict}  {result.explanation or action_type}")
        for reason in result.reasons:
            typer.echo(f"  - {reason}")

    if not result.cleared:
        raise typer.Exit(code=1)


# === budget commands (wraps BudgetTracker) ===


@budget_app.command("check")
def budget_check(
    agent_id: str = typer.Option(..., "--agent-id", help="Agent ID."),
    limit: float = typer.Option(..., "--limit", help="Budget ceiling."),
    estimated_cost: float = typer.Option(
        ..., "--estimated-cost", help="Cost of the pending action."
    ),
    current_spend: float = typer.Option(0.0, "--current-spend", help="Cumulative spend so far."),
    currency: str = typer.Option("USD", "--currency", help="Currency code."),
    json_out: bool = typer.Option(False, "--json", help="Print result as JSON."),
) -> None:
    """Pure-math check: would estimated_cost push current_spend past limit?

    Stateless. The CLI runs the same arithmetic as BudgetTracker.check() with
    the spend you supply. For server-side enforcement use 'asqav budget record'.
    """
    import json as json_mod

    _init_sdk()
    _require_tier("pro")
    from asqav import Agent, BudgetTracker

    agent = Agent.get(agent_id)
    tracker = BudgetTracker(agent, limit=limit, currency=currency)
    tracker._spend = float(current_spend)
    result = tracker.check(estimated_cost)

    if json_out:
        print(json_mod.dumps({
            "allowed": result.allowed,
            "current_spend": result.current_spend,
            "limit": result.limit,
            "remaining": result.remaining,
            "reason": result.reason,
        }, indent=2))
    else:
        verdict = (
            typer.style("ALLOWED", fg=typer.colors.GREEN, bold=True)
            if result.allowed
            else typer.style("DENIED", fg=typer.colors.RED, bold=True)
        )
        typer.echo(f"{verdict}  remaining={result.remaining:.4f} {currency}")
        if result.reason:
            typer.echo(f"  reason: {result.reason}")

    if not result.allowed:
        raise typer.Exit(code=1)


@budget_app.command("record")
def budget_record(
    agent_id: str = typer.Option(..., "--agent-id", help="Agent ID."),
    action_type: str = typer.Option(..., "--action", help="Action being recorded."),
    actual_cost: float = typer.Option(..., "--actual-cost", help="Realized cost."),
    limit: float = typer.Option(..., "--limit", help="Budget ceiling for the signed payload."),
    current_spend: float = typer.Option(
        0.0, "--current-spend", help="Cumulative spend prior to this record."
    ),
    currency: str = typer.Option("USD", "--currency", help="Currency code."),
) -> None:
    """Sign a spend record on the server (tamper-evident).

    Wraps BudgetTracker.record(). The signed payload includes cost, currency,
    cumulative spend after the record, and the configured limit.
    """
    _init_sdk()
    _require_tier("pro")
    from asqav import Agent, APIError, BudgetTracker

    try:
        agent = Agent.get(agent_id)
        tracker = BudgetTracker(agent, limit=limit, currency=currency)
        tracker._spend = float(current_spend)
        sig = tracker.record(action_type, actual_cost=actual_cost)
    except APIError as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)

    typer.echo(f"signature_id: {sig.signature_id}")
    typer.echo(f"action_id:    {sig.action_id}")
    typer.echo(f"verify_url:   {sig.verification_url}")


# === approve command (wraps approve_action) ===


@app.command()
def approve(
    session_id: str = typer.Argument(help="Signing session to approve."),
    entity_id: str = typer.Argument(help="Entity providing approval."),
    json_out: bool = typer.Option(False, "--json", help="Print result as JSON."),
) -> None:
    """Approve a pending signing-action session. Wraps asqav.approve_action()."""
    import json as json_mod

    _init_sdk()
    _require_tier("pro")
    from asqav import APIError, approve_action

    try:
        result = approve_action(session_id=session_id, entity_id=entity_id)
    except APIError as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)

    if json_out:
        print(json_mod.dumps({
            "session_id": result.session_id,
            "entity_id": result.entity_id,
            "signatures_collected": result.signatures_collected,
            "approvals_required": result.approvals_required,
            "status": result.status,
            "approved": result.approved,
        }, indent=2))
    else:
        progress = f"{result.signatures_collected}/{result.approvals_required}"
        verdict = (
            typer.style("APPROVED", fg=typer.colors.GREEN, bold=True)
            if result.approved
            else typer.style("PENDING", fg=typer.colors.YELLOW, bold=True)
        )
        typer.echo(f"{verdict}  {progress} signatures, status={result.status}")


# === compliance commands (wraps export_bundle) ===


audit_pack_app = typer.Typer(
    name="audit-pack",
    help="Audit Pack export and policy_digest resolution (IETF profile).",
    no_args_is_help=True,
)
app.add_typer(audit_pack_app, name="audit-pack")


@audit_pack_app.command("export")
def audit_pack_export(
    start: str = typer.Option(..., "--start", help="Window start, ISO 8601 (UTC)."),
    end: str = typer.Option(..., "--end", help="Window end, ISO 8601 (UTC). Half-open."),
    organization_id: str = typer.Option(
        "",
        "--organization-id",
        help="Defaults to the caller's org. Cross-org export requires admin scopes.",
    ),
    only_compliance: bool = typer.Option(
        True,
        "--only-compliance/--no-only-compliance",
        help="When true (default), only IETF Compliance Receipts are exported.",
    ),
    output_file: str = typer.Option(
        ..., "--output-file", help="Path to write the signed bundle JSON."
    ),
) -> None:
    """Export a signed Audit Pack bundle for receipts in [start, end).

    Calls POST `/audit-pack/export`. The bundle is signed with the org's
    most-recent active agent key over the JCS-canonical bundle bytes, so
    a verifier can rederive the digest and check the signature offline.
    """
    import json as json_mod

    _init_sdk()
    from asqav.client import _post

    body: dict = {
        "start": start,
        "end": end,
        "only_compliance": only_compliance,
    }
    if organization_id:
        body["organization_id"] = organization_id

    try:
        bundle = _post("/audit-pack/export", body)
    except Exception as exc:
        print(f"Error exporting audit pack: {exc}")
        raise typer.Exit(code=1)

    try:
        with open(output_file, "w") as f:
            json_mod.dump(bundle, f, indent=2, default=str)
    except OSError as exc:
        print(f"Error writing bundle: {exc}")
        raise typer.Exit(code=1)

    count = bundle.get("receipt_count", "?")
    digest = bundle.get("bundle_digest", "?")
    print(f"wrote {count} receipts to {output_file}")
    print(f"bundle_digest: {digest}")
    print(f"algorithm:     {bundle.get('bundle_signature_algorithm', '?')}")


@audit_pack_app.command("policy")
def audit_pack_policy(
    digest: str = typer.Argument(
        help="`sha256:<hex>` digest, or 64-hex (the prefix is added)."
    ),
    output: str = typer.Option(
        "json", "--output", "-o", help="Output format: json or text."
    ),
) -> None:
    """Resolve a `policy_digest` to the retained policy artefact JSON.

    Calls GET `/audit-pack/policy/{digest}`. The receipt's `policy_digest`
    is `sha256:<hex>` over the canonical artefact JSON; this command
    fetches the artefact so a verifier can rederive the digest offline.
    """
    import json as json_mod

    _init_sdk()
    from asqav.client import _get

    if not digest.startswith("sha256:"):
        if len(digest) == 64 and all(c in "0123456789abcdefABCDEF" for c in digest):
            digest = "sha256:" + digest.lower()
        else:
            print(
                "Error: digest must be `sha256:<64-hex>` or a bare 64-hex string."
            )
            raise typer.Exit(code=1)

    try:
        data = _get(f"/audit-pack/policy/{digest}")
    except Exception as exc:
        print(f"Error fetching artefact: {exc}")
        raise typer.Exit(code=1)

    if output == "text":
        print(f"digest:          {data.get('digest', '?')}")
        print(f"organization_id: {data.get('organization_id', '?')}")
        print(f"agent_id:        {data.get('agent_id', '?')}")
        print(f"action_type:     {data.get('action_type', '?')}")
        print(f"created_at:      {data.get('created_at', '?')}")
        print("--- artefact ---")
        print(json_mod.dumps(data.get("artefact", {}), indent=2))
    else:
        print(json_mod.dumps(data, indent=2, default=str))


# === payloads erase command (P4: GDPR right-to-erasure) ===


payloads_app = typer.Typer(
    name="payloads",
    help="Erase persisted payloads (P4: GDPR / UK-GDPR right-to-erasure).",
    no_args_is_help=True,
)
app.add_typer(payloads_app, name="payloads")


@payloads_app.command("erase")
def payloads_erase(
    signature_id: str = typer.Argument(help="Signature whose payload bytes are erased."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Erase the persisted JSON payload for a signature record.

    Calls DELETE `/signatures/payloads/{signature_id}`. The cryptographic
    `message` bytes and chain hash stay untouched so the receipt remains
    verifiable; the human-readable JSON payload is replaced with a
    tombstone marker. Idempotent: replaying on an already-erased record
    returns 204.
    """
    if not yes:
        ok = typer.confirm(
            f"Erase payload for {signature_id}? Verification stays intact, "
            f"but the JSON payload becomes a tombstone."
        )
        if not ok:
            print("Cancelled.")
            raise typer.Exit()

    _init_sdk()
    from asqav.client import _delete

    try:
        _delete(f"/signatures/payloads/{signature_id}")
    except Exception as exc:
        print(f"Error erasing payload: {exc}")
        raise typer.Exit(code=1)

    print(f"Payload erased for {signature_id}.")
    print("Tombstone written; signature remains cryptographically verifiable.")


# === org settings (compliance_mode_strict) ===


org_app = typer.Typer(
    name="org",
    help="Organization-level settings.",
    no_args_is_help=True,
)
app.add_typer(org_app, name="org")


@org_app.command("set-compliance-strict")
def org_set_compliance_strict(
    org_id: str = typer.Argument(help="Organization ID to update."),
    enable: bool = typer.Option(False, "--enable", help="Turn strict mode on."),
    disable: bool = typer.Option(False, "--disable", help="Turn strict mode off."),
) -> None:
    """Toggle per-org `compliance_mode_strict`.

    When True, the cloud rejects any sign request that is not flagged
    `compliance_mode=true` with HTTP 412 so non-instrumented call paths
    cannot slip past the receipts trail. Calls PATCH
    `/orgs/{org_id}` with `{"compliance_mode_strict": <bool>}`.
    """
    if enable == disable:
        print("Error: pass exactly one of --enable or --disable.")
        raise typer.Exit(code=1)

    _init_sdk()
    from asqav.client import _patch

    body = {"compliance_mode_strict": bool(enable)}
    try:
        out = _patch(f"/orgs/{org_id}", body)
    except Exception as exc:
        print(f"Error updating organization: {exc}")
        raise typer.Exit(code=1)

    state = "enabled" if enable else "disabled"
    print(f"compliance_mode_strict {state} for org {org_id}.")
    if isinstance(out, dict) and "compliance_mode_strict" in out:
        print(f"server confirms: compliance_mode_strict={out['compliance_mode_strict']}")


# === keys generate (local keypair) ===


keys_app = typer.Typer(
    name="keys",
    help="Local keypair management (algorithm agility).",
    no_args_is_help=True,
)
app.add_typer(keys_app, name="keys")


@keys_app.command("generate")
def keys_generate(
    algorithm: str = typer.Option(
        "ed25519",
        "--algorithm",
        help="ed25519|es256. ml-dsa-65 keygen is server-side only.",
    ),
    out: str = typer.Option(
        "",
        "--out",
        help="Path to write the private key bytes. Public key always prints to stdout.",
    ),
) -> None:
    """Generate a local keypair (offline / air-gapped flows).

    Ed25519 and ES256 emit PKCS#8 PEM. ML-DSA-65 is rejected with a
    `unsupported_algorithm` error; call `asqav agents create` to mint
    one via the cloud KMS.
    """
    from asqav import generate_local_keypair

    try:
        kp = generate_local_keypair(algorithm)  # type: ignore[arg-type]
    except (ValueError, ImportError) as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)

    if out:
        try:
            with open(out, "wb") as f:
                f.write(kp.private_key_pem)
        except OSError as exc:
            print(f"Error writing key file: {exc}")
            raise typer.Exit(code=1)
        try:
            os_module = __import__("os")
            os_module.chmod(out, 0o600)
        except Exception:
            pass
        print(f"Private key written to {out} (mode 0600).")

    print("--- public key ---")
    sys.stdout.write(kp.public_key_pem.decode("utf-8", errors="replace"))


# === replay verify (IETF chain verification) ===


# Mounted as a top-level command (not a subapp) so the existing
# `asqav replay` (session replay) keeps its current shape.
@app.command("replay-verify")
def replay_verify_cmd(
    agent_id: str = typer.Argument(help="Agent that owns the chain."),
    session_id: str = typer.Argument(help="Session whose chain to re-derive."),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Require every step to carry a cloud `signed_envelope`. Synthetic-shape fallback steps cannot prove byte-level integrity.",
    ),
    output: str = typer.Option("text", "--output", "-o", help="Output format: text or json."),
) -> None:
    """Re-derive a session's IETF chain (`sha256(canonical_json(signed_envelope))`).

    Wraps :func:`asqav.replay`. With ``--strict`` the command fails
    when any step lacks the cloud-emitted ``signed_envelope``.
    """
    import json as json_mod

    _init_sdk()
    _require_tier("pro")
    from asqav import APIError
    from asqav import replay as replay_api

    try:
        timeline = replay_api(agent_id, session_id)
    except APIError as exc:
        print(f"Error fetching timeline: {exc}")
        raise typer.Exit(code=1)

    timeline.verify_chain()
    chain_valid = bool(timeline.compliance_chain_valid)
    synthetic_steps = [s for s in timeline.steps if s.signed_envelope is None]
    has_synthetic = bool(synthetic_steps)
    strict_pass = chain_valid and not (strict and has_synthetic)

    if output == "json":
        print(json_mod.dumps({
            "compliance_chain_valid": chain_valid,
            "strict": strict,
            "strict_pass": strict_pass,
            "step_count": len(timeline.steps),
            "synthetic_step_count": len(synthetic_steps),
            "steps": [
                {
                    "index": s.index,
                    "signature_id": getattr(s, "signature_id", None),
                    "chain_valid": getattr(s, "chain_valid", None),
                    "has_envelope": s.signed_envelope is not None,
                }
                for s in timeline.steps
            ],
        }, indent=2, default=str))
        if not strict_pass:
            raise typer.Exit(code=1)
        return

    print(f"compliance_chain_valid: {chain_valid}")
    print(f"strict: {strict}")
    print(f"steps: {len(timeline.steps)} (synthetic: {len(synthetic_steps)})")
    for s in timeline.steps:
        ok = getattr(s, "chain_valid", False)
        shape = "envelope" if s.signed_envelope is not None else "synthetic"
        marker = "ok  " if ok else "FAIL"
        print(f"  [{marker}] step {s.index} ({shape}) sig={getattr(s, 'signature_id', '?')}")
    if strict and has_synthetic:
        print("strict mode FAILED: at least one step lacks the cloud signed_envelope.")
    if not strict_pass:
        raise typer.Exit(code=1)


# === migrate run (operator commands) ===


migrate_app = typer.Typer(
    name="migrate",
    help="Operator-only DB migration runner (X-Maintenance-Key required).",
    no_args_is_help=True,
)
app.add_typer(migrate_app, name="migrate")


_SUPPORTED_MIGRATIONS = {"v3-20", "v3-21", "v3-22"}


@migrate_app.command("run")
def migrate_run(
    migration: str = typer.Argument(help="One of v3-20, v3-21, v3-22."),
) -> None:
    """Trigger a maintenance migration.

    Reads the maintenance key from the `ASQAV_MAINTENANCE_KEY` env var and
    POSTs `/maintenance/run-migration-{migration}`. Refuses to run when
    the env var is missing.
    """
    import json as json_mod
    import os

    if migration not in _SUPPORTED_MIGRATIONS:
        print(
            f"Error: unsupported migration {migration!r}. "
            f"Supported: {sorted(_SUPPORTED_MIGRATIONS)}."
        )
        raise typer.Exit(code=1)

    maintenance_key = os.environ.get("ASQAV_MAINTENANCE_KEY")
    if not maintenance_key:
        print("Error: ASQAV_MAINTENANCE_KEY env var is required for migration commands.")
        raise typer.Exit(code=1)

    # `_post` cannot inject headers, so build a one-off urllib request with X-Maintenance-Key.
    from asqav import init as _init

    api_key = os.environ.get("ASQAV_API_KEY")
    if not api_key:
        print("Error: ASQAV_API_KEY env var is required.")
        raise typer.Exit(code=1)
    _init(api_key=api_key)

    base = os.environ.get("ASQAV_API_URL", "https://api.asqav.com/api/v1")
    url = f"{base}/maintenance/run-migration-{migration}"

    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url,
        method="POST",
        headers={
            "X-API-Key": api_key,
            "X-Maintenance-Key": maintenance_key,
            "Content-Type": "application/json",
        },
        data=b"{}",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        print(f"HTTP {exc.code} from /maintenance/run-migration-{migration}: {detail}")
        raise typer.Exit(code=1)
    except urllib.error.URLError as exc:
        print(f"Network error: {exc}")
        raise typer.Exit(code=1)

    try:
        parsed = json_mod.loads(body)
        print(json_mod.dumps(parsed, indent=2))
    except json_mod.JSONDecodeError:
        print(body)


@compliance_app.command("frameworks")
def compliance_frameworks() -> None:
    """List the compliance frameworks the bundle exporter recognizes."""
    from asqav.compliance import FRAMEWORKS

    for key, meta in sorted(FRAMEWORKS.items()):
        typer.echo(f"{key}\t{meta.get('name', key)}")


@compliance_app.command("export")
def compliance_export(
    session: str = typer.Option(..., "--session", help="Session ID to bundle signatures from."),
    framework: str = typer.Option(
        "eu_ai_act", "--framework", help="Compliance framework key."
    ),
    output: str = typer.Option(..., "--output", "-o", help="File to write the bundle JSON to."),
) -> None:
    """Pull a session's signatures and export a Merkle-rooted compliance bundle.

    Wraps asqav.client.get_session_signatures() + asqav.compliance.export_bundle().
    Run 'asqav compliance frameworks' to see valid framework keys.
    """
    _init_sdk()
    _require_tier("business")
    from asqav import APIError
    from asqav.client import get_session_signatures
    from asqav.compliance import export_bundle

    try:
        sigs = get_session_signatures(session)
    except APIError as exc:
        print(f"Error fetching signatures: {exc}")
        raise typer.Exit(code=1)

    try:
        bundle = export_bundle(sigs, framework=framework)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)

    bundle.to_file(output)
    typer.echo(f"wrote {bundle.receipt_count} receipts to {output}")
    typer.echo(f"merkle_root: {bundle.merkle_root}")
