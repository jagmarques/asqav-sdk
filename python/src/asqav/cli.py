"""
asqav CLI - Command-line interface for asqav Cloud.

Every CLI command is a thin wrapper over a public Python API. The CLI is for
humans and shell scripts; the Python API is for programmatic automation. They
have parity by construction so anything you can do at the terminal you can also
do from code.

Requires the 'cli' extra: pip install asqav[cli]

Commands:
    asqav verify <signature_id>            - Verify a signature (public, no auth)
    asqav agents list                      - List your agents
    asqav agents create <name>             - Create a new agent
    asqav replay <agent_id> <session_id>   - Replay a session's audit trail
    asqav sync                             - Sync local queue to API
    asqav queue list/count/clear           - Manage local queue
    asqav --version                        - Show version
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
    help="Quantum-safe control for AI agents.",
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
    """Quantum-safe control for AI agents."""


@app.command()
def verify(signature_id: str = typer.Argument(help="Signature ID to verify.")) -> None:
    """Verify a signature by ID (public, no auth needed)."""
    from asqav import APIError, verify_signature

    try:
        result = verify_signature(signature_id)
    except APIError as exc:
        if exc.status_code == 404:
            print(f"Signature not found: {signature_id}")
            raise typer.Exit(code=1)
        print(f"Error: {exc}")
        raise typer.Exit(code=1)

    status = "valid" if result.verified else "invalid"
    print(f"Signature: {status}")
    print(f"Agent: {result.agent_name} ({result.agent_id})")
    print(f"Action: {result.action_type}")
    print(f"Algorithm: {result.algorithm}")


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


# ---------------------------------------------------------------------------
# sync command
# ---------------------------------------------------------------------------


def _init_sdk() -> None:
    """Initialize SDK with API key from environment."""
    import os

    import asqav

    api_key = os.environ.get("ASQAV_API_KEY")
    if not api_key:
        print("Error: ASQAV_API_KEY environment variable required.")
        raise typer.Exit(code=1)
    asqav.init(api_key=api_key)


@app.command()
def sync() -> None:
    """Sync local queue to the asqav API."""
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


# ---------------------------------------------------------------------------
# queue subcommands
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# quickstart command
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# doctor command
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# preflight command (wraps Agent.preflight)
# ---------------------------------------------------------------------------


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
    from asqav import APIError, Agent

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


# ---------------------------------------------------------------------------
# budget commands (wraps BudgetTracker)
# ---------------------------------------------------------------------------


@budget_app.command("check")
def budget_check(
    agent_id: str = typer.Option(..., "--agent-id", help="Agent ID."),
    limit: float = typer.Option(..., "--limit", help="Budget ceiling."),
    estimated_cost: float = typer.Option(..., "--estimated-cost", help="Cost of the pending action."),
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
    current_spend: float = typer.Option(0.0, "--current-spend", help="Cumulative spend before this record."),
    currency: str = typer.Option("USD", "--currency", help="Currency code."),
) -> None:
    """Sign a spend record on the server (tamper-evident).

    Wraps BudgetTracker.record(). The signed payload includes cost, currency,
    cumulative spend after the record, and the configured limit.
    """
    _init_sdk()
    from asqav import APIError, Agent, BudgetTracker

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


# ---------------------------------------------------------------------------
# approve command (wraps approve_action)
# ---------------------------------------------------------------------------


@app.command()
def approve(
    session_id: str = typer.Argument(help="Signing session to approve."),
    entity_id: str = typer.Argument(help="Entity providing approval."),
    json_out: bool = typer.Option(False, "--json", help="Print result as JSON."),
) -> None:
    """Approve a pending signing-action session. Wraps asqav.approve_action()."""
    import json as json_mod

    _init_sdk()
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


# ---------------------------------------------------------------------------
# compliance commands (wraps export_bundle)
# ---------------------------------------------------------------------------


@compliance_app.command("frameworks")
def compliance_frameworks() -> None:
    """List the compliance frameworks the bundle exporter recognizes."""
    from asqav.compliance import FRAMEWORKS

    for key, meta in sorted(FRAMEWORKS.items()):
        typer.echo(f"{key}\t{meta.get('name', key)}")


@compliance_app.command("export")
def compliance_export(
    session: str = typer.Option(..., "--session", help="Session ID to bundle signatures from."),
    framework: str = typer.Option("eu_ai_act_art12", "--framework", help="Compliance framework key."),
    output: str = typer.Option(..., "--output", "-o", help="File to write the bundle JSON to."),
) -> None:
    """Pull a session's signatures and export a Merkle-rooted compliance bundle.

    Wraps asqav.client.get_session_signatures() + asqav.compliance.export_bundle().
    Run 'asqav compliance frameworks' to see valid framework keys.
    """
    _init_sdk()
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
