"""
asqav CLI - Command-line interface for asqav Cloud.

Requires the 'cli' extra: pip install asqav[cli]

Commands:
    asqav verify <signature_id>   - Verify a signature (public, no auth)
    asqav agents list             - List your agents
    asqav agents create <name>    - Create a new agent
    asqav sync                    - Sync local queue to API
    asqav queue list/count/clear  - Manage local queue
    asqav --version               - Show version
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

    agents = data.get("agents", [])
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
