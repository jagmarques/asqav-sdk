"""
asqav CLI - Command-line interface for asqav Cloud.

Requires the 'cli' extra: pip install asqav[cli]

Commands:
    asqav verify <signature_id>   - Verify a signature (public, no auth)
    asqav agents list             - List your agents
    asqav agents create <name>    - Create a new agent
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
