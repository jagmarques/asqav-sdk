"""Default-on asqav governance for an MCP (Model Context Protocol) server.

Install:
    pip install asqav[mcp]

Run:
    export ASQAV_API_KEY=sk_...
    python examples/mcp_example.py

One call to ``enable_mcp_governance`` turns governance on for the whole
server. Every tool call then signs an asqav receipt by default, with no
per-call flag. Signing is fail-open, so a signing outage never blocks a
tool call.
"""

from __future__ import annotations

import os

import asqav
from asqav.extras.mcp import enable_mcp_governance

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as err:
    raise SystemExit(
        "This example needs the MCP SDK. Install with: pip install asqav[mcp]"
    ) from err


def main() -> None:
    asqav.init(os.environ["ASQAV_API_KEY"])

    server = FastMCP("demo-server")

    # Turn governance on once. Every tool below is now governed.
    enable_mcp_governance(server, agent_name="demo-mcp-server")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool()
    def greet(name: str) -> str:
        """Return a greeting."""
        return f"hello {name}"

    print("MCP server governed by asqav. Tools:", ["add", "greet"])
    print("Run over stdio with: server.run() (omitted so the example stays offline)")


if __name__ == "__main__":
    main()
