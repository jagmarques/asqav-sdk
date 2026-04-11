"""smolagents integration example for asqav.

Demonstrates how to wrap smolagents tools with asqav governance so every
tool invocation is signed and recorded in the audit trail.

Requirements:
    pip install asqav[smolagents]

Usage:
    ASQAV_API_KEY=sk_... python examples/smolagents_example.py
"""

from __future__ import annotations

import os

import asqav
from asqav.extras.smolagents import AsqavSmolagentsHook

try:
    from smolagents import CodeAgent, HfApiModel, tool
except ImportError:
    raise SystemExit(
        "smolagents is required for this example.\n"
        "Install with: pip install asqav[smolagents]"
    )

# ---------------------------------------------------------------------------
# Initialise asqav
# ---------------------------------------------------------------------------

api_key = os.environ.get("ASQAV_API_KEY", "")
if not api_key:
    raise SystemExit(
        "Set the ASQAV_API_KEY environment variable before running this example.\n"
        "Get your key at https://asqav.com"
    )

asqav.init(api_key=api_key)

# Create a governance hook - one hook can wrap multiple tools.
hook = AsqavSmolagentsHook(agent_name="smolagents-demo")

# ---------------------------------------------------------------------------
# Define tools and wrap them with asqav signing
# ---------------------------------------------------------------------------


@tool
def search_web(query: str) -> str:
    """Search the web for up-to-date information.

    Args:
        query: The search query string.
    """
    # Replace with a real search implementation.
    return f"Search results for: {query}"


@tool
def read_file(path: str) -> str:
    """Read the contents of a local file.

    Args:
        path: Absolute or relative path to the file.
    """
    try:
        with open(path) as f:
            return f.read()
    except OSError as exc:
        return f"Error reading file: {exc}"


# Wrap each tool - this patches tool.forward to sign every call.
signed_search = hook.wrap_tool(search_web)
signed_read = hook.wrap_tool(read_file)

# ---------------------------------------------------------------------------
# Build and run the agent
# ---------------------------------------------------------------------------

model = HfApiModel()  # Uses the HF Inference API; set HF_TOKEN env var.

agent = CodeAgent(
    tools=[signed_search, signed_read],
    model=model,
)

print("Running smolagents agent with asqav governance...")
result = agent.run("Search for the latest news on AI safety and summarise it.")

print("\nAgent result:")
print(result)
print("\nView the signed audit trail at https://asqav.com/dashboard")
