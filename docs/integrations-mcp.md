# MCP integration (default-on governance)

asqav can sign a governance receipt on every tool call of an
[MCP](https://modelcontextprotocol.io) server. One function call turns it on
for the whole server, so governance is default-on rather than a per-call
flag. Signing is fail-open: a signing outage never blocks or changes a tool
call.

## Python

Install the extra:

```bash
pip install asqav[mcp]
```

Wire it in one call:

```python
import asqav
from asqav.extras.mcp import enable_mcp_governance
from mcp.server.fastmcp import FastMCP

asqav.init("sk_...")

server = FastMCP("my-server")
enable_mcp_governance(server, agent_name="my-mcp-server")

@server.tool()
def add(a: int, b: int) -> int:
    return a + b
```

Every call to `add` (or any other tool) now signs a `tool:call` receipt. A
tool that raises also signs `tool:error` and the error propagates unchanged.

`enable_mcp_governance` handles both server styles:

- `FastMCP` - it wraps the tool-dispatch funnel that both the direct API and
  real request routing pass through.
- The low-level `mcp.server.lowlevel.Server` - it wraps the `call_tool`
  decorator so handlers you register sign on each call.

Options mirror the other adapters: `api_key`, `agent_name`, `agent_id`, and
`observe` (log intent instead of signing).

## TypeScript

Install the peer:

```bash
npm install @asqav/sdk @modelcontextprotocol/sdk
```

```ts
import { init } from "@asqav/sdk";
import { enableMcpGovernance } from "@asqav/sdk/extras/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

init({ apiKey: process.env.ASQAV_API_KEY! });

const server = new McpServer({ name: "my-server", version: "1.0.0" });
enableMcpGovernance(server, { agentName: "my-mcp-server" });

server.registerTool("add", { description: "add" }, async ({ a, b }) => ({
  content: [{ type: "text", text: String(a + b) }],
}));
```

`enableMcpGovernance` wraps `registerTool` and the deprecated `tool`
registrar, so every tool registered after the call signs a `tool:call`
receipt when invoked.

## Notes

- Additive and non-breaking: importing the module without calling the enable
  function changes nothing.
- The MCP SDK is an optional extra (Python) / optional peer (TypeScript). The
  adapter never imports it directly; it operates on the server instance you
  pass in.
- Fail-open: governance signing never raises into the MCP request path.

Runnable examples: `python/examples/mcp_example.py`,
`typescript/examples/mcp.ts`.
