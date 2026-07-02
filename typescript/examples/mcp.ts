/**
 * Default-on Asqav governance for an MCP (Model Context Protocol) server.
 *
 * Install:
 *   npm install @asqav/sdk @modelcontextprotocol/sdk
 *
 * Run:
 *   ASQAV_API_KEY=sk_... npx tsx examples/mcp.ts
 *
 * One call to enableMcpGovernance turns governance on for the whole
 * server. Every tool call then signs an Asqav receipt by default, with no
 * per-call flag. Signing is fail-open, so a signing outage never blocks a
 * tool call.
 */

import { init } from "@asqav/sdk";
import { enableMcpGovernance } from "@asqav/sdk/extras/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

async function main(): Promise<void> {
  init({ apiKey: process.env.ASQAV_API_KEY! });

  const server = new McpServer({ name: "demo-server", version: "1.0.0" });

  // Turn governance on once. Every tool registered below is now governed.
  enableMcpGovernance(server, { agentName: "demo-mcp-server" });

  server.registerTool(
    "add",
    { description: "Add two integers", inputSchema: { a: z.number(), b: z.number() } },
    async ({ a, b }) => ({ content: [{ type: "text", text: String(a + b) }] }),
  );

  server.registerTool(
    "greet",
    { description: "Return a greeting", inputSchema: { name: z.string() } },
    async ({ name }) => ({ content: [{ type: "text", text: `hello ${name}` }] }),
  );

  // eslint-disable-next-line no-console
  console.log("MCP server governed by asqav. Tools: add, greet");
  // Connect a transport (StdioServerTransport) to serve; omitted to stay offline.
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exitCode = 1;
});
