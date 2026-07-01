/**
 * MCP (Model Context Protocol) integration for Asqav.
 *
 * Install (peer dependency):
 *   npm install @modelcontextprotocol/sdk
 *
 * Usage:
 *   import { init } from "@asqav/sdk";
 *   import { enableMcpGovernance } from "@asqav/sdk/extras/mcp";
 *   import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
 *
 *   init({ apiKey: process.env.ASQAV_API_KEY! });
 *   const server = new McpServer({ name: "my-server", version: "1.0.0" });
 *   enableMcpGovernance(server, { agentName: "my-mcp-server" });
 *   server.registerTool("add", { description: "add" }, async ({ a, b }) => ({
 *     content: [{ type: "text", text: String(a + b) }],
 *   }));
 *
 * One call to ``enableMcpGovernance`` turns governance on for the whole
 * server: every tool registered on it signs an Asqav receipt on each call,
 * by default. Signing is fail-open. This module never imports
 * ``@modelcontextprotocol/sdk`` (it duck-types the passed server), so the
 * peer stays optional and importing without calling changes nothing.
 */

import { AsqavAdapter, type AsqavAdapterOptions } from "./_base.js";

// Boundary type for the duck-typed server. ``any`` rest params let a
// concretely-typed McpServer.registerTool/tool satisfy this shape.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ToolRegistrar = (...args: any[]) => any;

/** Minimal shape of an MCP server: it exposes registerTool and/or tool. */
export interface McpServerLike {
  registerTool?: ToolRegistrar;
  tool?: ToolRegistrar;
}

export type AsqavMcpAdapterOptions = AsqavAdapterOptions;

const WRAP_FLAG = "__asqavWrapped";

/**
 * Signs an Asqav governance receipt on every MCP tool call. ``instrument``
 * wraps the server's tool registrars so each registered callback signs
 * ``tool:call`` before it runs. Fire-and-forget and fail-open, so signing
 * never blocks or alters the tool call.
 */
export class AsqavMcpAdapter extends AsqavAdapter {
  /** Wrap ``server``'s tool registrars in place; returns ``server``. */
  instrument<T extends McpServerLike>(server: T): T {
    const hasRegister = typeof server.registerTool === "function";
    const hasTool = typeof server.tool === "function";
    if (!hasRegister && !hasTool) {
      throw new Error(
        "enableMcpGovernance expects an MCP server exposing registerTool or tool "
        + "(e.g. McpServer from @modelcontextprotocol/sdk).",
      );
    }
    if (hasRegister) this.wrapRegistrar(server, "registerTool");
    if (hasTool) this.wrapRegistrar(server, "tool");
    return server;
  }

  private wrapRegistrar(server: McpServerLike, method: "registerTool" | "tool"): void {
    const original = server[method];
    if (!original) return;
    if ((original as unknown as Record<string, unknown>)[WRAP_FLAG]) return;
    const adapter = this;

    const wrapped = function (this: unknown, ...args: unknown[]): unknown {
      const name = typeof args[0] === "string" ? args[0] : "unknown";
      // The tool callback is the last function argument in every overload.
      let cbIndex = -1;
      for (let i = args.length - 1; i >= 0; i -= 1) {
        if (typeof args[i] === "function") {
          cbIndex = i;
          break;
        }
      }
      if (cbIndex >= 0) {
        const originalCb = args[cbIndex] as (...cbArgs: unknown[]) => unknown;
        args[cbIndex] = function (this: unknown, ...cbArgs: unknown[]): unknown {
          adapter.emitToolCall(name, cbArgs[0]);
          return originalCb.apply(this, cbArgs);
        };
      }
      return (original as ToolRegistrar).apply(this, args);
    };
    (wrapped as unknown as Record<string, unknown>)[WRAP_FLAG] = true;
    server[method] = wrapped as ToolRegistrar;
  }

  /** Sign a governance receipt for one MCP tool call (fail-open). */
  emitToolCall(name: string, args: unknown): void {
    const argKeys = args && typeof args === "object"
      ? Object.keys(args as Record<string, unknown>)
      : [];
    this.signAction({
      actionType: "tool:call",
      context: { tool: name, arg_keys: argKeys },
    });
  }
}

/**
 * Turn on default-on Asqav governance for an MCP server. After this one
 * call, every tool registered on ``server`` signs a receipt on each call.
 * The server is instrumented in place and returned.
 */
export function enableMcpGovernance<T extends McpServerLike>(
  server: T,
  opts: AsqavMcpAdapterOptions,
): T {
  const adapter = new AsqavMcpAdapter(opts);
  return adapter.instrument(server);
}
