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
 * server: every tool call signs an Asqav receipt by default, regardless of
 * whether the tool was registered before or after the call, and a tool that
 * throws signs ``tool:error``. Signing is fail-open. This module never
 * imports ``@modelcontextprotocol/sdk`` (it duck-types the passed server),
 * so the peer stays optional and importing without calling changes nothing.
 */

import { AsqavAdapter, defaultOnError, type AsqavAdapterOptions } from "./_base.js";

// Boundary type for the duck-typed server. ``any`` rest params let a
// concretely-typed McpServer.registerTool/tool satisfy this shape.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ToolRegistrar = (...args: any[]) => any;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyFn = (...args: any[]) => any;

/** Minimal server shape: a ``_registeredTools`` registry, or registrar-only (registerTool/tool). */
export interface McpServerLike {
  _registeredTools?: Record<string, unknown>;
  registerTool?: ToolRegistrar;
  tool?: ToolRegistrar;
}

export type AsqavMcpAdapterOptions = AsqavAdapterOptions;

const WRAP_FLAG = "__asqavWrapped";
// Symbol key so the funnel marker never shows up in Object.keys/entries
// (McpServer enumerates _registeredTools to build tools/list).
const FUNNEL_FLAG = Symbol.for("asqav.mcp.funnel");
// Dispatch-time handler field: ``handler`` on recent SDK versions,
// ``callback`` on older ones. Wrap whichever is present.
const HANDLER_KEYS = ["handler", "callback"] as const;
const MAX_ERROR_LEN = 200;

/** Signs ``tool:call`` per MCP tool call and ``tool:error`` on a throw, fail-open and order-independent. */
export class AsqavMcpAdapter extends AsqavAdapter {
  /** Wire signing into ``server``'s tool dispatch in place; returns ``server``. */
  instrument<T extends McpServerLike>(server: T): T {
    const registry = server._registeredTools;
    if (registry && typeof registry === "object") {
      this.wrapDispatchRegistry(server);
      return server;
    }
    const hasRegister = typeof server.registerTool === "function";
    const hasTool = typeof server.tool === "function";
    if (!hasRegister && !hasTool) {
      throw new Error(
        "enableMcpGovernance expects an MCP server exposing a tool registry, "
        + "registerTool or tool (e.g. McpServer from @modelcontextprotocol/sdk).",
      );
    }
    if (hasRegister) this.wrapRegistrar(server, "registerTool");
    if (hasTool) this.wrapRegistrar(server, "tool");
    return server;
  }

  // Single dispatch funnel: every tools/call lookup reads the registry, so a
  // Proxy get that wraps the handler signs order-independently.
  private wrapDispatchRegistry(server: McpServerLike): void {
    const target = server._registeredTools as Record<string | symbol, unknown>;
    if (target[FUNNEL_FLAG]) return;
    const adapter = this;

    const proxy = new Proxy(target, {
      get(t, prop, receiver): unknown {
        const value = Reflect.get(t, prop, receiver);
        if (typeof prop !== "string" || value === null || typeof value !== "object") {
          return value;
        }
        const tool = value as Record<string, unknown>;
        let wrapped: Record<string, unknown> | null = null;
        for (const key of HANDLER_KEYS) {
          if (typeof tool[key] === "function") {
            // Shallow copy keeps enabled/schemas/update intact; the stored
            // tool stays unwrapped so re-reads never double-wrap.
            wrapped = wrapped ?? { ...tool };
            wrapped[key] = adapter.wrapToolCallback(prop, tool[key] as AnyFn);
          }
        }
        return wrapped ?? value;
      },
    });

    target[FUNNEL_FLAG] = true;
    server._registeredTools = proxy as Record<string, unknown>;
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
        args[cbIndex] = adapter.wrapToolCallback(name, args[cbIndex] as AnyFn);
      }
      return (original as ToolRegistrar).apply(this, args);
    };
    (wrapped as unknown as Record<string, unknown>)[WRAP_FLAG] = true;
    server[method] = wrapped as ToolRegistrar;
  }

  // Shared per-call wrap: sign tool:call, run the tool, sign tool:error on a
  // sync throw or an async rejection, then rethrow.
  private wrapToolCallback(name: string, originalCb: AnyFn): AnyFn {
    if ((originalCb as unknown as Record<string, unknown>)[WRAP_FLAG]) return originalCb;
    const adapter = this;

    const wrapped = function (this: unknown, ...cbArgs: unknown[]): unknown {
      adapter.emitToolCall(name, cbArgs[0]);
      let out: unknown;
      try {
        out = originalCb.apply(this, cbArgs);
      } catch (err) {
        adapter.emitToolError(name, err);
        throw err;
      }
      if (out instanceof Promise) {
        return out.catch((err: unknown) => {
          adapter.emitToolError(name, err);
          throw err;
        });
      }
      return out;
    };
    (wrapped as unknown as Record<string, unknown>)[WRAP_FLAG] = true;
    return wrapped;
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

  /** Sign tool:error when a wrapped tool throws (fail-open). */
  emitToolError(name: string, err: unknown): void {
    // String(err) can itself throw (e.g. Object.create(null)). Keep the
    // stringification inside the fail-open boundary so it can never mask
    // the tool's own error.
    try {
      const errorType = err instanceof Error
        ? err.constructor.name
        : typeof err;
      const message = err instanceof Error ? err.message : String(err);
      this.signAction({
        actionType: "tool:error",
        context: {
          tool: name,
          error_type: errorType,
          error: message.slice(0, MAX_ERROR_LEN),
        },
      });
    } catch (emitErr) {
      const onError = this.options.onError ?? defaultOnError;
      try {
        onError(emitErr, { actionType: "tool:error" });
      } catch {
        // Swallow secondary errors from the user-supplied handler.
      }
    }
  }
}

/** Default-on Asqav governance: every tool call signs a receipt, even tools registered before this call. */
export function enableMcpGovernance<T extends McpServerLike>(
  server: T,
  opts: AsqavMcpAdapterOptions,
): T {
  const adapter = new AsqavMcpAdapter(opts);
  return adapter.instrument(server);
}
