import { describe, expect, it, vi } from "vitest";
import type { Agent } from "../../src/index.js";
import { AsqavMcpAdapter, enableMcpGovernance } from "../../src/extras/mcp.js";

// Fake Asqav agent: hermetic, no init() / network.
function makeFakeAgent() {
  const calls: Array<{ actionType: string; context: Record<string, unknown> }> = [];
  const sign = vi.fn(async (opts: { actionType: string; context?: Record<string, unknown> }) => {
    calls.push({ actionType: opts.actionType, context: opts.context ?? {} });
    return {
      signature: "sig",
      signatureId: "sig_1",
      actionId: "act_1",
      timestamp: "2026-01-01T00:00:00Z",
      verificationUrl: "https://asqav.com/verify/sig_1",
    };
  });
  return { agent: { agentId: "agt_fake", sign } as unknown as Agent, calls, sign };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyFn = (...a: any[]) => any;

// Fake McpServer mirroring the real SDK: tools live in _registeredTools and
// every call funnels through a dispatch that reads the registry at call time.
class FakeMcpServer {
  public _registeredTools: Record<string, { handler: AnyFn; enabled: boolean }> = {};

  registerTool(name: string, _config: unknown, cb: AnyFn): { name: string } {
    this._registeredTools[name] = { handler: cb, enabled: true };
    return { name };
  }

  tool(name: string, ...rest: unknown[]): { name: string } {
    this._registeredTools[name] = { handler: rest[rest.length - 1] as AnyFn, enabled: true };
    return { name };
  }

  // Mirrors the SDK's tools/call handler: registry lookup, then run.
  async callTool(name: string, args: unknown): Promise<unknown> {
    const tool = this._registeredTools[name];
    if (!tool) throw new Error(`Tool ${name} not found`);
    return tool.handler(args);
  }
}

// Duck-typed server with no registry: exercises the registrar fallback.
class RegistrarOnlyServer {
  public tools: Record<string, AnyFn> = {};

  registerTool(name: string, _config: unknown, cb: AnyFn): { name: string } {
    this.tools[name] = cb;
    return { name };
  }
}

describe("enableMcpGovernance", () => {
  it("throws when the server exposes neither registerTool nor tool", () => {
    const { agent } = makeFakeAgent();
    expect(() => enableMcpGovernance({} as never, { agent })).toThrow(/registerTool or tool/);
  });

  it("signs tool:call by default on every registered tool call", async () => {
    const { agent, calls, sign } = makeFakeAgent();
    const server = new FakeMcpServer();
    enableMcpGovernance(server, { agent });

    const impl = vi.fn(async () => ({ content: [{ type: "text", text: "9" }] }));
    server.registerTool("add", { description: "add" }, impl);
    const out = await server.callTool("add", { a: 4, b: 5 });
    await tick();

    expect(sign).toHaveBeenCalledTimes(1);
    expect(calls[0].actionType).toBe("tool:call");
    expect(calls[0].context.tool).toBe("add");
    expect(calls[0].context.arg_keys).toEqual(["a", "b"]);
    expect(impl).toHaveBeenCalledOnce();
    expect(out).toEqual({ content: [{ type: "text", text: "9" }] });
  });

  it("signs calls of a tool registered BEFORE enable (order-independent)", async () => {
    const { agent, calls, sign } = makeFakeAgent();
    const server = new FakeMcpServer();
    // Registration first, governance second: the dispatch funnel must still sign.
    server.registerTool("early", {}, async (args: { x: number }) => args.x * 2);
    enableMcpGovernance(server, { agent });

    const out = await server.callTool("early", { x: 21 });
    await tick();

    expect(out).toBe(42);
    expect(sign).toHaveBeenCalledTimes(1);
    expect(calls[0].actionType).toBe("tool:call");
    expect(calls[0].context.tool).toBe("early");
    expect(calls[0].context.arg_keys).toEqual(["x"]);
  });

  it("signs tool:error when a tool throws, then rethrows", async () => {
    const { agent, calls } = makeFakeAgent();
    const server = new FakeMcpServer();
    enableMcpGovernance(server, { agent });

    server.registerTool("boom", {}, async () => {
      throw new TypeError("kaboom");
    });
    await expect(server.callTool("boom", { a: 1 })).rejects.toThrow("kaboom");
    await tick();

    expect(calls.map((c) => c.actionType)).toEqual(["tool:call", "tool:error"]);
    expect(calls[1].context.tool).toBe("boom");
    expect(calls[1].context.error_type).toBe("TypeError");
    expect(calls[1].context.error).toBe("kaboom");
  });

  it("truncates tool:error messages to 200 chars and handles sync throws", async () => {
    const { agent, calls } = makeFakeAgent();
    const server = new FakeMcpServer();
    enableMcpGovernance(server, { agent });

    const long = "x".repeat(300);
    server.registerTool("sync-boom", {}, () => {
      throw new Error(long);
    });
    await expect(server.callTool("sync-boom", {})).rejects.toThrow();
    await tick();

    expect(calls[1].actionType).toBe("tool:error");
    expect((calls[1].context.error as string).length).toBe(200);
  });

  it("stays fail-open when a tool throws a non-stringifiable value", async () => {
    const { agent } = makeFakeAgent();
    const server = new FakeMcpServer();
    enableMcpGovernance(server, { agent });

    // Object.create(null) has no toString/Symbol.toPrimitive: String(err)
    // throws TypeError. The adapter must not let that mask the tool's own
    // thrown value.
    const bare = Object.create(null);
    server.registerTool("bare-boom", {}, () => {
      throw bare;
    });

    await expect(server.callTool("bare-boom", {})).rejects.toBe(bare);
    await tick();
  });

  it("is non-vacuous: an un-enabled server signs nothing (mutation control)", async () => {
    const { sign } = makeFakeAgent();
    const server = new FakeMcpServer();
    // No enableMcpGovernance call - the wrap is what drives signing.
    const impl = vi.fn(async () => "ok");
    server.registerTool("noop", {}, impl);
    await server.callTool("noop", {});
    await tick();
    expect(sign).not.toHaveBeenCalled();
  });

  it("wraps the deprecated tool() registrar too", async () => {
    const { agent, calls } = makeFakeAgent();
    const server = new FakeMcpServer();
    enableMcpGovernance(server, { agent });

    server.tool("echo", async (args: { msg: string }) => args.msg);
    await server.callTool("echo", { msg: "hi" });
    await tick();

    expect(calls[0].actionType).toBe("tool:call");
    expect(calls[0].context.tool).toBe("echo");
    expect(calls[0].context.arg_keys).toEqual(["msg"]);
  });

  it("is fail-open: a signing error never blocks the tool call", async () => {
    const { agent, sign } = makeFakeAgent();
    (sign as unknown as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("sign down"));
    const server = new FakeMcpServer();
    enableMcpGovernance(server, { agent });

    const impl = vi.fn(async () => "done");
    server.registerTool("run", {}, impl);
    const out = await server.callTool("run", { x: 1 });
    await tick();
    expect(out).toBe("done");
  });

  it("is idempotent: enabling twice signs once per tool call", async () => {
    const { agent, sign } = makeFakeAgent();
    const server = new FakeMcpServer();
    enableMcpGovernance(server, { agent });
    enableMcpGovernance(server, { agent });

    server.registerTool("t", {}, async () => "r");
    await server.callTool("t", { a: 1 });
    await tick();
    expect(sign).toHaveBeenCalledTimes(1);
  });

  it("preserves the tool's return value through the wrap", async () => {
    const { agent } = makeFakeAgent();
    const server = new FakeMcpServer();
    const adapter = enableMcpGovernance(server, { agent });
    expect(adapter).toBeInstanceOf(FakeMcpServer);

    server.registerTool("mul", {}, async (args: { a: number; b: number }) => args.a * args.b);
    const out = await server.callTool("mul", { a: 6, b: 7 });
    expect(out).toBe(42);
  });

  it("keeps the registry enumerable and flag-free for tools/list", () => {
    const { agent } = makeFakeAgent();
    const server = new FakeMcpServer();
    enableMcpGovernance(server, { agent });
    server.registerTool("a", {}, async () => "r");
    server.registerTool("b", {}, async () => "r");
    expect(Object.keys(server._registeredTools).sort()).toEqual(["a", "b"]);
    expect(server._registeredTools.a.enabled).toBe(true);
  });

  it("falls back to registrar wrapping when the server has no registry", async () => {
    const { agent, calls } = makeFakeAgent();
    const server = new RegistrarOnlyServer();
    enableMcpGovernance(server, { agent });

    server.registerTool("plainduck", {}, async () => {
      throw new Error("bad");
    });
    await expect(server.tools.plainduck({ q: 1 })).rejects.toThrow("bad");
    await tick();

    expect(calls.map((c) => c.actionType)).toEqual(["tool:call", "tool:error"]);
    expect(calls[0].context.tool).toBe("plainduck");
    expect(calls[1].context.error).toBe("bad");
  });
});

describe("AsqavMcpAdapter", () => {
  it("instrument returns the same server instance", () => {
    const { agent } = makeFakeAgent();
    const adapter = new AsqavMcpAdapter({ agent });
    const server = new FakeMcpServer();
    expect(adapter.instrument(server)).toBe(server);
  });
});
