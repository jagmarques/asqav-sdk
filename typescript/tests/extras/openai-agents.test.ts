import { describe, expect, it, vi } from "vitest";
import { AsqavRefusedError } from "../../src/extras/_base.js";
import { AsqavOpenAIAgentsAdapter } from "../../src/extras/openai-agents.js";
import type { Agent } from "../../src/index.js";

const [nodeMajor, nodeMinor] = process.versions.node.split(".").map(Number);
const modernNode = nodeMajor > 22 || (nodeMajor === 22 && nodeMinor >= 13);

function makeFakeAgent(opts: { failWith?: Error } = {}) {
  const calls: Array<{ actionType: string; context: Record<string, unknown>; result?: unknown }> = [];
  const sign = vi.fn(async (opts2: { actionType: string; context?: Record<string, unknown>; result?: unknown }) => {
    calls.push({ actionType: opts2.actionType, context: opts2.context ?? {}, result: opts2.result });
    if (opts.failWith) throw opts.failWith;
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

async function makeOpenAITool(execute: (input: { city: string }) => Promise<unknown>): Promise<any> {
  const { tool } = await import("@openai/agents");
  return tool({
    name: "weather",
    description: "weather for a city",
    strict: false,
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
      additionalProperties: false,
    },
    execute,
  } as never);
}

describe.skipIf(!modernNode)("AsqavOpenAIAgentsAdapter against a current FunctionTool", () => {
  it("wraps the current FunctionTool invoke surface", async () => {
    const { agent, calls, sign } = makeFakeAgent();
    const adapter = new AsqavOpenAIAgentsAdapter({ agent });
    const execute = vi.fn(async (input: { city: string }) => `hi ${input.city}`);

    const wrapped = adapter.wrapTool(await makeOpenAITool(execute));
    const out = await wrapped.invoke!({}, JSON.stringify({ city: "Lisbon" }));

    expect(out).toBe("hi Lisbon");
    expect(execute).toHaveBeenCalledTimes(1);
    expect(sign).toHaveBeenCalledTimes(2);
    expect(calls.map((c) => c.actionType)).toEqual(["tool:start", "tool:end"]);
    expect(calls[0].context.tool).toBe("weather");
  });

  it("ACCEPT: a refused signature stops the tool and raises", async () => {
    const { agent, sign } = makeFakeAgent({ failWith: new Error("policy denied") });
    const adapter = new AsqavOpenAIAgentsAdapter({ agent });
    const execute = vi.fn(async () => "ran");

    const wrapped = adapter.wrapTool(await makeOpenAITool(execute));

    await expect(
      wrapped.invoke!({}, JSON.stringify({ city: "Lisbon" })),
    ).rejects.toBeInstanceOf(AsqavRefusedError);
    expect(execute).not.toHaveBeenCalled();
    expect(sign).toHaveBeenCalledTimes(1);
  });

  it("NEGCTL: an allowed signature runs the tool exactly once", async () => {
    const { agent, sign } = makeFakeAgent();
    const adapter = new AsqavOpenAIAgentsAdapter({ agent });
    const execute = vi.fn(async (input: { city: string }) => ({ city: input.city }));

    const wrapped = adapter.wrapTool(await makeOpenAITool(execute));
    const out = await wrapped.invoke!({}, JSON.stringify({ city: "Lisbon" }));

    expect(out).toEqual({ city: "Lisbon" });
    expect(execute).toHaveBeenCalledTimes(1);
    expect(sign).toHaveBeenCalledTimes(2);
  });

  it("records the framework's error result from a FunctionTool", async () => {
    const { agent, calls } = makeFakeAgent();
    const adapter = new AsqavOpenAIAgentsAdapter({ agent });
    const wrapped = adapter.wrapTool(await makeOpenAITool(async () => {
      throw new Error("kaboom");
    }));

    const out = await wrapped.invoke!({}, JSON.stringify({ city: "Lisbon" }));

    expect(String(out)).toMatch(/error occurred/i);
    expect(calls.map((c) => c.actionType)).toEqual(["tool:start", "tool:end"]);
  });
});

describe("AsqavOpenAIAgentsAdapter tool and guardrail wrappers", () => {
  it("signs tool:error and re-throws when the underlying tool fails", async () => {
    const { agent, calls, sign } = makeFakeAgent();
    const adapter = new AsqavOpenAIAgentsAdapter({ agent });

    const tool = {
      name: "broken",
      execute: async () => {
        throw new Error("kaboom");
      },
    };
    const wrapped = adapter.wrapTool(tool);
    await expect(wrapped.execute!(null)).rejects.toThrow(/kaboom/);

    await tick();
    expect(sign).toHaveBeenCalledTimes(2);
    expect(calls.map((c) => c.actionType)).toEqual(["tool:start", "tool:error"]);
    expect(calls[1].context.error).toBe("kaboom");
    // 787: the failure travels as the receipt result
    expect(calls[1].result).toBe("Error: kaboom");
  });

  it("NEGCTL: an execute-only tool still runs once on an allowed signature", async () => {
    const { agent, sign } = makeFakeAgent();
    const adapter = new AsqavOpenAIAgentsAdapter({ agent });
    const execute = vi.fn(async (input: { city: string }) => input.city);

    const wrapped = adapter.wrapTool({ name: "echo", execute });
    const out = await wrapped.execute!({ city: "Porto" });

    expect(out).toBe("Porto");
    expect(execute).toHaveBeenCalledTimes(1);
    expect(sign.mock.calls.map((c) => c[0].actionType)).toEqual(["tool:start", "tool:end"]);
  });

  it("signs agent:input and agent:output via guardrail helpers", async () => {
    const { agent, calls } = makeFakeAgent();
    const adapter = new AsqavOpenAIAgentsAdapter({ agent });

    adapter.onAgentInput("hello", "agent-1");
    adapter.onAgentOutput({ answer: 42 }, "agent-1");

    await tick();
    expect(calls.map((c) => c.actionType)).toEqual(["agent:input", "agent:output"]);
    expect(calls[0].context.agent_name).toBe("agent-1");
    expect(calls[1].result).toEqual({ answer: 42 });
  });
});
