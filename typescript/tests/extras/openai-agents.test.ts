import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AsqavOpenAIAgentsAdapter } from "../../src/extras/openai-agents.js";
import type { Agent } from "../../src/index.js";

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

describe("AsqavOpenAIAgentsAdapter without @openai/agents installed", () => {
  it("loadPeer() throws a clear ImportError", async () => {
    await expect(AsqavOpenAIAgentsAdapter.loadPeer()).rejects.toThrow(/@openai\/agents/);
  });
});

describe("AsqavOpenAIAgentsAdapter wrapTool + guardrail hooks", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.doMock("@openai/agents", () => ({ __openaiAgentsStub: true }));
  });

  afterEach(() => {
    vi.doUnmock("@openai/agents");
  });

  it("signs tool:start and tool:end around a successful tool execute", async () => {
    const { agent, calls, sign } = makeFakeAgent();
    const adapter = new AsqavOpenAIAgentsAdapter({ agent });

    const tool = {
      name: "weather",
      execute: async (input: { city: string }) => ({ city: input.city, temp: 22 }),
    };
    const wrapped = adapter.wrapTool(tool);
    const out = await wrapped.execute!({ city: "Lisbon" });

    await tick();
    expect(out).toEqual({ city: "Lisbon", temp: 22 });
    expect(sign).toHaveBeenCalledTimes(2);
    expect(calls.map((c) => c.actionType)).toEqual(["tool:start", "tool:end"]);
    expect(calls[0].context.tool).toBe("weather");
    expect(calls[1].context.output_length).toBeGreaterThan(0);
  });

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
  });

  it("signs agent:input and agent:output via guardrail helpers", async () => {
    const { agent, calls } = makeFakeAgent();
    const adapter = new AsqavOpenAIAgentsAdapter({ agent });

    adapter.onAgentInput("hello", "agent-1");
    adapter.onAgentOutput({ answer: 42 }, "agent-1");

    await tick();
    expect(calls.map((c) => c.actionType)).toEqual(["agent:input", "agent:output"]);
    expect(calls[0].context.agent_name).toBe("agent-1");
    expect(calls[1].context.output_length).toBeGreaterThan(0);
  });
});
