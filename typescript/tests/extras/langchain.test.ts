import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Agent } from "../../src/index.js";

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

describe("AsqavCallbackHandler without @langchain/core installed", () => {
  it("throws a clear ImportError when create() is called", async () => {
    const { AsqavCallbackHandler } = await import("../../src/extras/langchain.js");
    const { agent } = makeFakeAgent();
    await expect(
      AsqavCallbackHandler.create({ agent }),
    ).rejects.toThrow(/@langchain\/core/);
  });
});

describe("AsqavCallbackHandler with @langchain/core mocked", () => {
  beforeEach(() => {
    vi.resetModules();
    // Provide a fake @langchain/core/callbacks/base so loadBase() succeeds.
    vi.doMock("@langchain/core/callbacks/base", () => {
      class BaseCallbackHandler {
        // The real class has many no-op handlers; an empty base is enough.
      }
      return { BaseCallbackHandler };
    });
  });

  afterEach(() => {
    vi.doUnmock("@langchain/core/callbacks/base");
  });

  it("signs handleLLMStart, handleLLMEnd, and tool events into the agent", async () => {
    const { AsqavCallbackHandler } = await import("../../src/extras/langchain.js");
    const { agent, calls, sign } = makeFakeAgent();
    const handler = await AsqavCallbackHandler.create({ agent });

    interface InnerHandler {
      handleLLMStart(s: Record<string, unknown>, prompts: string[]): void;
      handleLLMEnd(r: Record<string, unknown>): void;
      handleToolStart(s: Record<string, unknown>, input: string): void;
      handleToolEnd(out: unknown): void;
      handleLLMError(err: Error): void;
    }
    const inner = handler as unknown as InnerHandler;

    inner.handleLLMStart({ name: "gpt-4o" }, ["hello"]);
    inner.handleLLMEnd({ generations: [[{ text: "hi" }]], llmOutput: { tokenUsage: { total: 10 } } });
    inner.handleToolStart({ name: "weather" }, "lisbon");
    inner.handleToolEnd("sunny");
    inner.handleLLMError(new Error("boom"));

    await tick();
    expect(sign).toHaveBeenCalledTimes(5);
    expect(calls.map((c) => c.actionType)).toEqual([
      "llm:start",
      "llm:end",
      "tool:start",
      "tool:end",
      "llm:error",
    ]);
    expect(calls[0].context.model).toBe("gpt-4o");
    expect(calls[1].context.generation_count).toBe(1);
    expect(calls[1].context.token_usage).toEqual({ total: 10 });
    expect(calls[2].context.tool).toBe("weather");
    expect(calls[3].context.output_length).toBe("sunny".length);
    expect(calls[4].context.error).toBe("boom");
  });
});
