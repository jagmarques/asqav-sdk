import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AsqavMastraHook } from "../../src/extras/mastra.js";
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

describe("AsqavMastraHook without @mastra/core installed", () => {
  it("attach() throws a clear ImportError", async () => {
    const { agent } = makeFakeAgent();
    const hook = new AsqavMastraHook({ agent });
    await expect(
      hook.attach({ on: () => undefined }),
    ).rejects.toThrow(/@mastra\/core/);
  });
});

describe("AsqavMastraHook before/after wiring", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.doMock("@mastra/core", () => ({ __mastraStub: true }));
  });

  afterEach(() => {
    vi.doUnmock("@mastra/core");
  });

  it("signs step:before, step:after, and step:error events", async () => {
    const { AsqavMastraHook: Hook } = await import("../../src/extras/mastra.js");
    const { agent, calls, sign } = makeFakeAgent();
    const hook = new Hook({ agent });

    hook.before({ stepId: "fetch-weather", stepType: "tool", input: { city: "Lisbon" } });
    hook.after({ stepId: "fetch-weather", stepType: "tool", output: { temp: 22 } });
    hook.onError({ stepId: "fetch-weather", stepType: "tool", error: new Error("nope") });

    await tick();
    expect(sign).toHaveBeenCalledTimes(3);
    expect(calls.map((c) => c.actionType)).toEqual([
      "step:before",
      "step:after",
      "step:error",
    ]);
    expect(calls[0].context.step_id).toBe("fetch-weather");
    expect(calls[1].context.output_length).toBe(JSON.stringify({ temp: 22 }).length);
    expect(calls[2].context.error).toBe("nope");
  });

  it("attach(agent.on) subscribes to step:before / step:after / step:error", async () => {
    const { AsqavMastraHook: Hook } = await import("../../src/extras/mastra.js");
    const { agent, sign } = makeFakeAgent();
    const hook = new Hook({ agent });

    const listeners: Record<string, (evt: unknown) => void> = {};
    const fakeMastra = {
      on(evt: string, fn: (evt: unknown) => void) {
        listeners[evt] = fn;
      },
    };

    await hook.attach(fakeMastra);
    listeners["step:before"]({ stepId: "s1", stepType: "tool" });
    listeners["step:after"]({ stepId: "s1", stepType: "tool", output: "ok" });

    await tick();
    expect(sign).toHaveBeenCalledTimes(2);
  });
});
