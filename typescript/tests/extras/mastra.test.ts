import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AsqavRefusedError } from "../../src/extras/_base.js";
import { AsqavMastraHook } from "../../src/extras/mastra.js";
import { makeFakeAgent, tick } from "../helpers/signStub.js";

async function makeMastraAgent(execute: () => Promise<unknown>): Promise<any> {
  const [{ Agent }, { createTool }] = await Promise.all([
    import("@mastra/core/agent"),
    import("@mastra/core/tools"),
  ]);
  const echo = createTool({ id: "echo", description: "echo", execute });
  return new Agent({
    id: "probe",
    name: "probe",
    instructions: "test agent",
    model: "openai/gpt-4o-mini",
    tools: { echo },
  });
}

type ExecutableTools = Record<string, {
  execute?: (input: unknown, context: unknown) => Promise<unknown>;
}>;

describe("AsqavMastraHook signing", () => {
  it("signs step:before, step:after, and step:error events", async () => {
    const { agent, calls, sign } = makeFakeAgent();
    const hook = new AsqavMastraHook({ agent });

    await hook.before({ stepId: "fetch-weather", stepType: "tool", input: { city: "Lisbon" } });
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
    expect(calls[0].result).toBeUndefined();
    expect(calls[1].result).toEqual({ temp: 22 });
    expect(calls[2].result).toBe("Error: nope");
  });

  it("survives hostile outputs: BigInt and circular values still sign", async () => {
    const { agent, calls, sign } = makeFakeAgent();
    const hook = new AsqavMastraHook({ agent });

    const circular: Record<string, unknown> = {};
    circular.self = circular;
    hook.after({ stepId: "s-big", stepType: "tool", output: { n: 10n } });
    hook.after({ stepId: "s-circ", stepType: "tool", output: circular });

    await tick();
    expect(sign).toHaveBeenCalledTimes(2);
    expect(calls[0].actionType).toBe("step:after");
    expect(typeof calls[0].context.output_preview).toBe("string");
    expect(typeof calls[1].context.output_preview).toBe("string");
    expect(calls[0].result).toBeDefined();
    expect(calls[1].result).toBeDefined();
  });
});

describe("AsqavMastraHook with the peer absent", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.doMock("@mastra/core", () => {
      throw new Error("Cannot find module '@mastra/core'");
    });
  });

  afterEach(() => {
    vi.doUnmock("@mastra/core");
  });

  it("attach() names the missing @mastra/core peer", async () => {
    const { AsqavMastraHook: Hook } = await import("../../src/extras/mastra.js");
    const { agent } = makeFakeAgent();
    const hook = new Hook({ agent });
    await expect(
      hook.attach({ listTools: async () => ({}), __setTools: () => undefined }),
    ).rejects.toThrow(/@mastra\/core/);
  });
});

describe("AsqavMastraHook against a current @mastra/core Agent", () => {
  it("rejects an object that is not a current Mastra Agent", async () => {
    const { agent } = makeFakeAgent();
    const hook = new AsqavMastraHook({ agent });
    await expect(hook.attach({})).rejects.toThrow(/listTools/);
  });

  it("ACCEPT: a refused signature stops an attached tool and raises", async () => {
    const { agent, sign } = makeFakeAgent({ failWith: new Error("policy denied") });
    const execute = vi.fn(async () => ({ ok: true }));
    const mastra = await makeMastraAgent(execute);
    const hook = new AsqavMastraHook({ agent });

    await hook.attach(mastra);
    const tools = await mastra.listTools() as ExecutableTools;

    await expect(tools.echo.execute!({ v: 1 }, {})).rejects.toBeInstanceOf(AsqavRefusedError);
    expect(execute).not.toHaveBeenCalled();
    expect(sign).toHaveBeenCalledTimes(1);
  });

  it("NEGCTL: an allowed signature runs an attached tool exactly once", async () => {
    const { agent, sign } = makeFakeAgent();
    const execute = vi.fn(async () => ({ ok: true }));
    const mastra = await makeMastraAgent(execute);
    const hook = new AsqavMastraHook({ agent });

    await hook.attach(mastra);
    const tools = await mastra.listTools() as ExecutableTools;
    const out = await tools.echo.execute!({ v: 2 }, {});

    expect(out).toEqual({ ok: true });
    expect(execute).toHaveBeenCalledTimes(1);
    expect(sign.mock.calls.map((c) => c[0].actionType)).toEqual(["step:before", "step:after"]);
  });

  it("toolHooks fit a real Agent's hooks and stop a refused tool", async () => {
    const { agent } = makeFakeAgent({ failWith: new Error("policy denied") });
    const execute = vi.fn(async () => ({ ok: true }));
    const hook = new AsqavMastraHook({ agent });
    const [{ Agent }, { createTool }] = await Promise.all([
      import("@mastra/core/agent"),
      import("@mastra/core/tools"),
    ]);
    const hooked = new Agent({
      id: "hooked",
      name: "hooked",
      instructions: "test agent",
      model: "openai/gpt-4o-mini",
      tools: { echo: createTool({ id: "echo", description: "echo", execute }) },
      hooks: hook.toolHooks,
    });

    const tools = await hooked.getToolsForExecution({}) as ExecutableTools;

    await expect(tools.echo.execute!({ v: 3 }, {})).rejects.toBeInstanceOf(AsqavRefusedError);
    expect(execute).not.toHaveBeenCalled();
  });
});
