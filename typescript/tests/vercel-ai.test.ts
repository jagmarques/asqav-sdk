import { describe, expect, it, vi } from "vitest";
import {
  AsqavBlockedError,
  asqavGuard,
  createAsqavExporter,
  mapSpanNameToActionType,
  wrapTools,
  type Span,
} from "../src/extras/vercel-ai.js";
import type { Agent } from "../src/index.js";
import { makeFakeAgent, tick } from "./helpers/signStub.js";

describe("mapSpanNameToActionType", () => {
  it("maps known AI SDK span names", () => {
    expect(mapSpanNameToActionType("ai.generateText")).toBe("ai:completion");
    expect(mapSpanNameToActionType("ai.streamText")).toBe("ai:completion_stream");
    expect(mapSpanNameToActionType("ai.toolCall")).toBe("ai:tool_call");
    expect(mapSpanNameToActionType("ai.toolCall.weather")).toBe("ai:tool_call");
    expect(mapSpanNameToActionType("ai.embed")).toBe("ai:embed");
    expect(mapSpanNameToActionType("ai.embedMany")).toBe("ai:embed");
    expect(mapSpanNameToActionType("ai.generateObject")).toBe("ai:object");
  });

  it("falls back to ai:<rest> for unknown ai.* names", () => {
    expect(mapSpanNameToActionType("ai.something.weird")).toBe("ai:something_weird");
  });

  it("passes non-ai names through unchanged", () => {
    expect(mapSpanNameToActionType("custom.span")).toBe("custom.span");
    expect(mapSpanNameToActionType("")).toBe("ai:span");
  });
});

describe("createAsqavExporter", () => {
  it("returns an object with the OTel Tracer surface", () => {
    const { agent } = makeFakeAgent();
    const tracer = createAsqavExporter({ agent });
    expect(typeof tracer.startSpan).toBe("function");
    expect(typeof tracer.startActiveSpan).toBe("function");

    const span = tracer.startSpan("ai.generateText");
    expect(typeof span.setAttribute).toBe("function");
    expect(typeof span.setAttributes).toBe("function");
    expect(typeof span.setStatus).toBe("function");
    expect(typeof span.recordException).toBe("function");
    expect(typeof span.end).toBe("function");
    expect(typeof span.isRecording).toBe("function");
    expect(span.spanContext()).toEqual({ traceId: "", spanId: "", traceFlags: 0 });
  });

  it("requires either an agent or agentId", () => {
    expect(() => createAsqavExporter({} as never)).toThrow(/agent/i);
  });
});

describe("span.end -> sign call mapping", () => {
  it("signs with mapped action_type and forwards attributes + duration", async () => {
    const { agent, calls, sign } = makeFakeAgent();
    const tracer = createAsqavExporter({ agent });

    const span = tracer.startSpan("ai.generateText", {
      attributes: { "ai.model.id": "gpt-4o" },
    });
    span.setAttribute("ai.usage.completionTokens", 42);
    span.end();

    await tick();
    expect(sign).toHaveBeenCalledTimes(1);
    expect(calls[0].actionType).toBe("ai:completion");
    expect(calls[0].context["ai.model.id"]).toBe("gpt-4o");
    expect(calls[0].context["ai.usage.completionTokens"]).toBe(42);
    expect(calls[0].context.span_name).toBe("ai.generateText");
    expect(typeof calls[0].context.duration_ms).toBe("number");
    // 787: span attributes carry the outcome as the receipt result
    expect(calls[0].result).toEqual({ "ai.model.id": "gpt-4o", "ai.usage.completionTokens": 42 });
  });

  it("maps ai.toolCall span name to ai:tool_call", async () => {
    const { agent, calls } = makeFakeAgent();
    const tracer = createAsqavExporter({ agent });

    const span = tracer.startSpan("ai.toolCall");
    span.setAttribute("ai.toolCall.name", "getWeather");
    span.end();

    await tick();
    expect(calls[0].actionType).toBe("ai:tool_call");
    expect(calls[0].context["ai.toolCall.name"]).toBe("getWeather");
  });

  it("is idempotent: calling end() twice signs once", async () => {
    const { agent, sign } = makeFakeAgent();
    const tracer = createAsqavExporter({ agent });
    const span = tracer.startSpan("ai.generateText");
    span.end();
    span.end();
    await tick();
    expect(sign).toHaveBeenCalledTimes(1);
  });
});

describe("recordException + end signs as ai:error", () => {
  it("emits ai:error with the original action type preserved in context", async () => {
    const { agent, calls } = makeFakeAgent();
    const tracer = createAsqavExporter({ agent });

    const span = tracer.startSpan("ai.generateText");
    const err = new Error("model timed out");
    span.recordException(err);
    span.setStatus({ code: 2, message: "model timed out" });
    span.end();

    await tick();
    expect(calls[0].actionType).toBe("ai:error");
    expect(calls[0].context.original_action_type).toBe("ai:completion");
    expect(calls[0].context.error).toEqual({ name: "Error", message: "model timed out" });
  });
});

describe("fail-open: never throws when sign fails", () => {
  it("invokes onError but does not propagate", async () => {
    const { agent } = makeFakeAgent({ failWith: new Error("network down") });
    const onError = vi.fn();
    const tracer = createAsqavExporter({ agent, onError });

    const span = tracer.startSpan("ai.generateText");
    expect(() => span.end()).not.toThrow();

    await tick();
    expect(onError).toHaveBeenCalledTimes(1);
    const [errArg, spanInfo] = onError.mock.calls[0];
    expect((errArg as Error).message).toBe("network down");
    expect(spanInfo.name).toBe("ai.generateText");
  });
});

describe("startActiveSpan", () => {
  it("invokes the callback with a span and ends it after sync work", async () => {
    const { agent, sign } = makeFakeAgent();
    const tracer = createAsqavExporter({ agent });

    const result = tracer.startActiveSpan("ai.generateText", (span: Span) => {
      span.setAttribute("ai.model.id", "gpt-4o");
      return 42;
    });

    expect(result).toBe(42);
    await tick();
    expect(sign).toHaveBeenCalledTimes(1);
  });

  it("records exceptions thrown inside the callback", async () => {
    const { agent, calls } = makeFakeAgent();
    const tracer = createAsqavExporter({ agent });

    expect(() =>
      tracer.startActiveSpan("ai.generateText", () => {
        throw new Error("boom");
      }),
    ).toThrow(/boom/);

    await tick();
    expect(calls[0].actionType).toBe("ai:error");
  });
});

async function makeAiTool(
  execute: (input: { city: string }, options: unknown) => Promise<unknown>,
) {
  const { jsonSchema, tool } = await import("ai");
  const built = tool({
    description: "echo a city",
    inputSchema: jsonSchema({
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
      additionalProperties: false,
    }),
    execute,
  } as never);
  return built as unknown as {
    execute?: (input: { city: string }, options: unknown) => Promise<unknown>;
  };
}

describe("fail-closed tool guard", () => {
  it("ACCEPT: a refused signature stops the tool and raises", async () => {
    const { agent, sign } = makeFakeAgent({ failWith: new Error("policy denied") });
    const execute = vi.fn(async () => ({ temp: 22 }));
    const wrapped = asqavGuard(await makeAiTool(execute), { agent, toolName: "weather" });

    await expect(
      wrapped.execute!({ city: "Lisbon" }, {}),
    ).rejects.toBeInstanceOf(AsqavBlockedError);

    expect(execute).not.toHaveBeenCalled();
    expect(sign).toHaveBeenCalledTimes(1);
  });

  it("NEGCTL: an allowed signature runs the tool exactly once", async () => {
    const { agent, sign } = makeFakeAgent();
    const execute = vi.fn(async (input: { city: string }) => ({
      city: input.city,
      temp: 22,
    }));
    const wrapped = wrapTools({ weather: await makeAiTool(execute) }, { agent });

    const out = await wrapped.weather.execute!({ city: "Lisbon" }, {});

    expect(out).toEqual({ city: "Lisbon", temp: 22 });
    expect(execute).toHaveBeenCalledTimes(1);
    expect(sign).toHaveBeenCalledTimes(1);
  });

  it("blocks a non-signed verdict before execute", async () => {
    const agent = {
      agentId: "agt_fake",
      sign: vi.fn(async () => ({})),
    } as unknown as Agent;
    const execute = vi.fn(async () => ({ ok: true }));
    const wrapped = asqavGuard(await makeAiTool(execute), { agent });

    await expect(
      wrapped.execute!({ city: "Lisbon" }, {}),
    ).rejects.toBeInstanceOf(AsqavBlockedError);

    expect(execute).not.toHaveBeenCalled();
  });

  it("blocks a signed deny decision before execute", async () => {
    const { agent } = makeFakeAgent({ receipt: { decision: "deny" } });
    const execute = vi.fn(async () => ({ temp: 22 }));
    const wrapped = asqavGuard(await makeAiTool(execute), { agent, toolName: "weather" });

    await expect(
      wrapped.execute!({ city: "Lisbon" }, {}),
    ).rejects.toBeInstanceOf(AsqavBlockedError);

    expect(execute).not.toHaveBeenCalled();
  });

  it("blocks an upper-case DENY decision before execute", async () => {
    const { agent } = makeFakeAgent({ receipt: { decision: "DENY" } });
    const execute = vi.fn(async () => ({ temp: 22 }));
    const wrapped = asqavGuard(await makeAiTool(execute), { agent, toolName: "weather" });

    await expect(
      wrapped.execute!({ city: "Lisbon" }, {}),
    ).rejects.toBeInstanceOf(AsqavBlockedError);

    expect(execute).not.toHaveBeenCalled();
  });

  it("blocks a signed deny policyDecision before execute", async () => {
    const { agent } = makeFakeAgent({ receipt: { policyDecision: "deny" } });
    const execute = vi.fn(async () => ({ temp: 22 }));
    const wrapped = asqavGuard(await makeAiTool(execute), { agent, toolName: "weather" });

    await expect(
      wrapped.execute!({ city: "Lisbon" }, {}),
    ).rejects.toBeInstanceOf(AsqavBlockedError);

    expect(execute).not.toHaveBeenCalled();
  });

  it("blocks a signed rate_limit decision before execute", async () => {
    const { agent } = makeFakeAgent({ receipt: { decision: "rate_limit" } });
    const execute = vi.fn(async () => ({ temp: 22 }));
    const wrapped = asqavGuard(await makeAiTool(execute), { agent, toolName: "weather" });

    await expect(
      wrapped.execute!({ city: "Lisbon" }, {}),
    ).rejects.toBeInstanceOf(AsqavBlockedError);

    expect(execute).not.toHaveBeenCalled();
  });

  it("blocks an object signature with empty sig bytes before execute", async () => {
    const { agent } = makeFakeAgent({
      receipt: { signature: { alg: "ML-DSA-65", kid: "k1", sig: "" } },
    });
    const execute = vi.fn(async () => ({ temp: 22 }));
    const wrapped = asqavGuard(await makeAiTool(execute), { agent, toolName: "weather" });

    await expect(
      wrapped.execute!({ city: "Lisbon" }, {}),
    ).rejects.toBeInstanceOf(AsqavBlockedError);

    expect(execute).not.toHaveBeenCalled();
  });

  it("blocks an object signature without sig bytes before execute", async () => {
    const { agent } = makeFakeAgent({
      receipt: { signature: { alg: "ML-DSA-65", kid: "k1" } },
    });
    const execute = vi.fn(async () => ({ temp: 22 }));
    const wrapped = asqavGuard(await makeAiTool(execute), { agent, toolName: "weather" });

    await expect(
      wrapped.execute!({ city: "Lisbon" }, {}),
    ).rejects.toBeInstanceOf(AsqavBlockedError);

    expect(execute).not.toHaveBeenCalled();
  });

  it("NEGCTL: an allowed object signature runs the tool exactly once", async () => {
    const { agent } = makeFakeAgent({
      receipt: { signature: { alg: "ML-DSA-65", kid: "k1", sig: "c2lnLWJ5dGVz" } },
    });
    const execute = vi.fn(async () => ({ temp: 22 }));
    const wrapped = asqavGuard(await makeAiTool(execute), { agent, toolName: "weather" });

    await expect(
      wrapped.execute!({ city: "Lisbon" }, {}),
    ).resolves.toEqual({ temp: 22 });

    expect(execute).toHaveBeenCalledTimes(1);
  });
});
