import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  Agent,
  _resetForTests,
  clearHooks,
  init,
  registerAfter,
  registerBefore,
  type SignatureResponse,
} from "../src/index.js";

function mockJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function agentPayload(agentId = "agt_h"): unknown {
  return {
    agent_id: agentId,
    name: "hook-agent",
    public_key: "pk",
    key_id: "kid",
    algorithm: "ml-dsa-65",
    capabilities: [],
    created_at: "2026-01-01T00:00:00Z",
  };
}

function signPayload(signatureId = "sig_h"): unknown {
  return {
    signature: "sig-bytes",
    signature_id: signatureId,
    action_id: "act_h",
    timestamp: "2026-01-01T00:00:00Z",
    verification_url: `https://asqav.com/verify/${signatureId}`,
    algorithm: "ML-DSA-65",
    chain_hash: "hash",
  };
}

async function makeAgent(): Promise<Agent> {
  const fetchSpy = vi
    .spyOn(globalThis, "fetch")
    .mockResolvedValueOnce(mockJsonResponse(agentPayload()));
  const agent = await Agent.create({ name: "hook-agent" });
  fetchSpy.mockRestore();
  return agent;
}

describe("hooks", () => {
  beforeEach(() => {
    _resetForTests();
    clearHooks();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    clearHooks();
    vi.restoreAllMocks();
  });

  it("registerBefore mutates context reaching the fetch body", async () => {
    const agent = await makeAgent();
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(signPayload()));

    registerBefore("api:call", (_actionType, ctx) => ({
      ...ctx,
      injected: "yes",
    }));

    await agent.sign({ actionType: "api:call", context: { model: "gpt-4" } });

    const [, callInit] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(callInit.body as string) as {
      context: Record<string, unknown>;
    };
    expect(body.context).toEqual({ model: "gpt-4", injected: "yes" });
  });

  it("registerAfter fires with the response", async () => {
    const agent = await makeAgent();
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      mockJsonResponse(signPayload("sig_after")),
    );

    const seen: SignatureResponse[] = [];
    registerAfter("*", (resp) => {
      seen.push(resp);
    });

    const result = await agent.sign({ actionType: "api:call" });
    expect(seen).toHaveLength(1);
    expect(seen[0].signatureId).toBe("sig_after");
    expect(result.signatureId).toBe("sig_after");
  });

  it("failing hook does not crash sign", async () => {
    const agent = await makeAgent();
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      mockJsonResponse(signPayload()),
    );
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    registerBefore("*", () => {
      throw new Error("boom-before");
    });
    registerAfter("*", () => {
      throw new Error("boom-after");
    });

    const sig = await agent.sign({ actionType: "api:call" });
    expect(sig.signatureId).toBe("sig_h");
    expect(warnSpy).toHaveBeenCalled();
  });

  it("pattern strands:* matches strands:model_call but not langchain:tool_call", async () => {
    const agent = await makeAgent();
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(signPayload("sig_strands")))
      .mockResolvedValueOnce(mockJsonResponse(signPayload("sig_lc")));

    const calls: string[] = [];
    registerAfter("strands:*", (resp) => {
      calls.push(resp.signatureId);
    });

    await agent.sign({ actionType: "strands:model_call" });
    await agent.sign({ actionType: "langchain:tool_call" });

    expect(calls).toEqual(["sig_strands"]);
  });

  it("clearHooks empties registry", async () => {
    const agent = await makeAgent();
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      mockJsonResponse(signPayload()),
    );

    const calls: string[] = [];
    registerAfter("*", () => {
      calls.push("hit");
    });
    clearHooks();

    await agent.sign({ actionType: "api:call" });
    expect(calls).toEqual([]);
  });
});
