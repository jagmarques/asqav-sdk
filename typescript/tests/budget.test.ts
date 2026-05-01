import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  Agent,
  BudgetTracker,
  _resetForTests,
  init,
} from "../src/index.js";

function mockJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function makeAgent(): Agent {
  return Agent.attach({
    agent_id: "agt_1",
    name: "test",
    public_key: "pk",
    key_id: "kid",
    algorithm: "ml-dsa-65",
    capabilities: [],
    created_at: "2026-01-01T00:00:00Z",
  });
}

describe("BudgetTracker", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("rejects negative or non-finite limit", () => {
    expect(() => new BudgetTracker({ agent: makeAgent(), limit: -1 })).toThrow();
    expect(() => new BudgetTracker({ agent: makeAgent(), limit: Number.POSITIVE_INFINITY })).toThrow();
    expect(() => new BudgetTracker({ agent: makeAgent(), limit: Number.NaN })).toThrow();
  });

  it("check() allows costs within remaining budget", () => {
    const b = new BudgetTracker({ agent: makeAgent(), limit: 10 });
    const r = b.check(3);
    expect(r.allowed).toBe(true);
    expect(r.remaining).toBe(7);
    expect(r.reason).toBeUndefined();
  });

  it("check() blocks costs that exceed remaining budget with reason budget_exhausted", () => {
    const b = new BudgetTracker({ agent: makeAgent(), limit: 10 });
    const r = b.check(11);
    expect(r.allowed).toBe(false);
    expect(r.reason).toBe("budget_exhausted");
  });

  it("check() fail-closes on negative / NaN / infinite costs", () => {
    const b = new BudgetTracker({ agent: makeAgent(), limit: 10 });
    expect(b.check(-1).reason).toBe("invalid_cost");
    expect(b.check(Number.NaN).reason).toBe("invalid_cost");
    expect(b.check(Number.POSITIVE_INFINITY).reason).toBe("invalid_cost");
  });

  it("record() signs a budget:* action and updates cumulative spend", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockJsonResponse({
        signature: "sig_data",
        signature_id: "sig_1",
        action_id: "act_1",
        timestamp: "2026-05-01T00:00:00Z",
        verification_url: "https://x/y",
      }),
    );
    const b = new BudgetTracker({ agent: makeAgent(), limit: 10, currency: "USD" });

    const sig = await b.record("api:openai", 0.42, { model: "gpt-4" });

    expect(sig.signatureId).toBe("sig_1");
    const [, calledInit] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(calledInit.body as string);
    expect(body.action_type).toBe("budget:api:openai");
    expect(body.context.cost).toBe(0.42);
    expect(body.context.currency).toBe("USD");
    expect(body.context.cumulative_spend).toBe(0.42);
    expect(body.context.limit).toBe(10);
    expect(body.context.model).toBe("gpt-4");

    expect(b.status().spend).toBeCloseTo(0.42);
    expect(b.status().records).toBe(1);
  });

  it("record() refuses negative / non-finite costs", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(mockJsonResponse({}));
    const b = new BudgetTracker({ agent: makeAgent(), limit: 10 });
    await expect(b.record("api:openai", -1)).rejects.toThrow();
    await expect(b.record("api:openai", Number.NaN)).rejects.toThrow();
  });
});
