import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Agent, _resetForTests, init } from "../src/index.js";

function jsonResponse(body: unknown, status = 200): Response {
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

describe("Agent.preflight", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("clears active agent with no blocking policy", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ revoked: false, suspended: false }))
      .mockResolvedValueOnce(jsonResponse([
        { is_active: true, action_pattern: "data:*", action: "log", name: "audit" },
      ]));

    const r = await makeAgent().preflight("data:read");

    expect(r.cleared).toBe(true);
    expect(r.agentActive).toBe(true);
    expect(r.policyAllowed).toBe(true);
    expect(r.explanation).toMatch(/Allowed/);
    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });

  it("blocks when agent is revoked", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ revoked: true, suspended: false }))
      .mockResolvedValueOnce(jsonResponse([]));

    const r = await makeAgent().preflight("data:read");

    expect(r.cleared).toBe(false);
    expect(r.agentActive).toBe(false);
    expect(r.reasons).toContain("agent is revoked");
    expect(r.explanation).toMatch(/revoked/);
  });

  it("blocks when a matching policy has action=block", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ revoked: false, suspended: false }))
      .mockResolvedValueOnce(jsonResponse([
        { is_active: true, action_pattern: "data:*", action: "block", name: "no-data-access" },
      ]));

    const r = await makeAgent().preflight("data:read");

    expect(r.cleared).toBe(false);
    expect(r.policyAllowed).toBe(false);
    expect(r.reasons.some((s) => s.includes("no-data-access"))).toBe(true);
    expect(r.explanation).toMatch(/policy/);
  });

  it("fail-open: status check error does not block on its own", async () => {
    let call = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      call += 1;
      const url = typeof input === "string" ? input : (input as URL).toString();
      if (url.includes("/status")) {
        return jsonResponse({ detail: "boom" }, 400);
      }
      return jsonResponse([]);
    });

    const r = await makeAgent().preflight("data:read");

    expect(r.cleared).toBe(true);
    expect(r.reasons.some((s) => s.includes("status check failed"))).toBe(true);
    expect(call).toBeGreaterThanOrEqual(2);
  });

  it("ignores inactive policies", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ revoked: false, suspended: false }))
      .mockResolvedValueOnce(jsonResponse([
        { is_active: false, action_pattern: "*", action: "block", name: "disabled-rule" },
      ]));

    const r = await makeAgent().preflight("data:read");
    expect(r.cleared).toBe(true);
  });
});
