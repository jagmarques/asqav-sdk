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

  it("fail-closed: status check error blocks and marks checks incomplete", async () => {
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

    expect(r.cleared).toBe(false);
    expect(r.checksComplete).toBe(false);
    expect(r.reasons.some((s) => s.includes("status check failed"))).toBe(true);
    expect(r.explanation).toMatch(/could not verify/);
    expect(call).toBeGreaterThanOrEqual(2);
  });

  it("fail-closed: policy fetch error blocks and marks checks incomplete", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as URL).toString();
      if (url.includes("/policies")) {
        return jsonResponse({ detail: "boom" }, 500);
      }
      return jsonResponse({ revoked: false, suspended: false });
    });

    const r = await makeAgent().preflight("data:read");

    expect(r.cleared).toBe(false);
    expect(r.checksComplete).toBe(false);
    expect(r.reasons.some((s) => s.includes("policy check failed"))).toBe(true);
  });

  it("fail-closed: a non-list /policies response does not silently clear", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ revoked: false, suspended: false }))
      .mockResolvedValueOnce(jsonResponse({ policies: [] }));

    const r = await makeAgent().preflight("data:read");

    expect(r.cleared).toBe(false);
    expect(r.checksComplete).toBe(false);
    expect(r.policyAllowed).toBe(true);
    expect(r.reasons.some((s) => s.includes("unexpected response"))).toBe(true);
  });

  it.each([
    ["a list", ["x"]],
    ["a string", "revoked"],
    ["a number", 5],
    ["an empty list", []],
  ])("fail-closed: a non-object /status response (%s) does not silently clear", async (_label, body) => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(body))
      .mockResolvedValueOnce(jsonResponse([]));

    const r = await makeAgent().preflight("data:read");

    expect(r.cleared).toBe(false);
    expect(r.checksComplete).toBe(false);
    expect(r.agentActive).toBe(true);
    expect(r.reasons.some((s) => s.includes("unexpected response"))).toBe(true);
    expect(r.explanation).toMatch(/could not verify/);
  });

  it("clears when both checks complete with no block (checksComplete true)", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ revoked: false, suspended: false }))
      .mockResolvedValueOnce(jsonResponse([]));

    const r = await makeAgent().preflight("data:read");

    expect(r.cleared).toBe(true);
    expect(r.checksComplete).toBe(true);
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
