import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Agent, AuthenticationError, _resetForTests, govern } from "../src/index.js";

function mockAgentResponse(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    agent_id: "agt_govern_01",
    name: "my-agent",
    public_key: "pk_abc",
    key_id: "kid_abc",
    algorithm: "ml-dsa-65",
    capabilities: [],
    created_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function mockJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("govern()", () => {
  beforeEach(() => {
    _resetForTests();
    delete process.env.ASQAV_API_KEY;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("calls init and Agent.create with the right args, returns the agent", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(mockAgentResponse()));

    const agent = await govern({ apiKey: "sk_test", agentName: "my-agent" });

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/agents/create");
    expect(JSON.parse(init.body as string)).toMatchObject({
      name: "my-agent",
      algorithm: "ml-dsa-65",
    });
    expect(agent.agentId).toBe("agt_govern_01");
    expect(agent.name).toBe("my-agent");
  });

  it("retrieves an existing agent by id when agentId is supplied", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(mockAgentResponse()));

    const agent = await govern({ apiKey: "sk_test", agentId: "agt_govern_01" });

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/agents/agt_govern_01");
    expect(agent.agentId).toBe("agt_govern_01");
  });

  it("uses 'default-agent' when agentName is omitted", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(mockAgentResponse({ name: "default-agent" })));

    await govern({ apiKey: "sk_test" });

    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(init.body as string)).toMatchObject({ name: "default-agent" });
  });

  it("forwards capabilities and algorithm to Agent.create", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(mockAgentResponse({ algorithm: "ml-dsa-44" })));

    await govern({
      apiKey: "sk_test",
      agentName: "cap-agent",
      algorithm: "ml-dsa-44",
      capabilities: ["read", "write"],
    });

    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.algorithm).toBe("ml-dsa-44");
    expect(body.capabilities).toEqual(["read", "write"]);
  });

  it("throws AuthenticationError when no api key is supplied", async () => {
    await expect(govern({})).rejects.toThrow(AuthenticationError);
  });

  it("reads ASQAV_API_KEY from env when apiKey is omitted", async () => {
    process.env.ASQAV_API_KEY = "sk_from_env";
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(mockAgentResponse()));

    await expect(govern({ agentName: "env-agent" })).resolves.toBeDefined();
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });
});
