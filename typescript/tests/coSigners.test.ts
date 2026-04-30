import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Agent, _resetForTests, init } from "../src/index.js";

function mockJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const AGENT_PAYLOAD = {
  agent_id: "agt_alice",
  name: "alice",
  public_key: "pk",
  key_id: "kid",
  algorithm: "ml-dsa-65",
  capabilities: [],
  created_at: "2026-01-01T00:00:00Z",
};

describe("agent.sign with co_signers", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("forwards coSigners as co_signers in request body", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(AGENT_PAYLOAD))
      .mockResolvedValueOnce(
        mockJsonResponse({
          signature: "sig-bytes",
          signature_id: "sig_1",
          action_id: "act_1",
          timestamp: "2026-01-01T00:00:00Z",
          verification_url: "https://asqav.com/verify/sig_1",
          required_co_signers: ["agt_bob", "agt_carol"],
          co_signatures: [],
          countersign_url: "/api/v1/agents/{agent_id}/countersign/sig_1",
        }),
      );

    const agent = await Agent.create({ name: "alice" });
    const sig = await agent.sign({
      actionType: "verify:peer-output",
      context: { task: "review" },
      coSigners: ["agt_bob", "agt_carol"],
    });

    const signInit = fetchSpy.mock.calls[1]![1] as RequestInit;
    const sentBody = JSON.parse(signInit.body as string);
    expect(sentBody.co_signers).toEqual(["agt_bob", "agt_carol"]);
    expect(sig.requiredCoSigners).toEqual(["agt_bob", "agt_carol"]);
    expect(sig.coSignatures).toEqual([]);
    expect(sig.countersignUrl).toBe("/api/v1/agents/{agent_id}/countersign/sig_1");
  });

  it("omits co_signers field when not provided", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(AGENT_PAYLOAD))
      .mockResolvedValueOnce(
        mockJsonResponse({
          signature: "sig-bytes",
          signature_id: "sig_2",
          action_id: "act_2",
          timestamp: "2026-01-01T00:00:00Z",
          verification_url: "https://asqav.com/verify/sig_2",
        }),
      );

    const agent = await Agent.create({ name: "alice" });
    await agent.sign({ actionType: "api:call", context: {} });

    const signInit = fetchSpy.mock.calls[1]![1] as RequestInit;
    const sentBody = JSON.parse(signInit.body as string);
    expect(sentBody.co_signers).toBeUndefined();
  });
});

describe("agent.countersign", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("posts to /agents/{id}/countersign/{signature_id} with empty body", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(AGENT_PAYLOAD))
      .mockResolvedValueOnce(
        mockJsonResponse({
          signature: "sig-bytes",
          signature_id: "sig_1",
          action_id: "act_1",
          timestamp: "2026-01-01T00:00:00Z",
          verification_url: "https://asqav.com/verify/sig_1",
          required_co_signers: ["agt_alice"],
          co_signatures: [
            { agent_id: "agt_alice", signature: "co_sig", signed_at: "2026-01-01T00:00:01Z" },
          ],
        }),
      );

    const agent = await Agent.create({ name: "alice" });
    const result = await agent.countersign("sig_1");

    const [url, callInit] = fetchSpy.mock.calls[1] as [string, RequestInit];
    expect(url).toBe("https://api.example.test/api/v1/agents/agt_alice/countersign/sig_1");
    expect(callInit.method).toBe("POST");
    expect(JSON.parse(callInit.body as string)).toEqual({});
    expect(result.coSignatures).toEqual([
      { agentId: "agt_alice", signature: "co_sig", signedAt: "2026-01-01T00:00:01Z" },
    ]);
  });
});
