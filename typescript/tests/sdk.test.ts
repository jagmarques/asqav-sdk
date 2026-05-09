import { createHash } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  Agent,
  AuthenticationError,
  _resetForTests,
  init,
  verifySignature,
} from "../src/index.js";
import { canonicalJson } from "../src/jcs.js";

function mockJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("init", () => {
  beforeEach(() => {
    _resetForTests();
    delete process.env.ASQAV_API_KEY;
  });

  it("requires an api key", () => {
    expect(() => init({})).toThrow(AuthenticationError);
  });

  it("accepts api key via options", () => {
    expect(() => init({ apiKey: "sk_test" })).not.toThrow();
  });

  it("reads ASQAV_API_KEY from env", () => {
    process.env.ASQAV_API_KEY = "sk_env";
    expect(() => init({})).not.toThrow();
  });
});

describe("Agent.create", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("posts to /agents/create with the right body and headers", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockJsonResponse({
        agent_id: "agt_1",
        name: "test",
        public_key: "pk",
        key_id: "kid",
        algorithm: "ml-dsa-65",
        capabilities: ["read"],
        created_at: "2026-01-01T00:00:00Z",
      }),
    );

    const agent = await Agent.create({ name: "test", capabilities: ["read"] });

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [calledUrl, calledInit] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(calledUrl).toBe("https://api.example.test/api/v1/agents/create");
    expect(calledInit.method).toBe("POST");
    const headers = calledInit.headers as Record<string, string>;
    expect(headers["X-API-Key"]).toBe("sk_test");
    expect(headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(calledInit.body as string)).toEqual({
      name: "test",
      algorithm: "ml-dsa-65",
      capabilities: ["read"],
    });
    expect(agent.agentId).toBe("agt_1");
    expect(agent.name).toBe("test");
  });
});

describe("agent.sign", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("posts to /agents/{id}/sign with the right body", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        mockJsonResponse({
          agent_id: "agt_2",
          name: "signer",
          public_key: "pk",
          key_id: "kid",
          algorithm: "ml-dsa-65",
          capabilities: [],
          created_at: "2026-01-01T00:00:00Z",
        }),
      )
      .mockResolvedValueOnce(
        mockJsonResponse({
          signature: "sig-bytes",
          signature_id: "sig_1",
          action_id: "act_1",
          timestamp: "2026-01-01T00:00:00Z",
          verification_url: "https://asqav.com/verify/sig_1",
          algorithm: "ML-DSA-65",
          chain_hash: "abc",
        }),
      );

    const agent = await Agent.create({ name: "signer" });
    const sig = await agent.sign({
      actionType: "api:call",
      context: { model: "gpt-4" },
    });

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    const [signUrl, signInit] = fetchSpy.mock.calls[1] as [string, RequestInit];
    expect(signUrl).toBe("https://api.example.test/api/v1/agents/agt_2/sign");
    expect(signInit.method).toBe("POST");
    const expectedActionRef =
      "sha256:" +
      createHash("sha256")
        .update(canonicalJson({ action_type: "api:call", context: { model: "gpt-4" } }))
        .digest("hex");
    expect(JSON.parse(signInit.body as string)).toEqual({
      action_type: "api:call",
      context: { model: "gpt-4" },
      session_id: null,
      compliance_mode: true,
      action_ref: expectedActionRef,
    });

    expect(sig.signatureId).toBe("sig_1");
    expect(sig.chainHash).toBe("abc");
    expect(sig.verificationUrl).toBe("https://asqav.com/verify/sig_1");
  });
});

describe("verifySignature", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("hits GET /verify/{id}", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockJsonResponse({
        signature_id: "sig_x",
        agent_id: "agt_x",
        algorithm: "ML-DSA-65",
        signed_at: "2026-01-01T00:00:00Z",
        verified: true,
      }),
    );

    const result = await verifySignature("sig_x");
    const [url, callInit] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://api.example.test/api/v1/verify/sig_x");
    expect(callInit.method).toBe("GET");
    expect(result.verified).toBe(true);
    expect(result.algorithm).toBe("ML-DSA-65");
  });
});
