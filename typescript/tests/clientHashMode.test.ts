import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  Agent,
  _getModeForTests,
  _resetForTests,
  _setModeForTests,
  init,
} from "../src/index.js";

function mockJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const SIGN_RESPONSE = {
  signature: "sig-bytes",
  signature_id: "sig_001",
  action_id: "act_001",
  timestamp: "2026-04-28T12:00:00Z",
  verification_url: "https://verify.asqav.com/sig_001",
  algorithm: "ml-dsa-65",
};

const AGENT_DATA = {
  agent_id: "agt_001",
  name: "test",
  public_key: "pk",
  key_id: "kid",
  algorithm: "ml-dsa-65",
  capabilities: [],
  created_at: "2026-01-01T00:00:00Z",
};

async function makeAgent(): Promise<Agent> {
  const fetchSpy = vi
    .spyOn(globalThis, "fetch")
    .mockResolvedValueOnce(mockJsonResponse(AGENT_DATA));
  const agent = await Agent.create({ name: "test" });
  fetchSpy.mockRestore();
  return agent;
}

describe("Agent.sign body shape across modes", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("full-payload mode sends context, no hash", async () => {
    _setModeForTests("full-payload");
    const agent = await makeAgent();

    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(SIGN_RESPONSE));

    await agent.sign({
      actionType: "api:call",
      context: { prompt: "hi", user_id: "u_001" },
    });

    const calledInit = fetchSpy.mock.calls[0]![1] as RequestInit;
    const body = JSON.parse(calledInit.body as string);
    expect(body.context).toEqual({ prompt: "hi", user_id: "u_001" });
    expect(body.hash).toBeUndefined();
    expect(body.action_type).toBe("api:call");
  });

  it("hash-only mode sends hash, no context", async () => {
    _setModeForTests("hash-only");
    const agent = await makeAgent();

    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(SIGN_RESPONSE));

    await agent.sign({
      actionType: "api:call",
      context: { prompt: "hi", user_id: "u_001" },
    });

    const body = JSON.parse(
      (fetchSpy.mock.calls[0]![1] as RequestInit).body as string,
    );
    expect(body.context).toBeUndefined();
    expect(typeof body.hash).toBe("string");
    expect(body.hash.startsWith("sha256:")).toBe(true);
    expect(body.hash.length).toBe("sha256:".length + 64);
    expect(body.hash_algo).toBe("sha256");
    expect(body.action_type).toBe("api:call");
  });

  it("hash-only mode does not leak context fields into metadata", async () => {
    _setModeForTests("hash-only");
    const agent = await makeAgent();

    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(SIGN_RESPONSE));

    await agent.sign({
      actionType: "api:call",
      context: { prompt: "secret", user_id: "u_001", ip: "1.2.3.4", request_id: "req_xyz" },
    });

    const body = JSON.parse(
      (fetchSpy.mock.calls[0]![1] as RequestInit).body as string,
    );
    const metadata = body.metadata as Record<string, unknown>;
    for (const k of ["user_id", "ip", "prompt", "request_id"]) {
      expect(metadata[k]).toBeUndefined();
    }
    expect(metadata.agent_id).toBe("agt_001");
    expect(metadata.action_type).toBe("api:call");
  });

  it("hash-only surfaces _model_name / _tool_name when caller opts in", async () => {
    _setModeForTests("hash-only");
    const agent = await makeAgent();

    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(SIGN_RESPONSE));

    await agent.sign({
      actionType: "api:call",
      context: { prompt: "hi", _model_name: "gpt-4", _tool_name: "search" },
    });

    const body = JSON.parse(
      (fetchSpy.mock.calls[0]![1] as RequestInit).body as string,
    );
    expect(body.metadata.model_name).toBe("gpt-4");
    expect(body.metadata.tool_name).toBe("search");
  });

  it("hash-only with org salt produces a different hash than without", async () => {
    const agent = await makeAgent();

    _setModeForTests("hash-only");
    let fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(SIGN_RESPONSE));
    await agent.sign({ actionType: "api:call", context: { k: 3 } });
    const plain = JSON.parse(
      (fetchSpy.mock.calls[0]![1] as RequestInit).body as string,
    ).hash;
    fetchSpy.mockRestore();

    _setModeForTests("hash-only", new Uint8Array(32).fill(1));
    fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(SIGN_RESPONSE));
    await agent.sign({ actionType: "api:call", context: { k: 3 } });
    const salted = JSON.parse(
      (fetchSpy.mock.calls[0]![1] as RequestInit).body as string,
    ).hash;

    expect(plain).not.toBe(salted);
    expect(plain.startsWith("sha256:")).toBe(true);
    expect(salted.startsWith("sha256:")).toBe(true);
  });
});

describe("init() resolves mode from baseUrl", () => {
  beforeEach(() => {
    _resetForTests();
    delete process.env.ASQAV_MODE;
  });

  it("auto -> hash-only for *.asqav.com", () => {
    init({ apiKey: "sk_test", baseUrl: "https://api.asqav.com/api/v1" });
    expect(_getModeForTests()).toBe("hash-only");
  });

  it("auto -> full-payload for localhost", () => {
    init({ apiKey: "sk_test", baseUrl: "http://localhost:8000" });
    expect(_getModeForTests()).toBe("full-payload");
  });

  it("explicit override wins over URL", () => {
    init({
      apiKey: "sk_test",
      baseUrl: "https://api.asqav.com/api/v1",
      mode: "full-payload",
    });
    expect(_getModeForTests()).toBe("full-payload");
  });
});
