import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  Agent,
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
  user_intent_verified: true,
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

const INTENT = {
  signature: "AAAA",
  public_key: "BBBB",
  algorithm: "ed25519",
  signed_message: "CCCC",
  signed_at: "2026-04-28T12:00:00Z",
};

async function makeAgent(): Promise<Agent> {
  const spy = vi
    .spyOn(globalThis, "fetch")
    .mockResolvedValueOnce(mockJsonResponse(AGENT_DATA));
  const agent = await Agent.create({ name: "test" });
  spy.mockRestore();
  return agent;
}

describe("Agent.sign userIntent passthrough", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("forwards userIntent on the wire (full-payload)", async () => {
    _setModeForTests("full-payload");
    const agent = await makeAgent();

    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(SIGN_RESPONSE));

    const resp = await agent.sign({
      actionType: "api:call",
      context: { x: 1 },
      userIntent: INTENT,
    });

    const body = JSON.parse(
      (fetchSpy.mock.calls[0]![1] as RequestInit).body as string,
    );
    expect(body.user_intent).toEqual(INTENT);
    expect(resp.userIntentVerified).toBe(true);
  });

  it("forwards userIntent in hash-only mode and preserves invariants", async () => {
    _setModeForTests("hash-only");
    const agent = await makeAgent();

    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(SIGN_RESPONSE));

    await agent.sign({
      actionType: "api:call",
      context: { x: 1 },
      userIntent: INTENT,
    });

    const body = JSON.parse(
      (fetchSpy.mock.calls[0]![1] as RequestInit).body as string,
    );
    expect(body.user_intent).toEqual(INTENT);
    expect(body.context).toBeUndefined();
    expect(body.hash).toBeDefined();
  });

  it("omits user_intent from the body when not passed", async () => {
    _setModeForTests("full-payload");
    const agent = await makeAgent();

    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(SIGN_RESPONSE));

    await agent.sign({ actionType: "api:call", context: { x: 1 } });

    const body = JSON.parse(
      (fetchSpy.mock.calls[0]![1] as RequestInit).body as string,
    );
    expect(body.user_intent).toBeUndefined();
  });
});
