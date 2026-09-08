/**
 * `invocationRef` is forwarded as `invocation_ref`, unfenced.
 * Producer-asserted like `findingRef` / `approvalRef`, valid on any receipt
 * type: pre-action decisions and post-action observations alike.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Agent, _resetForTests, init } from "../src/index.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function fakeAgent(): Agent {
  return Agent.attach({
    agent_id: "agt_hook",
    name: "hook",
    public_key: "pk",
    key_id: "kid",
    algorithm: "ml-dsa-65",
    capabilities: [],
    created_at: "2026-05-25T00:00:00Z",
  });
}

function okBody(): Record<string, unknown> {
  return {
    signature: "sig",
    signature_id: "sig_test",
    action_id: "act_test",
    timestamp: "2026-06-01T19:00:00Z",
    verification_url: "https://verify",
  };
}

function readBody(spy: ReturnType<typeof vi.spyOn>): Record<string, unknown> {
  const init = spy.mock.calls[0][1] as RequestInit;
  return JSON.parse((init?.body as string) ?? "{}");
}

beforeEach(() => {
  _resetForTests();
  init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("invocation_ref forwarding", () => {
  it("forwards invocationRef as invocation_ref on a non-risk receipt", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "tool:Write",
      context: { tool_input: {} },
      complianceMode: true,
      receiptType: "protectmcp:decision",
      invocationRef: "toolu_abc123",
    });
    expect(readBody(spy).invocation_ref).toBe("toolu_abc123");
  });

  it("omits invocation_ref when unset", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "tool:Write",
      context: { tool_input: {} },
      complianceMode: true,
      receiptType: "protectmcp:decision",
    });
    expect("invocation_ref" in readBody(spy)).toBe(false);
  });
});
