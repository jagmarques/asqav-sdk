/** Default outcome: SDK sends the result digest unless told otherwise (787). */

import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { createHash } from "node:crypto";

import {
  Agent,
  _resetForTests,
  digestResultValue,
  init,
  toJsonSafeResult,
} from "../src/index.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function fakeAgent(): Agent {
  return Agent.attach({
    agent_id: "agt_outcome",
    name: "outcome",
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
    timestamp: "2026-05-25T00:00:00Z",
    verification_url: "https://verify",
  };
}

function readBody(spy: ReturnType<typeof vi.spyOn>): Record<string, unknown> {
  const init = spy.mock.calls[0][1] as RequestInit;
  return JSON.parse((init?.body as string) ?? "{}");
}

describe("default outcome (787)", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("digests and sends result by default", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    await fakeAgent().sign({ actionType: "api.call", result: "hello tool output" });
    const body = readBody(spy);
    expect(body.result_digest).toBe(
      `sha256:${createHash("sha256").update("hello tool output").digest("hex")}`,
    );
  });

  it("explicit resultDigest wins over result", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    const explicit = `sha256:${"b".repeat(64)}`;
    await fakeAgent().sign({ actionType: "api.call", result: "hello", resultDigest: explicit });
    expect(readBody(spy).result_digest).toBe(explicit);
  });

  it("no result sends no digest", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    await fakeAgent().sign({ actionType: "api.call" });
    expect(readBody(spy).result_digest).toBeUndefined();
  });

  it("digestResultValue is key-order stable", () => {
    expect(digestResultValue({ b: 1, a: 2 })).toBe(digestResultValue({ a: 2, b: 1 }));
    expect(digestResultValue("x")).toMatch(/^sha256:[0-9a-f]{64}$/);
  });

  it("attachResult links the origin", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    await fakeAgent().attachResult("act_origin_123", { status: "ok", rows: 3 });
    const body = readBody(spy);
    expect(body.receipt_type).toBe("protectmcp:observation:result_bound");
    expect(body.action_ref).toBe("act_origin_123");
    expect(body.result_digest).toBe(digestResultValue({ status: "ok", rows: 3 }));
  });

  it("BigInt results serialize instead of throwing (787)", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    await fakeAgent().sign({ actionType: "api.call", result: { total: 10n } });
    expect(readBody(spy).result_digest).toBe(
      digestResultValue(toJsonSafeResult({ total: 10n }) as Record<string, unknown>),
    );
  });
});
