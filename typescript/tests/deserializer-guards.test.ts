/**
 * Required-field deserializer guards throw AsqavResponseError, not a
 * silent `undefined`, when a cloud response omits a field the SDK
 * requires. Companion to verify-response-fields.test.ts, which pins
 * the positive case. This file pins the negative case: a sign
 * response missing action_id or signature_id must throw one typed,
 * catchable AsqavResponseError naming the field (rule 3.9).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Agent, AsqavResponseError, _resetForTests, init } from "../src/index.js";

function jsonResponse(body: unknown, status = 200): Response {
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

function fullSignPayload(): Record<string, unknown> {
  return {
    signature: "sig-bytes",
    signature_id: "sig_1",
    action_id: "act_1",
    timestamp: "2026-01-01T00:00:00Z",
    verification_url: "https://asqav.com/verify/sig_1",
  };
}

describe("agent.sign response missing a required field", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it.each(["action_id", "signature_id"])(
    "throws AsqavResponseError naming '%s', not a silent undefined",
    async (missingField) => {
      const signPayload = fullSignPayload();
      delete signPayload[missingField];

      vi.spyOn(globalThis, "fetch")
        .mockResolvedValueOnce(jsonResponse(AGENT_PAYLOAD))
        .mockResolvedValueOnce(jsonResponse(signPayload));

      const agent = await Agent.create({ name: "alice" });

      let caught: unknown;
      try {
        await agent.sign({ actionType: "api:call", context: {} });
      } catch (error) {
        caught = error;
      }

      expect(caught).toBeInstanceOf(AsqavResponseError);
      expect((caught as Error).message).toContain(missingField);
    },
  );
});
