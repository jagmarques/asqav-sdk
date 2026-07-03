// Prod nests actionRef / previousReceiptHash under `payload`.
// These cover the payload fallback and top-level precedence.

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
    agent_id: "agt_payload",
    name: "payload",
    public_key: "pk",
    key_id: "kid",
    algorithm: "ml-dsa-65",
    capabilities: [],
    created_at: "2026-05-04T00:00:00Z",
  });
}

// Mirrors the real prod wire: the top level omits these keys and the
// values arrive only under `payload`.
const ACTION_REF =
  "sha256:11894b66385ecd4a2f1be048e496435b91c8eb02ad09105c7be536bd87f9a48c";
const PREV_HASH = "0".repeat(64);

describe("sign response payload field fallback", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("resolves actionRef and previousReceiptHash from the payload", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-04T00:00:00Z",
        verification_url: "https://verify",
        payload: {
          action_ref: ACTION_REF,
          previousReceiptHash: PREV_HASH,
        },
      }),
    );

    const resp = await fakeAgent().sign({ actionType: "t.test" });

    expect(resp.actionRef).toBe(ACTION_REF);
    expect(resp.previousReceiptHash).toBe(PREV_HASH);
    // the payload object itself stays intact on the response
    expect((resp.payload as Record<string, unknown>).action_ref).toBe(
      ACTION_REF,
    );
  });

  it("accepts snake_case previous_receipt_hash inside the payload", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-04T00:00:00Z",
        verification_url: "https://verify",
        payload: { previous_receipt_hash: PREV_HASH },
      }),
    );

    const resp = await fakeAgent().sign({ actionType: "t.test" });

    expect(resp.previousReceiptHash).toBe(PREV_HASH);
  });

  it("keeps top-level values winning over the payload copy", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-04T00:00:00Z",
        verification_url: "https://verify",
        action_ref: "sha256:toplevel",
        previousReceiptHash: "1".repeat(64),
        payload: {
          action_ref: ACTION_REF,
          previousReceiptHash: PREV_HASH,
        },
      }),
    );

    const resp = await fakeAgent().sign({ actionType: "t.test" });

    expect(resp.actionRef).toBe("sha256:toplevel");
    expect(resp.previousReceiptHash).toBe("1".repeat(64));
  });
});
