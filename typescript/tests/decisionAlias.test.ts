/**
 * Spec-shape `decision` alias on the TS SDK.
 *
 * Covers the DECISION_MAP, the `mapPolicyDecisionToDecision` helper, and
 * end-to-end propagation of the `decision` token on a fake `agent.sign`
 * call (request body MUST carry `decision` when complianceMode is on,
 * response MUST surface both `policyDecision` and `decision`).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  Agent,
  DECISION_MAP,
  _resetForTests,
  init,
  mapPolicyDecisionToDecision,
} from "../src/index.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function fakeAgent(): Agent {
  return Agent.attach({
    agent_id: "agt_decision",
    name: "decision",
    public_key: "pk",
    key_id: "kid",
    algorithm: "ml-dsa-65",
    capabilities: [],
    created_at: "2026-05-04T00:00:00Z",
  });
}

describe("DECISION_MAP", () => {
  it("covers the full alias vocabulary", () => {
    expect(DECISION_MAP.permit).toBe("allow");
    expect(DECISION_MAP.deny).toBe("deny");
    expect(DECISION_MAP.rate_limit).toBe("rate_limit");
  });

  it("is idempotent for already-normalised input", () => {
    expect(DECISION_MAP.allow).toBe("allow");
    expect(mapPolicyDecisionToDecision("allow")).toBe("allow");
  });

  it("falls back to deny on unknown / null input", () => {
    expect(mapPolicyDecisionToDecision("yolo")).toBe("deny");
    expect(mapPolicyDecisionToDecision(null)).toBe("deny");
    expect(mapPolicyDecisionToDecision(undefined)).toBe("deny");
    expect(mapPolicyDecisionToDecision("")).toBe("deny");
  });
});

describe("agent.sign decision wire field", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("emits decision in the request body under complianceMode", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-04T00:00:00Z",
        verification_url: "https://verify",
        compliance_mode: true,
        policy_decision: "permit",
        decision: "allow",
      }),
    );

    const agent = fakeAgent();
    await agent.sign({
      actionType: "t.test",
      complianceMode: true,
      policyDecision: "permit",
    });

    const init = fetchSpy.mock.calls[0][1];
    const body = JSON.parse((init?.body as string) ?? "{}");
    expect(body.policy_decision).toBe("permit");
    expect(body.decision).toBe("allow");
  });

  it("does not emit decision when complianceMode is off", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-04T00:00:00Z",
        verification_url: "https://verify",
      }),
    );

    const agent = fakeAgent();
    await agent.sign({
      actionType: "t.test",
      complianceMode: false,
      policyDecision: "permit",
    });

    const init = fetchSpy.mock.calls[0][1];
    const body = JSON.parse((init?.body as string) ?? "{}");
    expect(body.policy_decision).toBe("permit");
    expect(body.decision).toBeUndefined();
  });

  it("surfaces decision on the response when cloud emits it", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-04T00:00:00Z",
        verification_url: "https://verify",
        compliance_mode: true,
        policy_decision: "deny",
        decision: "deny",
      }),
    );

    const agent = fakeAgent();
    const resp = await agent.sign({
      actionType: "t.test",
      complianceMode: true,
      policyDecision: "deny",
      reason: "policy_violation",
    });

    expect(resp.policyDecision).toBe("deny");
    expect(resp.decision).toBe("deny");
  });

  it("client-side maps decision when older cloud omits it", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-04T00:00:00Z",
        verification_url: "https://verify",
        compliance_mode: true,
        policy_decision: "permit",
        // older cloud: no `decision` field
      }),
    );

    const agent = fakeAgent();
    const resp = await agent.sign({
      actionType: "t.test",
      complianceMode: true,
      policyDecision: "permit",
    });

    expect(resp.policyDecision).toBe("permit");
    expect(resp.decision).toBe("allow");
  });

  it("decision undefined on non-compliance receipts", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-04T00:00:00Z",
        verification_url: "https://verify",
      }),
    );

    const agent = fakeAgent();
    const resp = await agent.sign({ actionType: "t.test" });

    expect(resp.decision).toBeUndefined();
  });
});
