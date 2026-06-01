/**
 * IETF Compliance Receipts profile fields on agent.sign().
 *
 * Covers the wire-format mapping (camelCase TS -> snake_case JSON), the
 * client-side `action_ref` derivation under `complianceMode`, and the
 * receipt-namespace / reason validation that mirrors the cloud.
 */

import { createHash } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  Agent,
  AsqavError,
  DORA_INCIDENT_CLASS_NAMESPACE,
  HIPAA_INCIDENT_CLASS_NAMESPACE,
  INCIDENT_CLASS_NAMESPACE,
  RECEIPT_TYPE_NAMESPACE,
  SANDBOX_STATE_NAMESPACE,
  _resetForTests,
  init,
} from "../src/index.js";
import { canonicalJson } from "../src/jcs.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function fakeAgent(): Agent {
  return Agent.attach({
    agent_id: "agt_ietf",
    name: "ietf",
    public_key: "pk",
    key_id: "kid",
    algorithm: "ml-dsa-65",
    capabilities: [],
    created_at: "2026-05-04T00:00:00Z",
  });
}

describe("agent.sign IETF profile fields", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("forwards all new fields with snake_case wire keys", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "s",
        signature_id: "sig_1",
        action_id: "act_1",
        timestamp: "2026-05-04T00:00:00Z",
        verification_url: "https://example.test/v",
        compliance_mode: true,
        previousReceiptHash: "0".repeat(64),
        action_ref: "sha256:" + "a".repeat(64),
        receipt_type: "protectmcp:decision",
      }),
    );

    const agent = fakeAgent();
    const sig = await agent.sign({
      actionType: "api:call",
      context: { model: "gpt-4" },
      complianceMode: true,
      receiptType: "protectmcp:decision",
      sandboxState: "enabled",
      iterationId: "iter-7",
      riskClass: "high",
      incidentClass: "cybersecurity_related",
      issuerId: "Acme Legal Entity Ltd",
      payloadDigest: "sha256:" + "b".repeat(64),
      actionRef: "sha256:" + "c".repeat(64),
      reason: "permitted",
      policyDecision: "permit",
    });

    const [, calledInit] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(calledInit.body as string) as Record<string, unknown>;

    expect(body.compliance_mode).toBe(true);
    expect(body.action_ref).toBe("sha256:" + "c".repeat(64));
    expect(body.sandbox_state).toBe("enabled");
    expect(body.iteration_id).toBe("iter-7");
    expect(body.risk_class).toBe("high");
    expect(body.incident_class).toBe("cybersecurity_related");
    expect(body.issuer_id).toBe("Acme Legal Entity Ltd");
    expect(body.payload_digest).toBe("sha256:" + "b".repeat(64));
    expect(body.receipt_type).toBe("protectmcp:decision");
    expect(body.reason).toBe("permitted");
    expect(body.policy_decision).toBe("permit");

    expect(sig.complianceMode).toBe(true);
    expect(sig.previousReceiptHash).toBe("0".repeat(64));
    expect(sig.actionRef).toBe("sha256:" + "a".repeat(64));
    expect(sig.receiptType).toBe("protectmcp:decision");
  });

  it("computes action_ref client-side when omitted under complianceMode", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "s",
        signature_id: "sig_1",
        action_id: "act_1",
        timestamp: "t",
        verification_url: "u",
      }),
    );

    const agent = fakeAgent();
    const ctx = { model: "gpt-4", k: 1 };
    await agent.sign({
      actionType: "api:call",
      context: ctx,
      complianceMode: true,
    });

    const [, calledInit] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(calledInit.body as string) as Record<string, unknown>;

    const expected = "sha256:" + createHash("sha256")
      .update(canonicalJson({ action_type: "api:call", context: ctx }))
      .digest("hex");
    expect(body.action_ref).toBe(expected);
  });

  it("does not set compliance_mode when complianceMode is false", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "s",
        signature_id: "sig_1",
        action_id: "act_1",
        timestamp: "t",
        verification_url: "u",
      }),
    );

    const agent = fakeAgent();
    await agent.sign({ actionType: "api:call", context: {}, complianceMode: false });

    const [, calledInit] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(calledInit.body as string) as Record<string, unknown>;
    expect(body.compliance_mode).toBeUndefined();
    expect(body.action_ref).toBeUndefined();
  });

  it("rejects unknown receiptType client-side", async () => {
    const agent = fakeAgent();
    await expect(
      agent.sign({
        actionType: "api:call",
        context: {},
        // @ts-expect-error: testing runtime guard
        receiptType: "rogue:type",
      }),
    ).rejects.toThrow(AsqavError);
  });

  it("rejects deny without reason", async () => {
    const agent = fakeAgent();
    await expect(
      agent.sign({
        actionType: "api:call",
        context: {},
        policyDecision: "deny",
      }),
    ).rejects.toThrow(/missing_reason/);
  });

  it("accepts each value in the receipt-type namespace", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      jsonResponse({
        signature: "s",
        signature_id: "sig_1",
        action_id: "act_1",
        timestamp: "t",
        verification_url: "u",
      }),
    );
    const agent = fakeAgent();
    for (const t of RECEIPT_TYPE_NAMESPACE) {
      const opts: Parameters<typeof agent.sign>[0] = {
        actionType: "api:call",
        context: {},
        receiptType: t,
      };
      // Rule 9 lockstep: configuration_change carries config_manifest_digest.
      if (t === "protectmcp:lifecycle:configuration_change") {
        opts.configManifestDigest = `sha256:${"0".repeat(64)}`;
      }
      // Rule 9 lockstep: result_bound carries result_digest.
      if (t === "protectmcp:observation:result_bound") {
        opts.resultDigest = `sha256:${"0".repeat(64)}`;
      }
      // Risk-acceptance carries approver_id + acceptance_reason and the
      // no-policy opt-out (policy_decision='none').
      if (t === "protectmcp:lifecycle:risk_acceptance") {
        opts.approverId = "approver:alice@example.com";
        opts.acceptanceReason = "accepted";
        opts.policyDecision = "none";
      }
      // Code-authorship carries repo_ref + commit_sha and the no-policy
      // opt-out (policy_decision='none').
      if (t === "protectmcp:lifecycle:code_authorship") {
        opts.repoRef = "github.com/acme/repo";
        opts.commitSha = "c0ffee0000000000000000000000000000000000";
        opts.policyDecision = "none";
      }
      await expect(agent.sign(opts)).resolves.toBeDefined();
    }
  });

  it("DORA_INCIDENT_CLASS_NAMESPACE is the canonical six values from JC 2024-33 Annex II", () => {
    expect([...DORA_INCIDENT_CLASS_NAMESPACE].sort()).toEqual(
      [
        "cybersecurity_related",
        "external_event",
        "other",
        "payment_related",
        "process_failure",
        "system_failure",
      ],
    );
  });

  it("forwards every canonical incident_class without raising", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      jsonResponse({
        signature: "s",
        signature_id: "sig_1",
        action_id: "act_1",
        timestamp: "t",
        verification_url: "u",
      }),
    );
    const agent = fakeAgent();
    for (const v of DORA_INCIDENT_CLASS_NAMESPACE) {
      await expect(
        agent.sign({
          actionType: "api:call",
          context: {},
          complianceMode: true,
          incidentClass: v,
        }),
      ).resolves.toBeDefined();
    }
  });

  it("rejects off-vocabulary DORA tokens before any HTTP call", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const agent = fakeAgent();
    for (const token of [
      "cyberattack",
      "payment_fraud",
      "malicious_actions",
      "human_error",
      "unknown",
    ]) {
      await expect(
        agent.sign({
          actionType: "api:call",
          context: {},
          complianceMode: true,
          incidentClass: token,
        }),
      ).rejects.toThrow(/invalid_incident_class/);
    }
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("rejects an out-of-vocabulary incident_class before any HTTP call", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const agent = fakeAgent();
    await expect(
      agent.sign({
        actionType: "api:call",
        context: {},
        complianceMode: true,
        incidentClass: "ICT-INC-001",
      }),
    ).rejects.toThrow(/invalid_incident_class/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  // === HIPAA incident_class vocabulary (GAP-E13) ===

  it("accepts the HIPAA security incident token", async () => {
    expect(HIPAA_INCIDENT_CLASS_NAMESPACE).toContain("hipaa_security_incident");
    expect(INCIDENT_CLASS_NAMESPACE).toContain("hipaa_security_incident");

    let capturedBody: Record<string, unknown> | null = null;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
      capturedBody = JSON.parse((init?.body as string) ?? "{}");
      return jsonResponse({
        signature: "sig",
        signature_id: "sigid",
        action_id: "act",
        timestamp: "2026-05-04T00:00:00Z",
        verification_url: "https://example.test",
      });
    });
    const agent = fakeAgent();
    await agent.sign({
      actionType: "api:call",
      context: {},
      complianceMode: true,
      incidentClass: "hipaa_security_incident",
    });
    expect((capturedBody as unknown as Record<string, unknown>).incident_class).toBe(
      "hipaa_security_incident",
    );
  });

  it("accepts an array mixing DORA and HIPAA tokens for cross-regime classification", async () => {
    let capturedBody: Record<string, unknown> | null = null;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
      capturedBody = JSON.parse((init?.body as string) ?? "{}");
      return jsonResponse({
        signature: "sig",
        signature_id: "sigid",
        action_id: "act",
        timestamp: "2026-05-04T00:00:00Z",
        verification_url: "https://example.test",
      });
    });
    const agent = fakeAgent();
    await agent.sign({
      actionType: "api:call",
      context: {},
      complianceMode: true,
      incidentClass: ["cybersecurity_related", "hipaa_security_incident"],
    });
    expect(
      (capturedBody as unknown as Record<string, unknown>).incident_class,
    ).toEqual(["cybersecurity_related", "hipaa_security_incident"]);
  });

  // === Sandbox state enum (GAP-E3) ===

  it("SANDBOX_STATE_NAMESPACE matches the upstream enum", () => {
    expect([...SANDBOX_STATE_NAMESPACE].sort()).toEqual(
      ["disabled", "enabled", "unavailable"],
    );
  });

  it("rejects an out-of-vocabulary sandbox_state before any HTTP call", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const agent = fakeAgent();
    await expect(
      agent.sign({
        actionType: "api:call",
        context: {},
        complianceMode: true,
        // @ts-expect-error -- intentional: testing runtime guard with a string outside the enum.
        sandboxState: "sandboxed",
      }),
    ).rejects.toThrow(/invalid_sandbox_state/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
