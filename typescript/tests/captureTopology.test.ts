/**
 * SDK parity for `capture_topology` + `observation` decision vocabulary.
 *
 * Mirrors the Python `tests/test_client_capture_topology.py` cases: the
 * `none -> observation` DECISION_MAP entry, the closed Literal of five
 * `captureTopology` values, and end-to-end wire-body assertions.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  Agent,
  CAPTURE_TOPOLOGY_NAMESPACE,
  DECISION_MAP,
  RECEIPT_TYPE_NAMESPACE,
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
    agent_id: "agt_capture",
    name: "capture",
    public_key: "pk",
    key_id: "kid",
    algorithm: "ml-dsa-65",
    capabilities: [],
    created_at: "2026-05-15T00:00:00Z",
  });
}

describe("DECISION_MAP observation entry", () => {
  it("maps `none` to `observation` for lifecycle opt-out receipts", () => {
    expect(DECISION_MAP.none).toBe("observation");
    expect(mapPolicyDecisionToDecision("none")).toBe("observation");
  });

  it("keeps the existing alias rows intact", () => {
    expect(DECISION_MAP.permit).toBe("allow");
    expect(DECISION_MAP.allow).toBe("allow");
    expect(DECISION_MAP.deny).toBe("deny");
    expect(DECISION_MAP.rate_limit).toBe("rate_limit");
  });
});

describe("CAPTURE_TOPOLOGY_NAMESPACE", () => {
  it("matches the cloud SignRequest Literal exactly", () => {
    expect([...CAPTURE_TOPOLOGY_NAMESPACE].sort()).toEqual(
      [
        "browser_extension",
        "ebpf_observer",
        "in_process_sdk",
        "mcp_proxy",
        "network_proxy",
        "passive_telemetry",
      ],
    );
  });
});

describe("RECEIPT_TYPE_NAMESPACE", () => {
  it("includes the observation token paired with passive_telemetry", () => {
    expect([...RECEIPT_TYPE_NAMESPACE].sort()).toEqual(
      [
        "protectmcp:acknowledgment",
        "protectmcp:decision",
        "protectmcp:lifecycle",
        "protectmcp:lifecycle:configuration_change",
        "protectmcp:lifecycle:risk_acceptance",
        "protectmcp:observation",
        // NSA CSI U/OO/6030316-26 alignment (cloud 0.5.0): result-bound
        // observation receipts use the `:result_bound` suffix.
        "protectmcp:observation:result_bound",
        "protectmcp:restraint",
      ],
    );
  });
});

describe("agent.sign capture_topology + observation wire fields", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("emits decision=observation under policyDecision=none + lifecycle receipt", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-15T00:00:00Z",
        verification_url: "https://verify",
        compliance_mode: true,
        policy_decision: "none",
        decision: "observation",
        receipt_type: "protectmcp:lifecycle",
      }),
    );

    const agent = fakeAgent();
    const resp = await agent.sign({
      actionType: "t.lifecycle",
      complianceMode: true,
      receiptType: "protectmcp:lifecycle",
      policyDecision: "none",
    });

    const init = fetchSpy.mock.calls[0][1];
    const body = JSON.parse((init?.body as string) ?? "{}");
    expect(body.policy_decision).toBe("none");
    expect(body.decision).toBe("observation");
    expect(body.receipt_type).toBe("protectmcp:lifecycle");
    expect(resp.decision).toBe("observation");
  });

  it("client-side maps `none` to `observation` when older cloud omits decision", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-15T00:00:00Z",
        verification_url: "https://verify",
        compliance_mode: true,
        policy_decision: "none",
        receipt_type: "protectmcp:lifecycle",
      }),
    );

    const agent = fakeAgent();
    const resp = await agent.sign({
      actionType: "t.lifecycle",
      complianceMode: true,
      receiptType: "protectmcp:lifecycle",
      policyDecision: "none",
    });

    expect(resp.decision).toBe("observation");
  });

  it("policyDecision=none does not require a reason", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-15T00:00:00Z",
        verification_url: "https://verify",
      }),
    );

    const agent = fakeAgent();
    await expect(
      agent.sign({
        actionType: "t.lifecycle",
        complianceMode: true,
        receiptType: "protectmcp:lifecycle",
        policyDecision: "none",
      }),
    ).resolves.toBeDefined();
  });

  it.each([
    "in_process_sdk",
    "network_proxy",
    "browser_extension",
    "ebpf_observer",
    "mcp_proxy",
    "passive_telemetry",
  ] as const)("passes captureTopology=%s through on the outgoing body", async (value) => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-15T00:00:00Z",
        verification_url: "https://verify",
      }),
    );

    const agent = fakeAgent();
    await agent.sign({
      actionType: "t.test",
      complianceMode: true,
      captureTopology: value,
      receiptType: value === "passive_telemetry"
        ? "protectmcp:observation"
        : "protectmcp:decision",
    });

    const init = fetchSpy.mock.calls[0][1];
    const body = JSON.parse((init?.body as string) ?? "{}");
    expect(body.capture_topology).toBe(value);
  });

  it("rejects an out-of-vocabulary captureTopology before the HTTP call", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    const agent = fakeAgent();
    await expect(
      agent.sign({
        actionType: "t.test",
        complianceMode: true,
        // @ts-expect-error - intentionally outside the closed Literal
        captureTopology: "rootkit",
      }),
    ).rejects.toThrow(/invalid_capture_topology/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("omits capture_topology when not supplied", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-15T00:00:00Z",
        verification_url: "https://verify",
      }),
    );

    const agent = fakeAgent();
    await agent.sign({ actionType: "t.test", complianceMode: true });

    const init = fetchSpy.mock.calls[0][1];
    const body = JSON.parse((init?.body as string) ?? "{}");
    expect(body.capture_topology).toBeUndefined();
  });

  it("drops capture_topology outside compliance mode", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-15T00:00:00Z",
        verification_url: "https://verify",
      }),
    );

    const agent = fakeAgent();
    await agent.sign({
      actionType: "t.test",
      complianceMode: false,
      captureTopology: "mcp_proxy",
    });

    const init = fetchSpy.mock.calls[0][1];
    const body = JSON.parse((init?.body as string) ?? "{}");
    expect(body.capture_topology).toBeUndefined();
  });

  it.each([
    "bogus",
    "in-process-sdk",
    "PASSIVE_TELEMETRY",
    "ROOTKIT",
  ] as const)("rejects out-of-vocabulary captureTopology=%s", async (bad) => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const agent = fakeAgent();
    await expect(
      agent.sign({
        actionType: "t.test",
        complianceMode: true,
        // @ts-expect-error - intentionally outside the closed Literal
        captureTopology: bad,
      }),
    ).rejects.toThrow(/invalid_capture_topology/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it.each([
    "protectmcp:decision",
    "protectmcp:restraint",
    "protectmcp:lifecycle",
    "protectmcp:lifecycle:configuration_change",
    "protectmcp:acknowledgment",
    "protectmcp:observation",
  ] as const)("accepts receiptType=%s on the outgoing body", async (value) => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-15T00:00:00Z",
        verification_url: "https://verify",
      }),
    );
    const agent = fakeAgent();
    const opts: Parameters<typeof agent.sign>[0] = {
      actionType: "t.test",
      complianceMode: true,
      receiptType: value,
      policyDecision: value.startsWith("protectmcp:lifecycle") ? "none" : "permit",
    };
    // Rule 9 (NSA CSI U/OO/6030316-26 alignment): configuration_change
    // receipts MUST carry config_manifest_digest.
    if (value === "protectmcp:lifecycle:configuration_change") {
      opts.configManifestDigest = `sha256:${"0".repeat(64)}`;
    }
    await agent.sign(opts);
    const init = fetchSpy.mock.calls[0][1];
    const body = JSON.parse((init?.body as string) ?? "{}");
    expect(body.receipt_type).toBe(value);
  });

  it.each([
    "protectmcp:bogus",
    "protectmcp:Observation",
    "decision",
    "observation",
  ] as const)("rejects out-of-vocabulary receiptType=%s", async (bad) => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const agent = fakeAgent();
    await expect(
      agent.sign({
        actionType: "t.test",
        complianceMode: true,
        // @ts-expect-error - intentionally outside the closed Literal
        receiptType: bad,
      }),
    ).rejects.toThrow(/invalid_receipt_type/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("rejects passive_telemetry paired with protectmcp:decision (rule 8)", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const agent = fakeAgent();
    await expect(
      agent.sign({
        actionType: "t.test",
        complianceMode: true,
        captureTopology: "passive_telemetry",
        receiptType: "protectmcp:decision",
      }),
    ).rejects.toThrow(/false_attestation_guard/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it.each([
    "protectmcp:decision",
    "protectmcp:restraint",
    "protectmcp:lifecycle",
    "protectmcp:lifecycle:configuration_change",
    "protectmcp:acknowledgment",
  ] as const)(
    "rule 8 widened: passive_telemetry paired with %s rejects with false_attestation_guard",
    async (receiptType) => {
      const fetchSpy = vi.spyOn(globalThis, "fetch");
      const agent = fakeAgent();
      const offending = receiptType.split(":").slice(1).join(":") || receiptType;
      await expect(
        agent.sign({
          actionType: "t.test",
          complianceMode: true,
          captureTopology: "passive_telemetry",
          receiptType,
        }),
      ).rejects.toThrow(/false_attestation_guard:/);
      await expect(
        agent.sign({
          actionType: "t.test",
          complianceMode: true,
          captureTopology: "passive_telemetry",
          receiptType,
        }),
      ).rejects.toThrow(/\(rule 8\)/);
      await expect(
        agent.sign({
          actionType: "t.test",
          complianceMode: true,
          captureTopology: "passive_telemetry",
          receiptType,
        }),
      ).rejects.toThrow(new RegExp(`not :${offending}`));
      expect(fetchSpy).not.toHaveBeenCalled();
    },
  );

  it("rule 8 widened: passive_telemetry paired with protectmcp:observation accepts", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-15T00:00:00Z",
        verification_url: "https://verify",
      }),
    );
    const agent = fakeAgent();
    await agent.sign({
      actionType: "t.test",
      complianceMode: true,
      captureTopology: "passive_telemetry",
      receiptType: "protectmcp:observation",
    });
    expect(fetchSpy).toHaveBeenCalled();
  });

  it("accepts passive_telemetry paired with protectmcp:observation", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        signature: "sig",
        signature_id: "sig_test",
        action_id: "act_test",
        timestamp: "2026-05-15T00:00:00Z",
        verification_url: "https://verify",
      }),
    );
    const agent = fakeAgent();
    await agent.sign({
      actionType: "t.test",
      complianceMode: true,
      captureTopology: "passive_telemetry",
      receiptType: "protectmcp:observation",
    });
    const init = fetchSpy.mock.calls[0][1];
    const body = JSON.parse((init?.body as string) ?? "{}");
    expect(body.capture_topology).toBe("passive_telemetry");
    expect(body.receipt_type).toBe("protectmcp:observation");
  });
});
