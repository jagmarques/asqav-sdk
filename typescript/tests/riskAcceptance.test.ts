/**
 * SDK parity for the risk-acceptance / exception receipt.
 *
 * Mirrors the deployed cloud SignRequest validators (routes/agents.py
 * `_validate_risk_acceptance_extensions` + RiskSnapshot) for
 * `receiptType='protectmcp:lifecycle:risk_acceptance'`. The SDK forwards the
 * nine extension fields verbatim (camelCase -> snake_case), validates the
 * sarifDigest shape, the risk_snapshot numeric-requires-source rule, the
 * field-fence to the risk-acceptance type, and requires complianceMode so the
 * fields are never silently dropped from the signed bytes.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  Agent,
  AsqavError,
  RECEIPT_TYPE_NAMESPACE,
  RISK_ACCEPTANCE_DOCS_URL,
  _resetForTests,
  init,
} from "../src/index.js";

const RISK_TYPE = "protectmcp:lifecycle:risk_acceptance";
const GOOD_SARIF = "sha256:" + "a".repeat(64);
const GOOD_SNAPSHOT = {
  snapshotAt: "2026-06-01T19:00:00Z",
  snapshotSource: "CISA KEV catalog + FIRST EPSS feed",
  epss: "0.94210",
  cvss: "9.8",
  cvssVector: "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
  kevListed: true,
  cveIds: ["CVE-2026-1234", "CVE-2026-5678"],
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function fakeAgent(): Agent {
  return Agent.attach({
    agent_id: "agt_risk",
    name: "risk",
    public_key: "pk",
    key_id: "kid",
    algorithm: "ml-dsa-65",
    capabilities: [],
    created_at: "2026-05-25T00:00:00Z",
  });
}

function okBody(extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    signature: "sig",
    signature_id: "sig_test",
    action_id: "act_test",
    timestamp: "2026-06-01T19:00:00Z",
    verification_url: "https://verify",
    ...extra,
  };
}

function readBody(spy: ReturnType<typeof vi.spyOn>): Record<string, unknown> {
  const init = spy.mock.calls[0][1] as RequestInit;
  return JSON.parse((init?.body as string) ?? "{}");
}

function goodRiskOptions(): Record<string, unknown> {
  return {
    actionType: "risk:accept",
    context: { k: "v" },
    complianceMode: true,
    receiptType: RISK_TYPE,
    policyDecision: "none",
    approverId: "approver:alice@example.com",
    initiatorId: "initiator:bob@example.com",
    acceptanceReason: "Compensating control in place; accepted 90 days.",
    acceptedAt: "2026-06-01T19:05:00Z",
    supersedes: "act_prior_0001",
    sarifDigest: GOOD_SARIF,
    findingRef: "github.com/acme/repo/security/code-scanning/42",
    approvalRef: "JIRA-SEC-9001",
    riskSnapshot: GOOD_SNAPSHOT,
  };
}

beforeEach(() => {
  _resetForTests();
  init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("risk-acceptance receipt namespace", () => {
  it("includes the risk_acceptance type in the namespace", () => {
    expect(RECEIPT_TYPE_NAMESPACE).toContain(RISK_TYPE);
  });
});

describe("risk-acceptance wire forwarding (valid)", () => {
  it("forwards all nine extension fields as snake_case", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign(goodRiskOptions() as any);
    const body = readBody(spy);
    expect(body.approver_id).toBe("approver:alice@example.com");
    expect(body.initiator_id).toBe("initiator:bob@example.com");
    expect(body.acceptance_reason).toBe(
      "Compensating control in place; accepted 90 days.",
    );
    expect(body.accepted_at).toBe("2026-06-01T19:05:00Z");
    expect(body.supersedes).toBe("act_prior_0001");
    expect(body.sarif_digest).toBe(GOOD_SARIF);
    expect(body.finding_ref).toBe("github.com/acme/repo/security/code-scanning/42");
    expect(body.approval_ref).toBe("JIRA-SEC-9001");
    expect(body.risk_snapshot).toEqual({
      snapshot_at: "2026-06-01T19:00:00Z",
      snapshot_source: "CISA KEV catalog + FIRST EPSS feed",
      epss: "0.94210",
      cvss: "9.8",
      cvss_vector: "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
      kev_listed: true,
      cve_ids: ["CVE-2026-1234", "CVE-2026-5678"],
    });
  });

  it("keeps cvss/epss as strings on the wire", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign(goodRiskOptions() as any);
    const body = readBody(spy);
    const snap = body.risk_snapshot as Record<string, unknown>;
    expect(typeof snap.cvss).toBe("string");
    expect(typeof snap.epss).toBe("string");
  });
});

describe("risk-acceptance rejections", () => {
  it("rejects a bad sarif_digest shape", async () => {
    await expect(
      fakeAgent().sign({ ...goodRiskOptions(), sarifDigest: "not-a-sha256" } as any),
    ).rejects.toThrow(/sarif_digest_not_sha256_wire_form/);
  });

  it("rejects a risk_snapshot numeric without snapshot_source", async () => {
    await expect(
      fakeAgent().sign({
        ...goodRiskOptions(),
        riskSnapshot: { snapshotAt: "2026-06-01T19:00:00Z", snapshotSource: "", cvss: "9.8" },
      } as any),
    ).rejects.toThrow(/risk_snapshot_numeric_requires_snapshot_source/);
  });

  it("rejects risk fields on a non-risk-acceptance receipt type", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "lifecycle:change",
        complianceMode: true,
        receiptType: "protectmcp:lifecycle",
        sarifDigest: GOOD_SARIF,
      } as any),
    ).rejects.toThrow(/risk_acceptance_fields_require_risk_acceptance_receipt/);
  });

  it("rejects a risk-acceptance receipt missing approver_id/acceptance_reason", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "risk:accept",
        complianceMode: true,
        receiptType: RISK_TYPE,
        policyDecision: "none",
      } as any),
    ).rejects.toThrow(/risk_acceptance_missing_required_field/);
  });

  it("rejects a risk-acceptance receipt without compliance_mode", async () => {
    await expect(
      fakeAgent().sign({
        ...goodRiskOptions(),
        complianceMode: false,
      } as any),
    ).rejects.toThrow(/risk_acceptance_requires_compliance_mode/);
  });

  it("rejects a risk-acceptance receipt with a non-none policy_decision", async () => {
    await expect(
      fakeAgent().sign({
        ...goodRiskOptions(),
        policyDecision: "permit",
      } as any),
    ).rejects.toThrow(/risk_acceptance_requires_no_policy_decision/);
  });

  it("attaches the docs_url to the validation error (rule 3.9)", async () => {
    try {
      await fakeAgent().sign({ ...goodRiskOptions(), sarifDigest: "bad" } as any);
      throw new Error("expected rejection");
    } catch (err) {
      expect(err).toBeInstanceOf(AsqavError);
      expect((err as AsqavError).docsUrl).toBe(RISK_ACCEPTANCE_DOCS_URL);
    }
  });
});

describe("SoD guard: risk_acceptance_self_approval_guard", () => {
  it.each(["user:alice@example.com", "uuid-1234", "alice"])(
    "rejects initiator_id == approver_id ('%s')",
    async (sameId: string) => {
      await expect(
        fakeAgent().sign({
          ...goodRiskOptions(),
          initiatorId: sameId,
          approverId: sameId,
        } as any),
      ).rejects.toThrow(/risk_acceptance_self_approval_guard/);
    },
  );

  it("guard message is byte-identical to the cloud validator prefix", async () => {
    try {
      await fakeAgent().sign({
        ...goodRiskOptions(),
        initiatorId: "x",
        approverId: "x",
      } as any);
      throw new Error("expected rejection");
    } catch (err) {
      expect(err).toBeInstanceOf(AsqavError);
      const msg = (err as AsqavError).message;
      expect(msg).toMatch(
        /^risk_acceptance_self_approval_guard: initiator_id equals approver_id; a self-approved risk acceptance is incoherent/,
      );
    }
  });

  it("allows different initiator and approver", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    const body = await fakeAgent().sign({
      ...goodRiskOptions(),
      initiatorId: "initiator:bob@example.com",
      approverId: "approver:alice@example.com",
    } as any);
    expect(spy).toHaveBeenCalled();
    void body;
  });

  it("allows absent initiator_id (only approver set)", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      ...goodRiskOptions(),
      initiatorId: undefined,
    } as any);
    expect(spy).toHaveBeenCalled();
  });

  it("attaches the docs_url to the SoD guard error", async () => {
    try {
      await fakeAgent().sign({
        ...goodRiskOptions(),
        initiatorId: "same",
        approverId: "same",
      } as any);
      throw new Error("expected rejection");
    } catch (err) {
      expect(err).toBeInstanceOf(AsqavError);
      expect((err as AsqavError).docsUrl).toBe(RISK_ACCEPTANCE_DOCS_URL);
    }
  });
});
