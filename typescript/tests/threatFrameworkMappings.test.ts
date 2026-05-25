/**
 * SDK parity for the threat-framework taxonomy wire fields.
 *
 * Seven optional wire fields on `agent.sign({...})` mirror the cloud
 * SignRequest cross-field validator: `mitre_techniques`, `mitre_atlas`,
 * `owasp_llm_top10`, `nist_ai_rmf`, `iso_42001`, `eu_ai_act_articles`,
 * and `rfc3161_timestamp`. The SDK forwards each verbatim, validates
 * each list field as a non-empty array of strings (each entry <= 128
 * chars), and validates `rfc3161Timestamp` as base64.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Agent, _resetForTests, init } from "../src/index.js";

const GOOD_MITRE_TECHNIQUES = ["T1059", "T1078"];
const GOOD_MITRE_ATLAS = ["AML.T0051", "AML.T0043"];
const GOOD_OWASP_LLM = ["LLM01", "LLM02"];
const GOOD_NIST_AI_RMF = ["GOVERN-1.1", "MEASURE-2.7"];
const GOOD_ISO_42001 = ["A.6.2.6"];
const GOOD_EU_AI_ACT = ["Article-12", "Article-15"];
const GOOD_RFC3161_B64 =
  "MIIGgzADAgEAMIIGegYJKoZIhvcNAQcCoIIGazCCBmcCAQMxDTALBglghkgBZQMEAgE=";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function fakeAgent(): Agent {
  return Agent.attach({
    agent_id: "agt_threat_fw",
    name: "threat_fw",
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
    timestamp: "2026-05-25T00:00:00Z",
    verification_url: "https://verify",
    ...extra,
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

describe("threat-framework taxonomy wire forwarding", () => {
  it.each([
    ["mitre_techniques", "mitreTechniques", GOOD_MITRE_TECHNIQUES],
    ["mitre_atlas", "mitreAtlas", GOOD_MITRE_ATLAS],
    ["owasp_llm_top10", "owaspLlmTop10", GOOD_OWASP_LLM],
    ["nist_ai_rmf", "nistAiRmf", GOOD_NIST_AI_RMF],
    ["iso_42001", "iso42001", GOOD_ISO_42001],
    ["eu_ai_act_articles", "euAiActArticles", GOOD_EU_AI_ACT],
  ])("forwards %s to the wire as snake_case", async (wire, kwarg, value) => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "api:call",
      context: { k: "v" },
      complianceMode: true,
      [kwarg]: value,
    } as any);
    const body = readBody(spy);
    expect(body[wire]).toEqual(value);
  });

  it("forwards rfc3161_timestamp to the wire", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "api:call",
      context: { k: "v" },
      complianceMode: true,
      rfc3161Timestamp: GOOD_RFC3161_B64,
    });
    const body = readBody(spy);
    expect(body.rfc3161_timestamp).toBe(GOOD_RFC3161_B64);
  });

  it("forwards all seven fields together", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "api:call",
      context: { k: "v" },
      complianceMode: true,
      mitreTechniques: GOOD_MITRE_TECHNIQUES,
      mitreAtlas: GOOD_MITRE_ATLAS,
      owaspLlmTop10: GOOD_OWASP_LLM,
      nistAiRmf: GOOD_NIST_AI_RMF,
      iso42001: GOOD_ISO_42001,
      euAiActArticles: GOOD_EU_AI_ACT,
      rfc3161Timestamp: GOOD_RFC3161_B64,
    });
    const body = readBody(spy);
    expect(body.mitre_techniques).toEqual(GOOD_MITRE_TECHNIQUES);
    expect(body.mitre_atlas).toEqual(GOOD_MITRE_ATLAS);
    expect(body.owasp_llm_top10).toEqual(GOOD_OWASP_LLM);
    expect(body.nist_ai_rmf).toEqual(GOOD_NIST_AI_RMF);
    expect(body.iso_42001).toEqual(GOOD_ISO_42001);
    expect(body.eu_ai_act_articles).toEqual(GOOD_EU_AI_ACT);
    expect(body.rfc3161_timestamp).toBe(GOOD_RFC3161_B64);
  });

  it("omits all seven fields from the wire when caller passes none", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "api:call",
      context: { k: "v" },
      complianceMode: true,
    });
    const body = readBody(spy);
    for (const k of [
      "mitre_techniques",
      "mitre_atlas",
      "owasp_llm_top10",
      "nist_ai_rmf",
      "iso_42001",
      "eu_ai_act_articles",
      "rfc3161_timestamp",
    ]) {
      expect(body[k]).toBeUndefined();
    }
  });
});

describe("threat-framework taxonomy validation", () => {
  it.each([
    ["mitre_techniques", "mitreTechniques"],
    ["mitre_atlas", "mitreAtlas"],
    ["owasp_llm_top10", "owaspLlmTop10"],
    ["nist_ai_rmf", "nistAiRmf"],
    ["iso_42001", "iso42001"],
    ["eu_ai_act_articles", "euAiActArticles"],
  ])("rejects empty list on %s with verbatim guard token", async (wire, kwarg) => {
    await expect(
      fakeAgent().sign({
        actionType: "api:call",
        context: { k: "v" },
        complianceMode: true,
        [kwarg]: [],
      } as any),
    ).rejects.toThrow(`${wire}_must_be_non_empty_list`);
  });

  it("rejects non-string entry with verbatim guard token", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "api:call",
        context: { k: "v" },
        complianceMode: true,
        owaspLlmTop10: ["LLM01", 42 as any],
      }),
    ).rejects.toThrow("owasp_llm_top10_entry_invalid");
  });

  it("rejects oversize entry with verbatim guard token", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "api:call",
        context: { k: "v" },
        complianceMode: true,
        iso42001: ["A".repeat(129)],
      }),
    ).rejects.toThrow("iso_42001_entry_invalid");
  });

  it("rejects empty rfc3161Timestamp with verbatim guard token", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "api:call",
        context: { k: "v" },
        complianceMode: true,
        rfc3161Timestamp: "",
      }),
    ).rejects.toThrow("rfc3161_timestamp_not_base64");
  });

  it("rejects invalid base64 rfc3161Timestamp with verbatim guard token", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "api:call",
        context: { k: "v" },
        complianceMode: true,
        rfc3161Timestamp: "not!!base64@@",
      }),
    ).rejects.toThrow("rfc3161_timestamp_not_base64");
  });
});
