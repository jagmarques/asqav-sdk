/**
 * SDK parity for the build-provenance 4-tuple wire fields.
 *
 * Four optional wire fields on `agent.sign({...})` mirror the cloud
 * 0.5.1 SignRequest: `executable_hash`, `sbom_digest`,
 * `slsa_provenance_pointer`, `supply_chain_pointer`. The SDK forwards
 * each verbatim, validates the `sha256:<64-hex>` shape on the two
 * digest fields, and validates http(s) URL form on the two pointer
 * fields.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Agent, _resetForTests, init } from "../src/index.js";

const GOOD_EXECUTABLE_HASH = "sha256:" + "a".repeat(64);
const GOOD_SBOM_DIGEST = "sha256:" + "b".repeat(64);
const GOOD_SLSA_URL = "https://attestations.example.com/slsa/build-1.intoto.jsonl";
const GOOD_SUPPLY_CHAIN_URL = "https://rekor.sigstore.dev/api/v1/log/entries/abc";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function fakeAgent(): Agent {
  return Agent.attach({
    agent_id: "agt_exec_hash",
    name: "exec_hash",
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
  init({ apiKey: "k", apiUrl: "https://api.test" });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("executableHash wire forwarding", () => {
  it("forwards executable_hash to the request body", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "build:provenance",
      context: { k: "v" },
      complianceMode: true,
      executableHash: GOOD_EXECUTABLE_HASH,
    });
    const body = readBody(spy);
    expect(body.executable_hash).toBe(GOOD_EXECUTABLE_HASH);
  });

  it("forwards sbom_digest to the request body", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "build:provenance",
      context: { k: "v" },
      complianceMode: true,
      sbomDigest: GOOD_SBOM_DIGEST,
    });
    const body = readBody(spy);
    expect(body.sbom_digest).toBe(GOOD_SBOM_DIGEST);
  });

  it("forwards slsa_provenance_pointer to the request body", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "build:provenance",
      context: { k: "v" },
      complianceMode: true,
      slsaProvenancePointer: GOOD_SLSA_URL,
    });
    const body = readBody(spy);
    expect(body.slsa_provenance_pointer).toBe(GOOD_SLSA_URL);
  });

  it("forwards supply_chain_pointer to the request body", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "build:provenance",
      context: { k: "v" },
      complianceMode: true,
      supplyChainPointer: GOOD_SUPPLY_CHAIN_URL,
    });
    const body = readBody(spy);
    expect(body.supply_chain_pointer).toBe(GOOD_SUPPLY_CHAIN_URL);
  });

  it("forwards all four when supplied together", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "build:provenance",
      context: { k: "v" },
      complianceMode: true,
      executableHash: GOOD_EXECUTABLE_HASH,
      sbomDigest: GOOD_SBOM_DIGEST,
      slsaProvenancePointer: GOOD_SLSA_URL,
      supplyChainPointer: GOOD_SUPPLY_CHAIN_URL,
    });
    const body = readBody(spy);
    expect(body.executable_hash).toBe(GOOD_EXECUTABLE_HASH);
    expect(body.sbom_digest).toBe(GOOD_SBOM_DIGEST);
    expect(body.slsa_provenance_pointer).toBe(GOOD_SLSA_URL);
    expect(body.supply_chain_pointer).toBe(GOOD_SUPPLY_CHAIN_URL);
  });
});

describe("executableHash shape guards", () => {
  it("rejects malformed executable_hash before the HTTP roundtrip", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "build:provenance",
        context: { k: "v" },
        complianceMode: true,
        executableHash: "deadbeef",
      }),
    ).rejects.toThrow(/digest_format_guard: executable_hash/);
  });

  it("rejects malformed sbom_digest before the HTTP roundtrip", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "build:provenance",
        context: { k: "v" },
        complianceMode: true,
        sbomDigest: "not-a-digest",
      }),
    ).rejects.toThrow(/digest_format_guard: sbom_digest/);
  });

  it("rejects non-http slsa_provenance_pointer", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "build:provenance",
        context: { k: "v" },
        complianceMode: true,
        slsaProvenancePointer: "not a url",
      }),
    ).rejects.toThrow(/pointer_url_guard: slsa_provenance_pointer/);
  });

  it("rejects non-http supply_chain_pointer", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "build:provenance",
        context: { k: "v" },
        complianceMode: true,
        supplyChainPointer: "ftp://nope",
      }),
    ).rejects.toThrow(/pointer_url_guard: supply_chain_pointer/);
  });
});
