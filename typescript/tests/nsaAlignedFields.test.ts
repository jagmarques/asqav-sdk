/**
 * SDK parity for the cycle-30 NSA-aligned receipt extensions.
 *
 * Five new wire fields land on the cloud SignRequest per NSA CSI
 * U/OO/6030316-26: `result_digest`, `expires_at`, `nonce`,
 * `tool_fingerprint`, `config_manifest_digest` and `cve_inventory_digest`.
 * The SDK forwards caller-supplied digests verbatim, computes
 * `tool_fingerprint` from `toolName` + JSON schema, and auto-generates a
 * 24-hex-char `nonce` when the caller omits one. The cross-field
 * validator mirrors cloud rule 9: a
 * `protectmcp:lifecycle:configuration_change` receipt without
 * `configManifestDigest` fails before the HTTP roundtrip.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  Agent,
  _resetForTests,
  computeToolFingerprint,
  generateNonce,
  init,
} from "../src/index.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function fakeAgent(): Agent {
  return Agent.attach({
    agent_id: "agt_nsa",
    name: "nsa",
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

describe("generateNonce helper", () => {
  it("returns a 24-hex-char string", () => {
    const nonce = generateNonce();
    expect(nonce).toMatch(/^[0-9a-f]{24}$/);
  });
});

describe("computeToolFingerprint helper", () => {
  it("is deterministic under key reordering", () => {
    const fp1 = computeToolFingerprint("search", { a: 1, b: 2 });
    const fp2 = computeToolFingerprint("search", { b: 2, a: 1 });
    expect(fp1).toBe(fp2);
    expect(fp1.startsWith("sha256:")).toBe(true);
  });
});

describe("nonce auto-generation", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
  });
  afterEach(() => vi.restoreAllMocks());

  it("auto-generates a 24-hex-char nonce when caller omits one", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "api.call",
      complianceMode: true,
    });
    const body = readBody(spy);
    expect(body.nonce).toMatch(/^[0-9a-f]{24}$/);
  });

  it("forwards an explicit nonce verbatim", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "api.call",
      complianceMode: true,
      nonce: "0123456789abcdef01234567",
    });
    const body = readBody(spy);
    expect(body.nonce).toBe("0123456789abcdef01234567");
  });
});

describe("toolFingerprint auto-compute + passthrough", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
  });
  afterEach(() => vi.restoreAllMocks());

  it("derives tool_fingerprint from toolName + toolSchema", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "api.call",
      complianceMode: true,
      toolName: "search",
      toolSchema: { args: { q: "string" } },
    });
    const body = readBody(spy);
    expect(typeof body.tool_fingerprint).toBe("string");
    expect((body.tool_fingerprint as string).startsWith("sha256:")).toBe(true);
    expect((body.tool_fingerprint as string).length).toBe("sha256:".length + 64);
  });

  it("explicit toolFingerprint overrides auto-derivation", async () => {
    const explicit = `sha256:${"a".repeat(64)}`;
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "api.call",
      complianceMode: true,
      toolName: "search",
      toolSchema: { args: { q: "string" } },
      toolFingerprint: explicit,
    });
    const body = readBody(spy);
    expect(body.tool_fingerprint).toBe(explicit);
  });

  it("omits tool_fingerprint when toolSchema is not provided", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "api.call",
      complianceMode: true,
      toolName: "search",
    });
    const body = readBody(spy);
    expect(body).not.toHaveProperty("tool_fingerprint");
  });
});

describe("caller-forwarded NSA-aligned digest fields", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
  });
  afterEach(() => vi.restoreAllMocks());

  const cases = [
    { wire: "result_digest", opt: "resultDigest", value: `sha256:${"b".repeat(64)}` },
    { wire: "expires_at", opt: "expiresAt", value: "2026-06-01T00:00:00Z" },
    { wire: "config_manifest_digest", opt: "configManifestDigest", value: `sha256:${"c".repeat(64)}` },
    { wire: "cve_inventory_digest", opt: "cveInventoryDigest", value: `sha256:${"d".repeat(64)}` },
  ] as const;

  for (const { wire, opt, value } of cases) {
    it(`forwards ${opt} as ${wire}`, async () => {
      const spy = vi
        .spyOn(globalThis, "fetch")
        .mockResolvedValue(jsonResponse(okBody()));
      const sigOpts = {
        actionType: "api.call",
        complianceMode: true,
        [opt]: value,
      } as unknown as Parameters<Agent["sign"]>[0];
      await fakeAgent().sign(sigOpts);
      const body = readBody(spy);
      expect(body[wire]).toBe(value);
    });
  }

  it("omits all five NSA-aligned wire keys when not provided", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "api.call",
      complianceMode: true,
    });
    const body = readBody(spy);
    for (const k of [
      "result_digest",
      "expires_at",
      "tool_fingerprint",
      "config_manifest_digest",
      "cve_inventory_digest",
    ]) {
      expect(body).not.toHaveProperty(k);
    }
  });
});

describe("Rule 9: configuration_change requires configManifestDigest", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
  });
  afterEach(() => vi.restoreAllMocks());

  it("rejects configuration_change without configManifestDigest verbatim", async () => {
    const spy = vi.spyOn(globalThis, "fetch");
    await expect(
      fakeAgent().sign({
        actionType: "api.call",
        complianceMode: true,
        receiptType: "protectmcp:lifecycle:configuration_change",
        policyDecision: "none",
      }),
    ).rejects.toThrow(
      "false_attestation_guard: receipt_type=protectmcp:lifecycle:configuration_change requires config_manifest_digest (rule 9)",
    );
    expect(spy).not.toHaveBeenCalled();
  });

  it("accepts configuration_change when configManifestDigest is present", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    const digest = `sha256:${"e".repeat(64)}`;
    await fakeAgent().sign({
      actionType: "api.call",
      complianceMode: true,
      receiptType: "protectmcp:lifecycle:configuration_change",
      policyDecision: "none",
      configManifestDigest: digest,
    });
    const body = readBody(spy);
    expect(body.config_manifest_digest).toBe(digest);
    expect(body.receipt_type).toBe(
      "protectmcp:lifecycle:configuration_change",
    );
  });

  it("does not gate plain protectmcp:lifecycle receipts", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "api.call",
      complianceMode: true,
      receiptType: "protectmcp:lifecycle",
      policyDecision: "none",
    });
    const body = readBody(spy);
    expect(body.receipt_type).toBe("protectmcp:lifecycle");
    expect(body).not.toHaveProperty("config_manifest_digest");
  });
});
