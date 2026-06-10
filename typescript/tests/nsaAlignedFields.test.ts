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
  RECEIPT_TYPE_NAMESPACE,
  _resetForTests,
  computeConfigManifestDigest,
  computeCveInventoryDigest,
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
    expect(fp1).toMatch(/^[0-9a-f]{32}$/);
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
    expect(body.tool_fingerprint as string).toMatch(/^[0-9a-f]{32}$/);
  });

  it("explicit toolFingerprint overrides auto-derivation", async () => {
    const explicit = "a".repeat(32);
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
      "configuration_change_missing_config_manifest_digest: receipt_type=protectmcp:lifecycle:configuration_change requires config_manifest_digest (sha256:<64 hex>).",
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

// === Rule 10: expiry_collision_guard (validSeconds vs expiresAt) ===

describe("Rule 10: expiry_collision_guard", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
  });
  afterEach(() => vi.restoreAllMocks());

  it("rejects passing both validSeconds and expiresAt verbatim", async () => {
    const spy = vi.spyOn(globalThis, "fetch");
    await expect(
      fakeAgent().sign({
        actionType: "api.call",
        complianceMode: true,
        validSeconds: 3600,
        expiresAt: "2026-06-01T00:00:00Z",
      }),
    ).rejects.toThrow(
      "expiry_collision_guard: pass either valid_seconds or expires_at, not both (rule 10)",
    );
    expect(spy).not.toHaveBeenCalled();
  });

  it("passing neither uses the server-side default", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    await fakeAgent().sign({ actionType: "api.call", complianceMode: true });
    const body = readBody(spy);
    expect(body).not.toHaveProperty("valid_seconds");
    expect(body).not.toHaveProperty("expires_at");
  });

  it("only validSeconds is forwarded verbatim", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "api.call",
      complianceMode: true,
      validSeconds: 7200,
    });
    const body = readBody(spy);
    expect(body.valid_seconds).toBe(7200);
    expect(body).not.toHaveProperty("expires_at");
  });

  it("converts an ISO expiresAt to valid_seconds and drops expires_at", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    // Far-future horizon -> a large positive duration. The SDK converts ISO expiresAt
    // to valid_seconds client-side so the field is never silently dropped.
    await fakeAgent().sign({
      actionType: "api.call",
      complianceMode: true,
      expiresAt: "2099-06-01T00:00:00Z",
    });
    const body = readBody(spy);
    expect(body).not.toHaveProperty("expires_at");
    expect(body.valid_seconds as number).toBeGreaterThan(0);
  });

  it("converts a POSIX-epoch expiresAt to valid_seconds", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "api.call",
      complianceMode: true,
      expiresAt: Date.now() / 1000 + 3600,
    });
    const body = readBody(spy);
    expect(body).not.toHaveProperty("expires_at");
    expect(body.valid_seconds as number).toBeGreaterThanOrEqual(3590);
    expect(body.valid_seconds as number).toBeLessThanOrEqual(3601);
  });
});

// === Rule 11: digest_format_guard (sha256:<64-hex>) ===

const VALID_SHA = `sha256:${"f".repeat(64)}`;

describe("Rule 11: digest_format_guard", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
  });
  afterEach(() => vi.restoreAllMocks());

  const digestFields: Array<{ wire: string; opt: keyof Parameters<Agent["sign"]>[0]; token: (w: string) => string }> = [
    {
      wire: "config_manifest_digest",
      opt: "configManifestDigest",
      token: (w) => `${w}_not_sha256_wire_form: must look like 'sha256:<64 lowercase hex>'.`,
    },
    {
      wire: "cve_inventory_digest",
      opt: "cveInventoryDigest",
      token: (w) => `${w}_not_sha256_wire_form: must look like 'sha256:<64 lowercase hex>'.`,
    },
  ];

  for (const { wire, opt, token } of digestFields) {
    it(`${wire}: valid format passes`, async () => {
      const spy = vi
        .spyOn(globalThis, "fetch")
        .mockResolvedValue(jsonResponse(okBody()));
      const sigOpts = {
        actionType: "api.call",
        complianceMode: true,
        [opt]: VALID_SHA,
      } as unknown as Parameters<Agent["sign"]>[0];
      await fakeAgent().sign(sigOpts);
      const body = readBody(spy);
      expect(body[wire]).toBe(VALID_SHA);
    });

    it(`${wire}: wrong prefix is rejected verbatim`, async () => {
      const spy = vi.spyOn(globalThis, "fetch");
      const sigOpts = {
        actionType: "api.call",
        complianceMode: true,
        [opt]: `md5:${"0".repeat(64)}`,
      } as unknown as Parameters<Agent["sign"]>[0];
      await expect(fakeAgent().sign(sigOpts)).rejects.toThrow(token(wire));
      expect(spy).not.toHaveBeenCalled();
    });

    it(`${wire}: wrong length is rejected verbatim`, async () => {
      const spy = vi.spyOn(globalThis, "fetch");
      const sigOpts = {
        actionType: "api.call",
        complianceMode: true,
        [opt]: "sha256:abc123",
      } as unknown as Parameters<Agent["sign"]>[0];
      await expect(fakeAgent().sign(sigOpts)).rejects.toThrow(token(wire));
      expect(spy).not.toHaveBeenCalled();
    });

    it(`${wire}: non-hex characters are rejected verbatim`, async () => {
      const spy = vi.spyOn(globalThis, "fetch");
      const sigOpts = {
        actionType: "api.call",
        complianceMode: true,
        [opt]: `sha256:${"Z".repeat(64)}`,
      } as unknown as Parameters<Agent["sign"]>[0];
      await expect(fakeAgent().sign(sigOpts)).rejects.toThrow(token(wire));
      expect(spy).not.toHaveBeenCalled();
    });
  }

  // tool_fingerprint uses the bare cloud wire form (32 hex, no prefix),
  // distinct from the self-describing sha256:<64-hex> form of the others.
  it("tool_fingerprint: 32 bare hex passes", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    const fp = "a".repeat(32);
    await fakeAgent().sign({
      actionType: "api.call",
      complianceMode: true,
      toolFingerprint: fp,
    });
    const body = readBody(spy);
    expect(body.tool_fingerprint).toBe(fp);
  });

  for (const bad of [
    `sha256:${"a".repeat(64)}`, // self-describing form now rejected
    "a".repeat(31), // too short
    "a".repeat(33), // too long
    "Z".repeat(32), // non-hex / uppercase
  ]) {
    it(`tool_fingerprint: '${bad.slice(0, 12)}...' rejected verbatim`, async () => {
      const spy = vi.spyOn(globalThis, "fetch");
      await expect(
        fakeAgent().sign({
          actionType: "api.call",
          complianceMode: true,
          toolFingerprint: bad,
        }),
      ).rejects.toThrow(
        "tool_fingerprint_not_32_hex_chars: must be 32 lowercase hex chars (SHA-256[:32]).",
      );
      expect(spy).not.toHaveBeenCalled();
    });
  }
});

describe("digest helpers", () => {
  it("computeToolFingerprint emits 32 bare hex chars", () => {
    const fp = computeToolFingerprint("search", { q: "string" });
    expect(fp).toMatch(/^[0-9a-f]{32}$/);
  });

  it("computeConfigManifestDigest is deterministic under key reorder", () => {
    const d1 = computeConfigManifestDigest({ a: 1, b: [2, 3] });
    const d2 = computeConfigManifestDigest({ b: [2, 3], a: 1 });
    expect(d1).toBe(d2);
    expect(d1).toMatch(/^sha256:[a-f0-9]{64}$/);
  });

  it("computeCveInventoryDigest is deterministic across snapshots", () => {
    const cves = [{ id: "CVE-2026-0001", severity: "high" }];
    const d1 = computeCveInventoryDigest(cves);
    const d2 = computeCveInventoryDigest([{ ...cves[0] }]);
    expect(d1).toBe(d2);
    expect(d1).toMatch(/^sha256:[a-f0-9]{64}$/);
  });
});

// === BLOCKER 5: protectmcp:observation:result_bound receipt type ===

describe("protectmcp:observation:result_bound receipt type", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
  });
  afterEach(() => vi.restoreAllMocks());

  it("is a member of RECEIPT_TYPE_NAMESPACE", () => {
    expect(RECEIPT_TYPE_NAMESPACE as readonly string[]).toContain(
      "protectmcp:observation:result_bound",
    );
  });

  for (const receiptType of [...RECEIPT_TYPE_NAMESPACE].sort()) {
    it(`every namespace member round-trips: ${receiptType}`, async () => {
      const spy = vi
        .spyOn(globalThis, "fetch")
        .mockResolvedValue(jsonResponse(okBody()));
      const opts: Record<string, unknown> = {
        actionType: "api.call",
        complianceMode: true,
        receiptType,
      };
      if (receiptType === "protectmcp:lifecycle:configuration_change") {
        opts.configManifestDigest = computeConfigManifestDigest({ mode: "test" });
      }
      if (receiptType === "protectmcp:observation:result_bound") {
        opts.resultDigest = `sha256:${"0".repeat(64)}`;
      }
      if (receiptType === "protectmcp:lifecycle:risk_acceptance") {
        opts.approverId = "approver:alice@example.com";
        opts.acceptanceReason = "accepted";
      }
      if (receiptType === "protectmcp:lifecycle:code_authorship") {
        opts.repoRef = "github.com/acme/repo";
        opts.commitSha = "c0ffee0000000000000000000000000000000000";
      }
      if (
        receiptType.startsWith("protectmcp:lifecycle")
        || receiptType.startsWith("protectmcp:observation")
      ) {
        opts.policyDecision = "none";
      }
      await fakeAgent().sign(opts as unknown as Parameters<Agent["sign"]>[0]);
      const body = readBody(spy);
      expect(body.receipt_type).toBe(receiptType);
    });
  }

  it("result_bound carries result_digest verbatim", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));
    const resultDigest = `sha256:${"9".repeat(64)}`;
    await fakeAgent().sign({
      actionType: "api.call",
      complianceMode: true,
      receiptType: "protectmcp:observation:result_bound",
      policyDecision: "none",
      resultDigest,
    });
    const body = readBody(spy);
    expect(body.receipt_type).toBe("protectmcp:observation:result_bound");
    expect(body.result_digest).toBe(resultDigest);
  });
});
