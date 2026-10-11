import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { runCli } from "../src/cli.js";
import { _resetForTests, APIError, AsqavError, init, request, verify } from "../src/index.js";

const ENV_KEYS = ["ASQAV_CREDENTIALS_PATH", "ASQAV_API_KEY", "ASQAV_API_BASE", "ASQAV_API_URL"];
const FRESH = {
  signature_id: "sig_1",
  agent_id: "agt_1",
  agent_name: "my-agent",
  action_type: "test",
  algorithm: "ml-dsa-65",
  signed_at: "2026-10-11T00:00:00Z",
  verified: false,
  verification_detail: {
    signer_key_match: true,
    signature_valid: true,
    algorithm_match: true,
    agent_active: true,
    validation_label: "anchor_pending_no_cryptographic_proof",
    anchor_status_ots: "pending",
  },
};

function respond(body: unknown, status = 200, type = "application/json"): Response {
  const raw = typeof body === "string" ? body : JSON.stringify(body);
  return new Response(raw, { status, headers: { "Content-Type": type } });
}

describe("first-receipt onboarding (TypeScript)", () => {
  const saved: Record<string, string | undefined> = {};
  let credsFile = "";
  let stdoutSpy: ReturnType<typeof vi.spyOn>;
  const urls: string[] = [];

  beforeEach(() => {
    for (const k of ENV_KEYS) {
      saved[k] = process.env[k];
      delete process.env[k];
    }
    credsFile = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "asqav-756-")), "credentials");
    process.env.ASQAV_CREDENTIALS_PATH = credsFile;
    urls.length = 0;
    _resetForTests();
    stdoutSpy = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
    vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
      throw new Error(`__EXIT__${code ?? 0}`);
    }) as never);
  });

  afterEach(() => {
    for (const k of ENV_KEYS) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
    vi.restoreAllMocks();
  });

  function stubFetch(make: () => Response | Promise<Response>): void {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: string | URL | Request) => {
      urls.push(String(input));
      return make();
    });
  }

  function printed(): string {
    return stdoutSpy.mock.calls.map((c: unknown[]) => String(c[0])).join("");
  }

  it("verify prints a fresh receipt as valid with its anchor pending", async () => {
    stubFetch(() => respond(FRESH));
    await runCli(["verify", "sig_1"]);
    expect(printed()).toContain("Signature: valid");
    expect(printed()).toContain("Anchor: pending");
    expect(printed()).not.toContain("invalid");
  });

  it.each([
    "signer_key_changed",
    "agent_revoked_before_issuance",
    "chain_break",
    "signature_expired",
    "agent_unknown",
    "compliance_mode_anchor_required",
  ])("verify reads a server rejection (%s) as invalid and exits 1", async (label) => {
    const detail = { ...FRESH.verification_detail, validation_label: label, anchor_valid_ots: true };
    stubFetch(() => respond({ ...FRESH, verification_detail: detail }));
    await expect(runCli(["verify", "sig_1"])).rejects.toThrow("__EXIT__1");
    expect(printed()).toContain(`Signature: invalid (${label})`);
  });

  it("verify reads a passing attestation as valid and exits 0", async () => {
    const detail = {
      ...FRESH.verification_detail,
      validation_label: "valid_commitment_not_rederivable",
      anchor_valid_ots: true,
      anchor_status_ots: "valid",
    };
    stubFetch(() => respond({ ...FRESH, verification_detail: detail }));
    await runCli(["verify", "sig_1"]);
    expect(printed()).toContain("Signature: valid");
  });

  it("verify reads one invalid anchor as invalid", async () => {
    const detail = {
      ...FRESH.verification_detail,
      validation_label: "anchor_invalid",
      anchor_valid_ots: true,
      anchor_status_ots: "valid",
      anchor_status_rfc3161: "invalid",
    };
    stubFetch(() => respond({ ...FRESH, verification_detail: detail }));
    await expect(runCli(["verify", "sig_1"])).rejects.toThrow("__EXIT__1");
    expect(printed()).toContain("Anchor: invalid");
  });

  it.each([
    [["migrate", "run", "v3-20"], "ASQAV_MAINTENANCE_KEY"],
    [["org", "halt", "org_1", "--yes"], "ASQAV_SESSION_TOKEN"],
  ])("%j sends credentials only to the saved base", async (argv, envName) => {
    const base = "http://127.0.0.1:9/api/v1";
    fs.writeFileSync(credsFile, JSON.stringify({ api_key: "sk_test", api_base: base }));
    process.env[envName] = "secret_test_value";
    stubFetch(() => respond({ ok: true }));
    try {
      await runCli(argv);
    } finally {
      delete process.env[envName];
    }
    expect(urls.length).toBeGreaterThan(0);
    expect(urls.every((u) => u.startsWith(base))).toBe(true);
    expect(urls.some((u) => u.includes("asqav.com"))).toBe(false);
  });

  it("verify() reports signature validity and anchor state", async () => {
    stubFetch(() => respond(FRESH));
    const result = await verify("sig_1");
    expect(result).toMatchObject({ verified: false, signatureValid: true, anchor: "pending" });
  });

  it("init() uses the base saved by asqav login", async () => {
    fs.writeFileSync(credsFile, JSON.stringify({ api_key: "sk_test", api_base: "http://saved.test/api/v1" }));
    stubFetch(() => respond([]));
    init();
    await request("GET", "/agents");
    expect(urls[0]).toBe("http://saved.test/api/v1/agents");
  });

  it("resolves the base as arg, ASQAV_API_URL, ASQAV_API_BASE, saved login", async () => {
    fs.writeFileSync(credsFile, JSON.stringify({ api_key: "sk_test", api_base: "http://file.test" }));
    stubFetch(() => respond([]));
    process.env.ASQAV_API_BASE = "http://base.test";
    init();
    await request("GET", "/a");
    process.env.ASQAV_API_URL = "http://url.test";
    init();
    await request("GET", "/b");
    init({ baseUrl: "http://arg.test" });
    await request("GET", "/c");
    init();
    await request("GET", "/d");
    expect(urls).toEqual(["http://base.test/a", "http://url.test/b", "http://arg.test/c", "http://arg.test/d"]);
  });

  it("verify() wraps a refused connection in an AsqavError", async () => {
    stubFetch(() => Promise.reject(new TypeError("fetch failed", { cause: new Error("connect ECONNREFUSED") })));
    const err = await verify("sig_1").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(AsqavError);
    expect(String((err as Error).message)).toContain("ECONNREFUSED");
    expect(String((err as Error).message)).not.toContain("\n");
  });

  it("verify() turns an HTML error page into one line", async () => {
    stubFetch(() => respond("<html><head><title>502 Bad Gateway</title></head>\n<body>cf</body></html>", 502, "text/html"));
    const err = (await verify("sig_1").catch((e: unknown) => e)) as APIError;
    expect(err).toBeInstanceOf(APIError);
    expect(err.message).not.toMatch(/<html|\n/);
    expect(err.message).toContain("502");
  });

  it.each([
    [{ detail: [{ loc: ["body", "name"], msg: "String should have at most 64 characters" }] },
      "name: String should have at most 64 characters"],
    [{ detail: "API key lacks scope agents:write", code: "scope_missing" },
      "API key lacks scope agents:write (scope_missing)"],
  ])("keeps the server reason %#", async (body, reason) => {
    stubFetch(() => respond(body, 422));
    init({ apiKey: "sk_test", baseUrl: "http://stub.test/api/v1" });
    await expect(request("POST", "/agents/create", {})).rejects.toThrow(reason);
  });
});
