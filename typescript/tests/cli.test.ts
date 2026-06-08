import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { runCli } from "../src/cli.js";
import { _resetForTests } from "../src/index.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("asqav CLI (TypeScript)", () => {
  let exitSpy: ReturnType<typeof vi.spyOn>;
  let stdoutSpy: ReturnType<typeof vi.spyOn>;
  let stderrSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    _resetForTests();
    process.env.ASQAV_API_KEY = "sk_test";
    exitSpy = vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
      throw new Error(`__EXIT__${code ?? 0}`);
    }) as never);
    stdoutSpy = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
    stderrSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
  });

  afterEach(() => {
    delete process.env.ASQAV_API_KEY;
    vi.restoreAllMocks();
  });

  function output(): string {
    const out = stdoutSpy.mock.calls.map((c: unknown[]) => String(c[0])).join("");
    const err = stderrSpy.mock.calls.map((c: unknown[]) => String(c[0])).join("");
    return out + err;
  }

  it("prints version with --version", async () => {
    await expect(runCli(["--version"])).resolves.not.toThrow();
    expect(output()).toMatch(/asqav 0\.\d+\.\d+/);
  });

  it("prints help with --help", async () => {
    await runCli(["--help"]);
    expect(output()).toContain("AI agent governance CLI");
    expect(output()).toContain("preflight");
    expect(output()).toContain("compliance");
  });

  it("requires ASQAV_API_KEY for agents list", async () => {
    delete process.env.ASQAV_API_KEY;
    try {
      await runCli(["agents", "list"]);
    } catch (e) {
      // process.exit mocked via throw
    }
    expect(output()).toContain("ASQAV_API_KEY");
    expect(exitSpy).toHaveBeenCalledWith(1);
  });

  it("runs preflight on the free tier without a tier gate", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      // Agent.get
      .mockResolvedValueOnce(
        jsonResponse({
          agent_id: "agt_x",
          name: "n",
          public_key: "pk",
          key_id: "kid",
          algorithm: "ml-dsa-65",
          capabilities: [],
          created_at: "2026-01-01T00:00:00Z",
        }),
      )
      // /agents/{id}/status
      .mockResolvedValueOnce(jsonResponse({ revoked: false, suspended: false }))
      // /policies
      .mockResolvedValueOnce(jsonResponse([]));

    try {
      await runCli(["preflight", "agt_x", "data:read"]);
    } catch (e) {
      // CLEARED -> no exit; if BLOCKED -> exit 1. Either way swallowed.
    }
    expect(output()).toMatch(/CLEARED|BLOCKED/);
    expect(output()).not.toMatch(/Pro tier/);
    expect(output()).not.toMatch(/Upgrade/);
    // No /account call is made as a gate; the first request is Agent.get.
    const firstUrl = String(fetchSpy.mock.calls[0]?.[0] ?? "");
    expect(firstUrl).not.toContain("/account");
  });

  it("compliance frameworks lists static set", async () => {
    await runCli(["compliance", "frameworks"]);
    const out = output();
    expect(out).toContain("eu_ai_act");
    expect(out).toContain("dora");
    expect(out).toContain("nydfs_500");
    expect(out).toContain("colorado_ai");
    expect(out).toContain("texas_traiga");
    expect(out).toContain("nist_ai_rmf");
    expect(out).toContain("circia");
    expect(out).toContain("hipaa_security");
    expect(out).toContain("sec_17a4a");
    expect(out).toContain("sec_17a4b");
  });

  it("rejects unknown command with usage hint", async () => {
    try {
      await runCli(["does-not-exist"]);
    } catch {
      // exit
    }
    expect(output()).toMatch(/Unknown command/);
    expect(exitSpy).toHaveBeenCalledWith(1);
  });

  it("verify --output text surfaces IETF VerificationDetail axes", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      jsonResponse({
        signature_id: "sig_1",
        agent_id: "agt_1",
        agent_name: "agt-1",
        action_id: "act_1",
        action_type: "x.y",
        signature: "sig",
        algorithm: "ml-dsa-65",
        signed_at: "2026-05-04T00:00:00Z",
        verified: true,
        verification_url: "https://verify.example/sig_1",
        verification_detail: {
          signer_key_match: true,
          signature_valid: true,
          algorithm_match: true,
          agent_active: true,
          validation_label: "valid",
          chain_valid: true,
          anchor_status_ots: "pending",
          anchor_status_rfc3161: "valid",
          signed_at_skew_seconds: 0.5,
        },
      }),
    );
    await runCli(["verify", "sig_1"]);
    const out = output();
    expect(out).toContain("Chain valid: true");
    expect(out).toContain("Anchor (OTS): pending");
    expect(out).toContain("Anchor (RFC 3161): valid");
    expect(out).toContain("signed_at skew: 0.5");
  });

  it("verify --output json emits the full detail", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      jsonResponse({
        signature_id: "sig_2",
        agent_id: "agt_2",
        algorithm: "ed25519",
        signed_at: "2026-05-04T00:00:00Z",
        verified: true,
        verification_detail: {
          signer_key_match: true,
          signature_valid: true,
          algorithm_match: true,
          agent_active: true,
          validation_label: "valid",
          chain_valid: true,
          anchor_status_ots: "valid",
          anchor_status_rfc3161: "valid",
          signed_at_skew_seconds: 0,
        },
      }),
    );
    await runCli(["verify", "sig_2", "--output", "json"]);
    const parsed = JSON.parse(output());
    expect(parsed.signatureId).toBe("sig_2");
    expect(parsed.verificationDetail.chainValid).toBe(true);
    expect(parsed.verificationDetail.anchorStatusOts).toBe("valid");
  });

  it("sign rejects --policy-decision deny without --reason", async () => {
    try {
      await runCli([
        "sign",
        "--agent-id", "agt_x",
        "--action-type", "x.y",
        "--policy-decision", "deny",
      ]);
    } catch {
      // exit
    }
    expect(output()).toContain("--reason is required");
    expect(exitSpy).toHaveBeenCalledWith(1);
  });

  it("sign exits with usage when --agent-id missing", async () => {
    try {
      await runCli(["sign", "--action-type", "x.y"]);
    } catch {
      // exit
    }
    expect(output()).toContain("Usage: asqav sign");
    expect(exitSpy).toHaveBeenCalledWith(1);
  });

  it("sign prints text output and surfaces compliance fields", async () => {
    vi.spyOn(globalThis, "fetch")
      // Agent.get
      .mockResolvedValueOnce(
        jsonResponse({
          agent_id: "agt_x",
          name: "n",
          public_key: "pk",
          key_id: "kid",
          algorithm: "ml-dsa-65",
          capabilities: [],
          created_at: "2026-01-01T00:00:00Z",
        }),
      )
      // /agents/{id}/sign
      .mockResolvedValueOnce(
        jsonResponse({
          signature: "sig_b64",
          signature_id: "sig_1",
          action_id: "act_1",
          timestamp: "2026-05-04T00:00:00Z",
          verification_url: "https://verify.example/sig_1",
          algorithm: "ml-dsa-65",
          policy_digest: "sha256:abc",
          compliance_mode: true,
          receipt_type: "protectmcp:decision",
          action_ref: "sha256:def",
          previous_receipt_hash: "0".repeat(64),
          decision: "allow",
        }),
      );
    await runCli([
      "sign",
      "--agent-id", "agt_x",
      "--action-type", "x.y",
      "--session-id", "sess_1",
    ]);
    const out = output();
    expect(out).toContain("signature_id: sig_1");
    expect(out).toContain("compliance_mode: true");
    expect(out).toContain("receipt_type:  protectmcp:decision");
    expect(out).toContain("action_ref:    sha256:def");
    expect(out).toContain("decision:      allow");
    expect(out).toContain("policy_digest: sha256:abc");
    expect(out).toContain("verify_url:   https://verify.example/sig_1");
  });

  it("sign --output json emits the camelCase payload", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        jsonResponse({
          agent_id: "agt_y",
          name: "n",
          public_key: "pk",
          key_id: "kid",
          algorithm: "ed25519",
          capabilities: [],
          created_at: "2026-01-01T00:00:00Z",
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          signature: "sig_b64",
          signature_id: "sig_2",
          action_id: "act_2",
          timestamp: "2026-05-04T00:00:00Z",
          verification_url: "https://verify.example/sig_2",
          algorithm: "ed25519",
        }),
      );
    await runCli([
      "sign",
      "--agent-id", "agt_y",
      "--action-type", "x.y",
      "--no-compliance-mode",
      "--output", "json",
      "--decision", "allow",
    ]);
    const parsed = JSON.parse(output());
    expect(parsed.signatureId).toBe("sig_2");
    expect(parsed.algorithm).toBe("ed25519");
  });

  it("audit-pack policy normalizes a bare 64-hex digest", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        digest: "sha256:" + "a".repeat(64),
        organization_id: "org_x",
        agent_id: "agt_x",
        action_type: "payment.wire_transfer",
        artefact: { rule: "deny_over_500k" },
        created_at: "2026-04-01T00:00:00Z",
      }),
    );
    await runCli(["audit-pack", "policy", "a".repeat(64), "--output", "text"]);
    expect(fetchSpy.mock.calls[0]?.[0]).toContain(`/audit-pack/policy/sha256:${"a".repeat(64)}`);
    expect(output()).toContain("deny_over_500k");
  });

  it("payloads erase requires --yes", async () => {
    try {
      await runCli(["payloads", "erase", "sig_abc"]);
    } catch {
      // exit
    }
    expect(output()).toContain("Refusing to erase without --yes");
    expect(exitSpy).toHaveBeenCalledWith(1);
  });

  it("payloads erase --yes calls DELETE", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(null, { status: 204 }),
    );
    await runCli(["payloads", "erase", "sig_abc", "--yes"]);
    expect(fetchSpy.mock.calls[0]?.[0]).toContain("/signatures/payloads/sig_abc");
    expect((fetchSpy.mock.calls[0]?.[1] as RequestInit).method).toBe("DELETE");
    expect(output()).toContain("Payload erased for sig_abc");
  });

  it("org set-compliance-strict requires exactly one of --enable / --disable", async () => {
    try {
      await runCli(["org", "set-compliance-strict", "org_x"]);
    } catch {
      // exit
    }
    expect(output()).toContain("exactly one of --enable or --disable");
    expect(exitSpy).toHaveBeenCalledWith(1);
  });

  it("org set-compliance-strict --enable patches the org", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ id: "org_x", compliance_mode_strict: true }),
    );
    await runCli(["org", "set-compliance-strict", "org_x", "--enable"]);
    const init = fetchSpy.mock.calls[0]?.[1] as RequestInit;
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body as string)).toEqual({ compliance_mode_strict: true });
    expect(output()).toContain("compliance_mode_strict enabled");
  });

  it("keys generate ed25519 emits an SPKI DER public key", async () => {
    await runCli(["keys", "generate", "--algorithm", "ed25519"]);
    expect(output()).toContain("public key (SPKI DER, base64)");
  });

  it("keys generate ml-dsa-65 errors with server-side hint", async () => {
    try {
      await runCli(["keys", "generate", "--algorithm", "ml-dsa-65"]);
    } catch {
      // exit
    }
    expect(output()).toContain("server-side");
    expect(exitSpy).toHaveBeenCalledWith(1);
  });

  it("migrate run refuses without ASQAV_MAINTENANCE_KEY", async () => {
    delete process.env.ASQAV_MAINTENANCE_KEY;
    try {
      await runCli(["migrate", "run", "v3-20"]);
    } catch {
      // exit
    }
    expect(output()).toContain("ASQAV_MAINTENANCE_KEY");
    expect(exitSpy).toHaveBeenCalledWith(1);
  });

  it("migrate run rejects unsupported migrations", async () => {
    process.env.ASQAV_MAINTENANCE_KEY = "mk_test";
    try {
      await runCli(["migrate", "run", "v9-99"]);
    } catch {
      // exit
    }
    delete process.env.ASQAV_MAINTENANCE_KEY;
    expect(output()).toContain("unsupported migration");
    expect(exitSpy).toHaveBeenCalledWith(1);
  });

  it("migrate run v3-22 POSTs with X-Maintenance-Key", async () => {
    process.env.ASQAV_MAINTENANCE_KEY = "mk_test";
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ status: "ok", applied: 1, migration: "v3-22" }),
    );
    await runCli(["migrate", "run", "v3-22"]);
    delete process.env.ASQAV_MAINTENANCE_KEY;
    const url = fetchSpy.mock.calls[0]?.[0] as string;
    const init = fetchSpy.mock.calls[0]?.[1] as RequestInit;
    expect(url).toContain("/maintenance/run-migration-v3-22");
    expect((init.headers as Record<string, string>)["X-Maintenance-Key"]).toBe("mk_test");
    expect(output()).toContain("v3-22");
  });

  it("replay-verify --strict rejects synthetic-shape steps", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      jsonResponse({
        steps: [
          { signature_id: "sig_1" /* no signed_envelope */ },
        ],
      }),
    );
    try {
      await runCli(["replay-verify", "agt_x", "sess_x", "--strict"]);
    } catch {
      // exit
    }
    expect(output()).toContain("strict mode FAILED");
    expect(exitSpy).toHaveBeenCalledWith(1);
  });
});
