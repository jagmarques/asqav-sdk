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
    expect(output()).toMatch(/asqav 0\.2\.\d+/);
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

  it("blocks Pro command on Free tier (gating)", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        jsonResponse({ tier: "free", organization_id: "o", organization_name: "n" }),
      );
    try {
      await runCli(["preflight", "agt_x", "data:read"]);
    } catch (e) {
      // exit thrown
    }
    expect(output()).toMatch(/Pro tier/);
    expect(exitSpy).toHaveBeenCalledWith(2);
  });

  it("allows Pro command on Pro tier", async () => {
    vi.spyOn(globalThis, "fetch")
      // /account
      .mockResolvedValueOnce(
        jsonResponse({ tier: "pro", organization_id: "o", organization_name: "n" }),
      )
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
  });

  it("skips gating when /account is unreachable (fail-open)", async () => {
    vi.spyOn(globalThis, "fetch")
      // /account fails
      .mockResolvedValueOnce(jsonResponse({ detail: "not found" }, 404))
      .mockResolvedValueOnce(jsonResponse({ detail: "not found" }, 404))
      .mockResolvedValueOnce(jsonResponse({ detail: "not found" }, 404))
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
      .mockResolvedValueOnce(jsonResponse({ revoked: false, suspended: false }))
      .mockResolvedValueOnce(jsonResponse([]));

    try {
      await runCli(["preflight", "agt_x", "data:read"]);
    } catch (e) {
      // ignore exit
    }
    // No "Pro tier" gating message
    expect(output()).not.toMatch(/Pro tier/);
  });

  it("compliance frameworks lists static set", async () => {
    await runCli(["compliance", "frameworks"]);
    const out = output();
    expect(out).toContain("eu_ai_act_art12");
    expect(out).toContain("dora_ict");
    expect(out).toContain("soc2");
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
});
