import { createHash, createHmac } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  Agent, AuthenticationError, DetectorBlockedError, _resetForTests, clearDetectors,
  clearHooks, govern, init, registerBefore, registerDetector, request,
  type InitOptions,
} from "../src/index.js";
import { canonicalizeAction } from "../src/canonicalize.js";

const tenants = {
  A: { apiKey: "fixture-key-A-not-a-credential", baseUrl: "https://tenant-a.invalid/api/v1" },
  B: { apiKey: "fixture-key-B-not-a-credential", baseUrl: "https://tenant-b.invalid/api/v1" },
};
type Tenant = keyof typeof tenants;
type Call = { tenant: Tenant; path: string; body: Record<string, unknown> };
const calls: Call[] = [];
const payload = (tenant: Tenant) => ({
  agent_id: `agt_${tenant}`, name: "fixture", public_key: "fixture-public-key",
  key_id: "fixture-key-id", algorithm: "ml-dsa-65", capabilities: [], created_at: "2026-09-08",
});
const signResponse = {
  signature: "fixture-signature", signature_id: "sig_fixture", action_id: "act_fixture",
  timestamp: "2026-09-08", verification_url: "https://fixture.invalid/receipt",
};
let intercept: ((call: Call) => Promise<void>) | undefined;

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}

function configure(tenant: Tenant, extra: Partial<InitOptions> = {}) {
  init({ ...tenants[tenant], mode: "full-payload", ...extra });
}

beforeEach(() => {
  vi.stubEnv("ASQAV_MODE", "");
  _resetForTests();
  clearHooks();
  clearDetectors();
  calls.length = 0;
  intercept = undefined;
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, options) => {
    const url = new URL(String(input));
    const tenant = url.hostname === "tenant-a.invalid" ? "A" : "B";
    expect(url.origin).toBe(new URL(tenants[tenant].baseUrl).origin);
    expect(new Headers(options?.headers).get("x-api-key")).toBe(tenants[tenant].apiKey);
    const call: Call = { tenant, path: url.pathname, body: JSON.parse(String(options?.body ?? "{}")) };
    calls.push(call);
    await intercept?.(call);
    if (url.pathname.endsWith("/policies")) return Response.json([]);
    if (url.pathname.endsWith("/status")) return Response.json({ revoked: false });
    return Response.json({ ...payload(tenant), ...signResponse,
      session_id: `ses_${tenant}`, status: "active", started_at: "2026-09-08" });
  });
});

afterEach(() => {
  clearHooks();
  clearDetectors();
  vi.restoreAllMocks();
  vi.useRealTimers();
  vi.unstubAllEnvs();
  _resetForTests();
});

describe("Agent connection lifetime", () => {
  it.each(["create", "get", "attach", "govern-create", "govern-get"] as const)(
    "%s retains A while later defaults select B", async (factory) => {
      configure("A");
      const agent = factory === "create" ? await Agent.create({ name: "fixture" })
        : factory === "get" ? await Agent.get("agt_A")
        : factory === "attach" ? Agent.attach(payload("A"))
        : await govern({ ...tenants.A, ...(factory === "govern-get" ? { agentId: "agt_A" } : {}) });
      await govern({ ...tenants.B, agentName: "fixture" });
      calls.length = 0;
      await agent.sign({ actionType: "api:call" });
      await agent.countersign("sig_fixture");
      await agent.startSession();
      await agent.endSession();
      await agent.revoke();
      expect((await agent.preflight("api:call")).cleared).toBe(true);
      expect(calls).toHaveLength(7);
      expect(calls.every((call) => call.tenant === "A")).toBe(true);
      expect(calls.map((call) => call.path)).toEqual([
        "/api/v1/agents/agt_A/sign", "/api/v1/agents/agt_A/countersign/sig_fixture",
        "/api/v1/sessions/", "/api/v1/sessions/ses_A", "/api/v1/agents/agt_A/revoke",
        "/api/v1/agents/agt_A/status", "/api/v1/policies",
      ]);
      await request("GET", "/policies");
      expect(calls.at(-1)?.tenant).toBe("B");
      const future = await Agent.get("agt_B");
      await future.sign({ actionType: "api:call" });
      expect(calls.at(-1)?.tenant).toBe("B");
    },
  );

  it.each(["A", "B"] as const)("captures %s before create/get responses resolve", async (first) => {
    const second = first === "A" ? "B" : "A";
    for (const operation of ["create", "get"] as const) {
      const entered = deferred();
      const release = deferred();
      configure(first);
      intercept = async (call) => {
        if (call.tenant === first && !call.path.endsWith("/sign")) {
          entered.resolve();
          await release.promise;
        }
      };
      const pending = operation === "create" ? Agent.create({ name: "fixture" }) : Agent.get(`agt_${first}`);
      await entered.promise;
      const other = await govern({ ...tenants[second], agentName: "fixture" });
      await other.sign({ actionType: "api:call" });
      release.resolve();
      const original = await pending;
      await original.sign({ actionType: "api:call" });
      expect(calls.slice(-4).map((call) => call.tenant)).toEqual([first, second, second, first]);
    }
  });

  it("uses its original connection for the second preflight read", async () => {
    configure("A");
    const agent = Agent.attach(payload("A"));
    const entered = deferred();
    const release = deferred();
    intercept = async (call) => {
      if (call.path.endsWith("/status")) { entered.resolve(); await release.promise; }
    };
    const pending = agent.preflight("api:call");
    await entered.promise;
    configure("B");
    release.resolve();
    expect((await pending).checksComplete).toBe(true);
    expect(calls.map((call) => call.tenant)).toEqual(["A", "A"]);
  });

  it("does not bind an uninitialized attached Agent during a later init", async () => {
    const agent = Agent.attach(payload("A"));
    configure("B");
    await expect(agent.sign({ actionType: "api:call" })).rejects.toBeInstanceOf(AuthenticationError);
    await expect(agent.countersign("sig_fixture")).rejects.toBeInstanceOf(AuthenticationError);
    expect(calls).toHaveLength(0);
  });

  it("publishes no partial default when initialization fails", async () => {
    configure("A", { mode: "hash-only", orgSalt: new Uint8Array([1, 2, 3]) });
    expect(() => configure("B", { mode: "invalid" as InitOptions["mode"] })).toThrow("mode must");
    const agent = await Agent.get("agt_A");
    await agent.sign({ actionType: "api:call" });
    expect(calls.map((call) => call.tenant)).toEqual(["A", "A"]);
    const canonical = canonicalizeAction("api:call", {});
    expect(calls.at(-1)?.body.hash).toBe(`sha256:${createHmac("sha256", new Uint8Array([1, 2, 3])).update(canonical).digest("hex")}`);
  });

  it.each([false, true])("owns salt bytes even when input is a Buffer: %s", async (buffer) => {
    const salt = buffer ? Buffer.alloc(32, 7) : new Uint8Array(32).fill(7);
    configure("A", { mode: "hash-only", orgSalt: salt });
    salt.fill(9);
    const agent = Agent.attach(payload("A"));
    configure("B");
    await agent.sign({ actionType: "api:call", context: { private: "fixture" } });
    const canonical = canonicalizeAction("api:call", { private: "fixture" });
    expect(calls[0].body.hash).toBe(`sha256:${createHmac("sha256", Buffer.alloc(32, 7)).update(canonical).digest("hex")}`);
    expect(Object.getOwnPropertyNames(agent).sort()).toEqual([
      "agentId", "algorithm", "capabilities", "createdAt", "keyId", "name", "publicKey", "sessionId",
    ]);
    expect(Object.getOwnPropertySymbols(agent)).toEqual([]);
    const serialized = JSON.stringify(agent);
    for (const secret of [tenants.A.apiKey, tenants.A.baseUrl, "orgSalt", "hash-only"]) {
      expect(serialized).not.toContain(secret);
    }
  });
});

describe.each(["hash-only", "full-payload"] as const)("bound %s signing", (mode) => {
  it.each(["hook", "schema", "detector"] as const)("keeps mode and salt across %s reinitialization", async (boundary) => {
    const salt = new Uint8Array(32).fill(5);
    configure("A", { mode, orgSalt: salt });
    const agent = Agent.attach(payload("A"));
    const context: Record<string, unknown> = { private: "fixture", _model_name: "fixture-model" };
    const entered = deferred();
    const release = deferred();
    const switchDefault = () => configure("B", { mode: mode === "hash-only" ? "full-payload" : "hash-only" });
    if (boundary === "hook") registerBefore("*", () => { switchDefault(); });
    const verdict = { allow: true, confidence: 1, labels: [], detector: "fixture", reason: "fixture" };
    if (boundary === "detector") registerDetector({ name: "fixture", inspect: async () => {
      entered.resolve(); await release.promise; return verdict;
    } });
    const pending = agent.sign({
      actionType: "api:call", context, policyDecision: "permit",
      contextSchema: boundary === "schema" ? switchDefault : undefined,
    });
    if (boundary === "detector") {
      await entered.promise; switchDefault(); release.resolve();
    }
    await pending;
    expect(calls[0].tenant).toBe("A");
    const body = calls[0].body;
    const signedContext = boundary === "detector" ? { ...context, _detectors: [verdict] } : context;
    expect(body.policy_decision).toBe("permit");
    expect(body.action_ref).toMatch(/^sha256:[0-9a-f]{64}$/);
    if (mode === "hash-only") {
      const canonical = canonicalizeAction("api:call", signedContext);
      expect(body.hash).toBe(`sha256:${createHmac("sha256", salt).update(canonical).digest("hex")}`);
      expect(body.hash_algo).toBe("hmac-sha256");
      expect(body.payload_size).toBe(canonical.byteLength);
      expect(body.metadata).toEqual({ agent_id: "agt_A", action_type: "api:call", model_name: "fixture-model" });
      expect(body).not.toHaveProperty("context");
    } else {
      expect(body.context).toEqual(signedContext);
      expect(body).not.toHaveProperty("hash");
    }
    clearHooks(); clearDetectors();
    await Agent.attach(payload("B")).sign({ actionType: "api:call", context });
    expect(calls.at(-1)?.tenant).toBe("B");
    expect(calls.at(-1)?.body).toHaveProperty(mode === "hash-only" ? "context" : "hash");
  });
});

it.each(["deny", "throw"] as const)("preserves late detector %s blocking before HTTP", async (failure) => {
  configure("A");
  const agent = Agent.attach(payload("A"));
  registerDetector({ name: "fixture", inspect: () => {
    configure("B");
    if (failure === "throw") throw new Error("fixture detector failure");
    return { allow: false, confidence: 1, labels: ["fixture-deny"], detector: "fixture", reason: "fixture" };
  } });
  await expect(agent.sign({ actionType: "api:call" })).rejects.toBeInstanceOf(DetectorBlockedError);
  expect(calls).toHaveLength(0);
  clearDetectors();
  await agent.sign({ actionType: "api:call" });
  expect(calls[0].tenant).toBe("A");
});

it("retains unsalted hashing when a later default supplies a salt", async () => {
  configure("A", { mode: "hash-only" });
  const agent = Agent.attach(payload("A"));
  configure("B", { mode: "hash-only", orgSalt: new Uint8Array([1]) });
  await agent.sign({ actionType: "api:call" });
  expect(calls[0].body.hash_algo).toBe("sha256");
  expect(calls[0].body.hash).toBe(`sha256:${createHash("sha256").update(canonicalizeAction("api:call", {})).digest("hex")}`);
});

it.each([undefined, "fixture suspension"])("preserves suspended-agent preflight: %s", async (reason) => {
  configure("A");
  const agent = Agent.attach(payload("A"));
  configure("B");
  vi.mocked(fetch).mockImplementation(async (input, options) => {
    expect(new Headers(options?.headers).get("x-api-key")).toBe(tenants.A.apiKey);
    expect(String(input)).toMatch(/^https:\/\/tenant-a\.invalid\//);
    return String(input).endsWith("/status")
      ? Response.json({ suspended: true, suspended_reason: reason }) : Response.json([]);
  });
  const result = await agent.preflight("api:call");
  expect(result).toMatchObject({ cleared: false, agentActive: false, checksComplete: true });
  expect(result.explanation).toBe(`Blocked: agent is suspended${reason ? ` (${reason})` : ""}`);
});

it.each(["network", "rate-limit", "retry-success"])("retains bound retry transport: %s", async (failure) => {
  configure("A");
  const agent = Agent.attach(payload("A"));
  vi.useFakeTimers();
  let attempts = 0;
  vi.mocked(fetch).mockImplementation(async (input, options) => {
    expect(String(input)).toMatch(/^https:\/\/tenant-a\.invalid\//);
    expect(new Headers(options?.headers).get("x-api-key")).toBe(tenants.A.apiKey);
    expect(JSON.parse(String(options?.body)).context).toEqual({ private: "fixture" });
    configure("B", { mode: "hash-only" });
    attempts++;
    if (failure === "network") throw new Error("fixture network failure");
    if (failure === "rate-limit") return Response.json({}, { status: 429 });
    return Response.json(attempts < 3 ? {} : signResponse, { status: attempts < 3 ? 503 : 200 });
  });
  const pending = agent.sign({ actionType: "api:call", context: { private: "fixture" } });
  const checked = failure === "retry-success" ? expect(pending).resolves.toMatchObject({ signatureId: "sig_fixture" })
    : expect(pending).rejects.toThrow(failure === "network" ? "Network error: fixture network failure" : "Rate limit exceeded");
  await vi.runAllTimersAsync();
  await checked;
  expect(attempts).toBe(3);
});

it("preserves non-JSON error details and empty successful responses", async () => {
  configure("A");
  const agent = Agent.attach(payload("A"));
  configure("B");
  vi.mocked(fetch).mockResolvedValueOnce(new Response("fixture refused", { status: 400 }));
  await expect(agent.revoke()).rejects.toThrow();
  vi.mocked(fetch).mockResolvedValueOnce(new Response(null, { status: 204 }));
  await expect(agent.revoke()).resolves.toBeUndefined();
  vi.mocked(fetch).mockResolvedValueOnce(new Response("not-json", { status: 200 }));
  await expect(request("GET", "/fixture")).resolves.toBeUndefined();
});
