import { createHmac } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  Agent, AsqavClient, AuthenticationError, _resetForTests, clearDetectors,
  clearHooks, init, registerBefore, registerDetector, request, type InitOptions,
} from "../src/index.js";

const options = (tenant: string): InitOptions => ({
  apiKey: `fixture-key-${tenant}-not-a-credential`,
  baseUrl: `https://client-${tenant.toLowerCase()}.invalid/api/v1`, mode: "full-payload",
});
type Call = { origin: string; key: string | null; path: string; body: Record<string, unknown> };
const calls: Call[] = [];
let intercept: ((call: Call) => Promise<void>) | undefined;
const canonical = '{"action_type":"api:call","context":{"private":"fixture"}}';
const digest = () => `sha256:${createHmac("sha256", Buffer.alloc(32, 7)).update(canonical).digest("hex")}`;

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}

beforeEach(() => {
  vi.stubEnv("ASQAV_MODE", "");
  vi.stubEnv("ASQAV_API_URL", undefined);
  vi.stubEnv("ASQAV_API_KEY", "fixture-env-key-not-a-credential");
  _resetForTests(); clearHooks(); clearDetectors();
  calls.length = 0; intercept = undefined;
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = new URL(String(input));
    expect(["client-a.invalid", "client-b.invalid", "client-z.invalid", "api.asqav.com"]).toContain(url.hostname);
    const call: Call = { origin: url.origin, key: new Headers(init?.headers).get("x-api-key"),
      path: url.pathname, body: JSON.parse(String(init?.body ?? "{}")) };
    calls.push(call); await intercept?.(call);
    return Response.json({ agent_id: "agt_fixture", name: "fixture", public_key: "fixture-public",
      key_id: "fixture-id", algorithm: "ml-dsa-65", capabilities: [], created_at: "2026-09-08",
      signature: "fixture-signature", signature_id: "sig_fixture", action_id: "act_fixture",
      timestamp: "2026-09-08", verification_url: "https://fixture.invalid" });
  });
});

afterEach(() => {
  vi.restoreAllMocks(); vi.unstubAllEnvs();
  _resetForTests(); clearHooks(); clearDetectors();
});

describe("explicit client connections", () => {
  it.each(["A", "B"])("creates %s lazily after another client without replacing defaults", async (first) => {
    const second = first === "A" ? "B" : "A";
    init(options("Z"));
    const a = new AsqavClient(options(first));
    const b = new AsqavClient(options(second));
    expect(calls).toHaveLength(0);
    const agent = await a.createAgent({ name: "fixture", algorithm: "ml-dsa-87", capabilities: ["api:call"] });
    expect(agent).toBeInstanceOf(Agent);
    expect(calls[0].body).toEqual({ name: "fixture", algorithm: "ml-dsa-87", capabilities: ["api:call"] });
    await agent.sign({ actionType: "api:call" });
    expect(calls.map((call) => call.key)).toEqual([options(first).apiKey, options(first).apiKey]);
    expect(calls.every((call) => call.origin === new URL(options(first).baseUrl!).origin)).toBe(true);
    await (await b.getAgent("agt_other")).sign({ actionType: "api:call" });
    expect(calls.slice(-2).every((call) => call.key === options(second).apiKey)).toBe(true);
    await request("GET", "/policies");
    expect(calls.at(-1)).toMatchObject({ key: options("Z").apiKey, origin: "https://client-z.invalid" });
    expect(Object.getOwnPropertyNames(a)).toEqual([]);
    expect(Object.getOwnPropertySymbols(a)).toEqual([]);
    expect(JSON.stringify(a)).toBe("{}");
    for (const value of [options(first).apiKey!, options(first).baseUrl!]) expect(JSON.stringify(agent)).not.toContain(value);
  });

  it("does not initialize static APIs when a client is constructed", async () => {
    const client = new AsqavClient(options("A"));
    await expect(Agent.create({ name: "fixture" })).rejects.toBeInstanceOf(AuthenticationError);
    await expect(Agent.get("agt_fixture")).rejects.toBeInstanceOf(AuthenticationError);
    expect(calls).toHaveLength(0);
    await client.getAgent("agt_fixture");
    expect(calls[0].key).toBe(options("A").apiKey);
  });

  it.each([undefined, "https://client-a.invalid/api/v1"])("captures its omitted base independently: %s", async (base) => {
    init(options("Z"));
    vi.stubEnv("ASQAV_API_URL", base);
    const client = new AsqavClient({ apiKey: options("A").apiKey });
    vi.stubEnv("ASQAV_API_URL", options("B").baseUrl);
    init(options("B"));
    await (await client.getAgent("agt_fixture")).sign({ actionType: "api:call" });
    const origin = base ? "https://client-a.invalid" : "https://api.asqav.com";
    expect(calls.every((call) => call.origin === origin && call.key === options("A").apiKey)).toBe(true);
    expect(calls.at(-1)?.body).toHaveProperty(base ? "context" : "hash");
    await new AsqavClient({ apiKey: options("B").apiKey }).getAgent("agt_second");
    expect(calls.at(-1)?.origin).toBe("https://client-b.invalid");
  });

  it("retains default init base inheritance while clients use explicit or environment bases", async () => {
    init(options("Z"));
    vi.stubEnv("ASQAV_API_URL", options("A").baseUrl);
    const client = new AsqavClient(options("B"));
    init({ apiKey: options("A").apiKey });
    await request("GET", "/policies");
    expect(calls[0]).toMatchObject({ origin: "https://client-z.invalid", key: options("A").apiKey });
    await client.getAgent("agt_fixture");
    expect(calls[1]).toMatchObject({ origin: "https://client-b.invalid", key: options("B").apiKey });
  });

  it("resolves environment credentials and mode when constructing without options", async () => {
    vi.stubEnv("ASQAV_API_URL", options("A").baseUrl);
    vi.stubEnv("ASQAV_MODE", "hash-only");
    const client = new AsqavClient();
    vi.stubEnv("ASQAV_MODE", "full-payload");
    vi.stubEnv("ASQAV_API_KEY", options("B").apiKey);
    await (await client.createAgent({ name: "fixture" })).sign({ actionType: "api:call" });
    expect(calls[0].key).toBe("fixture-env-key-not-a-credential");
    expect(calls.at(-1)?.body).toHaveProperty("hash");
    expect(calls.at(-1)?.body).not.toHaveProperty("context");
  });

  it.each(["createAgent", "getAgent"] as const)("retains %s binding across a held response", async (method) => {
    init(options("Z"));
    const a = new AsqavClient(options("A"));
    const b = new AsqavClient(options("B"));
    const entered = deferred(), release = deferred();
    intercept = async (call) => {
      if (call.key === options("A").apiKey && !call.path.endsWith("/sign")) {
        entered.resolve(); await release.promise;
      }
    };
    const pending = method === "createAgent" ? a.createAgent({ name: "fixture" }) : a.getAgent("agt_fixture");
    await entered.promise;
    await (await b.createAgent({ name: "other" })).sign({ actionType: "api:call" });
    init(options("Z")); release.resolve();
    await (await pending).sign({ actionType: "api:call" });
    expect(calls.map((call) => call.key)).toEqual(["A", "B", "B", "A"].map((tenant) => options(tenant).apiKey));
  });

  it.each([false, true])("owns options and salt bytes, including Buffer inputs: %s", async (buffer) => {
    const salt = buffer ? Buffer.alloc(32, 7) : new Uint8Array(32).fill(7);
    const supplied: InitOptions = { ...options("A"), mode: "hash-only", orgSalt: salt };
    const client = new AsqavClient(supplied);
    Object.assign(supplied, options("B")); salt.fill(9);
    init(options("Z"));
    const agent = await client.createAgent({ name: "fixture" });
    await agent.sign({ actionType: "api:call", context: { private: "fixture" } });
    expect(calls.at(-1)).toMatchObject({ key: options("A").apiKey, origin: "https://client-a.invalid" });
    expect(calls.at(-1)?.body).toMatchObject({ hash: digest(), hash_algo: "hmac-sha256", payload_size: Buffer.byteLength(canonical) });
    expect(calls.at(-1)?.body).not.toHaveProperty("context");
  });

  it("rejects invalid client configuration without partially replacing defaults", async () => {
    init(options("Z"));
    expect(() => new AsqavClient({ ...options("B"), mode: "invalid" as InitOptions["mode"] })).toThrow("mode must");
    expect(calls).toHaveLength(0);
    await request("GET", "/policies");
    expect(calls[0]).toMatchObject({ key: options("Z").apiKey, origin: "https://client-z.invalid" });
    await expect(new AsqavClient(options("A")).createAgent({ name: "fixture", algorithm: "fixture-unsupported" })).rejects.toThrow("unsupported_algorithm");
    expect(calls).toHaveLength(1);
  });

  it.each(["hook", "detector"])("keeps late %s behavior without sharing connection defaults", async (boundary) => {
    const a = new AsqavClient({ ...options("A"), mode: "hash-only", orgSalt: Buffer.alloc(32, 7) });
    const agent = await a.getAgent("agt_fixture");
    const entered = deferred(), release = deferred();
    const verdict = { allow: true, confidence: 1, labels: [], detector: "fixture", reason: "fixture" };
    if (boundary === "hook") registerBefore("*", () => { init(options("Z")); });
    else registerDetector({ name: "fixture", inspect: async () => { entered.resolve(); await release.promise; return verdict; } });
    const pending = agent.sign({ actionType: "api:call", context: { private: "fixture" } });
    if (boundary === "detector") { await entered.promise; init(options("Z")); release.resolve(); }
    await pending;
    expect(calls.at(-1)).toMatchObject({ key: options("A").apiKey, origin: "https://client-a.invalid" });
    expect(calls.at(-1)?.body.hash_algo).toBe("hmac-sha256");
    expect(calls.at(-1)?.body).not.toHaveProperty("context");
    if (boundary === "hook") expect(calls.at(-1)?.body.hash).toBe(digest());
  });
});
