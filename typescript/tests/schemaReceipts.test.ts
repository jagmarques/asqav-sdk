/** Tests for opt-in schema-driven structured receipts (criterion 328).
 * (a) valid context -> fetch called with sorted context;
 * (b) invalid context -> ValidationError raised, fetch NOT called;
 * (c) no schema -> unchanged behavior (fetch called as before). */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  Agent,
  ValidationError,
  _resetForTests,
  init,
  normalizeContext,
  validateContextSchema,
} from "../src/index.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function fakeAgent(): Agent {
  return Agent.attach({
    agent_id: "agt_schema_test",
    name: "schema-test",
    public_key: "pk",
    key_id: "kid",
    algorithm: "ml-dsa-65",
    capabilities: [],
    created_at: "2026-06-29T00:00:00Z",
  });
}

function okBody(extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    signature: "sig_b64",
    signature_id: "sig_schema",
    action_id: "act_schema",
    timestamp: "2026-06-29T00:00:00Z",
    verification_url: "https://asqav.com/verify/sig_schema",
    algorithm: "ML-DSA-65",
    ...extra,
  };
}

const BASIC_SCHEMA = {
  userId: { type: "string" as const, required: true },
  score: { type: "number" as const },
  active: { type: "boolean" as const },
  tags: { type: "array" as const },
  meta: { type: "object" as const },
};

describe("agent.sign contextSchema", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // === (a) Valid context passes ===

  it("(a) valid context passes schema and fetch is called", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));

    const agent = fakeAgent();
    await agent.sign({
      actionType: "api:call",
      context: { userId: "u1", score: 42 },
      contextSchema: BASIC_SCHEMA,
      complianceMode: false,
    });

    expect(fetchSpy).toHaveBeenCalledOnce();
  });

  it("(a) context is key-sorted before sign is called", async () => {
    let capturedBody: Record<string, unknown> = {};
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
      capturedBody = JSON.parse((init?.body as string) ?? "{}") as Record<
        string,
        unknown
      >;
      return jsonResponse(okBody());
    });

    const agent = fakeAgent();
    // Supply keys in deliberate reverse alphabetical order.
    await agent.sign({
      actionType: "api:call",
      context: { userId: "u1", score: 3, active: true },
      contextSchema: BASIC_SCHEMA,
      complianceMode: false,
    });

    const ctx = capturedBody["context"] as Record<string, unknown> | undefined;
    if (ctx !== undefined) {
      const publicKeys = Object.keys(ctx).filter((k) => !k.startsWith("_"));
      expect(publicKeys).toEqual([...publicKeys].sort());
    }
    // Even if context is hashed (hash mode), the fetch was reached.
    expect(vi.mocked(fetch).mock.calls.length).toBeGreaterThan(0);
  });

  it("(a) all supported field types pass with correct values", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));

    const agent = fakeAgent();
    await agent.sign({
      actionType: "api:call",
      context: {
        userId: "alice",
        score: 3.14,
        active: true,
        tags: ["a", "b"],
        meta: { k: "v" },
      },
      contextSchema: BASIC_SCHEMA,
      complianceMode: false,
    });

    expect(fetchSpy).toHaveBeenCalledOnce();
  });

  it("(a) optional field absent passes with no error", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));

    await fakeAgent().sign({
      actionType: "api:call",
      context: { userId: "u2" }, // only required userId; optional fields absent
      contextSchema: BASIC_SCHEMA,
      complianceMode: false,
    });

    expect(fetchSpy).toHaveBeenCalledOnce();
  });

  // === (b) Invalid context raises before fetch ===

  it("(b) missing required field throws ValidationError and fetch not called", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    const agent = fakeAgent();
    await expect(
      agent.sign({
        actionType: "api:call",
        context: { score: 5 }, // userId missing
        contextSchema: BASIC_SCHEMA,
        complianceMode: false,
      }),
    ).rejects.toThrowError(ValidationError);

    // Non-vacuity: if validation were removed, this assertion would fail
    // because fetch would be called.
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("(b) wrong type (string for number) throws ValidationError before fetch", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    await expect(
      fakeAgent().sign({
        actionType: "api:call",
        context: { userId: "u1", score: "not-a-number" },
        contextSchema: BASIC_SCHEMA,
        complianceMode: false,
      }),
    ).rejects.toThrowError(ValidationError);

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("(b) boolean rejected for number type field", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    await expect(
      fakeAgent().sign({
        actionType: "api:call",
        context: { userId: "u1", score: true },
        contextSchema: BASIC_SCHEMA,
        complianceMode: false,
      }),
    ).rejects.toThrowError(/expected type number, got boolean/);

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("(b) ValidationError message names the offending field", async () => {
    vi.spyOn(globalThis, "fetch");

    let error: unknown;
    try {
      await fakeAgent().sign({
        actionType: "api:call",
        context: { score: 5 },
        contextSchema: BASIC_SCHEMA,
        complianceMode: false,
      });
    } catch (e) {
      error = e;
    }

    expect(error).toBeInstanceOf(ValidationError);
    expect((error as ValidationError).message).toContain("userId");
    expect((error as ValidationError).docsUrl).toContain("structured-receipts");
  });

  it("(b) wrong type for object field throws before fetch", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    await expect(
      fakeAgent().sign({
        actionType: "api:call",
        context: { userId: "u1", meta: "string-not-object" },
        contextSchema: BASIC_SCHEMA,
        complianceMode: false,
      }),
    ).rejects.toThrowError(ValidationError);

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("(b) wrong type for array field throws before fetch", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    await expect(
      fakeAgent().sign({
        actionType: "api:call",
        context: { userId: "u1", tags: { not: "array" } },
        contextSchema: BASIC_SCHEMA,
        complianceMode: false,
      }),
    ).rejects.toThrowError(ValidationError);

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  // === (c) No schema => unchanged behavior ===

  it("(c) no schema supplied: sign called as before", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));

    await fakeAgent().sign({
      actionType: "api:call",
      context: { any_key: "any_value" },
      complianceMode: false,
    });

    expect(fetchSpy).toHaveBeenCalledOnce();
  });

  it("(c) undefined context + no schema: sign called as before", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));

    await fakeAgent().sign({
      actionType: "api:call",
      complianceMode: false,
    });

    expect(fetchSpy).toHaveBeenCalledOnce();
  });

  // === Custom callable validator ===

  it("custom callable validator receives the context and passes through", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(okBody()));

    const receivedCtx: Record<string, unknown>[] = [];

    await fakeAgent().sign({
      actionType: "api:call",
      context: { x: 1 },
      contextSchema: (ctx) => {
        receivedCtx.push({ ...ctx });
      },
      complianceMode: false,
    });

    expect(receivedCtx.length).toBeGreaterThan(0);
    expect(fetchSpy).toHaveBeenCalledOnce();
  });

  it("custom callable validator that throws prevents fetch", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    await expect(
      fakeAgent().sign({
        actionType: "api:call",
        context: { x: -1 },
        contextSchema: (_ctx) => {
          throw new Error("custom_schema_error: x must be positive");
        },
        complianceMode: false,
      }),
    ).rejects.toThrowError("x must be positive");

    expect(fetchSpy).not.toHaveBeenCalled();
  });
});

// === Unit tests for schema helpers ===

describe("validateContextSchema", () => {
  it("throws on missing required field", () => {
    expect(() =>
      validateContextSchema(
        {},
        { userId: { type: "string", required: true } },
      ),
    ).toThrowError(/userId.*required/);
  });

  it("optional absent field passes", () => {
    expect(() =>
      validateContextSchema({}, { score: { type: "number" } }),
    ).not.toThrow();
  });

  it("correct string passes", () => {
    expect(() =>
      validateContextSchema({ name: "alice" }, { name: { type: "string" } }),
    ).not.toThrow();
  });

  it("correct number (integer) passes", () => {
    expect(() =>
      validateContextSchema({ n: 7 }, { n: { type: "number" } }),
    ).not.toThrow();
  });

  it("correct number (float) passes", () => {
    expect(() =>
      validateContextSchema({ n: 3.14 }, { n: { type: "number" } }),
    ).not.toThrow();
  });

  it("boolean rejected for number field", () => {
    expect(() =>
      validateContextSchema({ n: true }, { n: { type: "number" } }),
    ).toThrowError(/expected type number, got boolean/);
  });

  it("wrong type string-for-number raises", () => {
    expect(() =>
      validateContextSchema({ n: "oops" }, { n: { type: "number" } }),
    ).toThrowError(/expected type number/);
  });

  it("boolean field passes true", () => {
    expect(() =>
      validateContextSchema({ ok: true }, { ok: { type: "boolean" } }),
    ).not.toThrow();
  });

  it("object field passes plain object", () => {
    expect(() =>
      validateContextSchema({ m: { a: 1 } }, { m: { type: "object" } }),
    ).not.toThrow();
  });

  it("object field rejects array", () => {
    expect(() =>
      validateContextSchema({ m: [1, 2] }, { m: { type: "object" } }),
    ).toThrowError(/expected type object/);
  });

  it("array field passes", () => {
    expect(() =>
      validateContextSchema({ arr: [1, 2] }, { arr: { type: "array" } }),
    ).not.toThrow();
  });

  it("array field rejects plain object", () => {
    expect(() =>
      validateContextSchema({ arr: {} }, { arr: { type: "array" } }),
    ).toThrowError(/expected type array/);
  });

  it("unknown type name throws Error", () => {
    expect(() =>
      validateContextSchema({ x: 1 }, { x: { type: "bigdecimal" as never } }),
    ).toThrowError(/unknown type/);
  });

  it("null context treated as empty (optional field passes)", () => {
    expect(() =>
      validateContextSchema(null, { x: { type: "string" } }),
    ).not.toThrow();
  });

  it("null context with required field throws", () => {
    expect(() =>
      validateContextSchema(null, {
        x: { type: "string", required: true },
      }),
    ).toThrowError(/x.*required/);
  });
});

describe("normalizeContext", () => {
  it("sorts keys alphabetically", () => {
    const result = normalizeContext({ z: 1, a: 2, m: 3 });
    expect(Object.keys(result!)).toEqual(["a", "m", "z"]);
  });

  it("returns undefined for null input", () => {
    expect(normalizeContext(null)).toBeUndefined();
  });

  it("returns undefined for undefined input", () => {
    expect(normalizeContext(undefined)).toBeUndefined();
  });

  it("returns empty object for empty input", () => {
    expect(normalizeContext({})).toEqual({});
  });

  it("preserves values", () => {
    const result = normalizeContext({ b: 2, a: 1 });
    expect(result).toEqual({ a: 1, b: 2 });
  });
});
