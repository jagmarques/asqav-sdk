/**
 * Tests for the pluggable detector framework (criterion 331). Non-vacuity: dropping
 * the block-throw fails deny tests; dropping the _detectors stamp fails receipt tests.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  Agent,
  DetectorBlockedError,
  _resetForTests,
  clearDetectors,
  init,
  registerDetector,
  runDetectors,
} from "../src/index.js";
import type { DetectorPlugin, DetectorResult } from "../src/index.js";
import { OpaDetector } from "../src/extras/opa.js";
import { PresidioDetector } from "../src/extras/presidio.js";

// ─── Shared helpers ──────────────────────────────────────────────────────────

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function fakeAgent(): Agent {
  return Agent.attach({
    agent_id: "agt_det_test",
    name: "detector-test",
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
    signature_id: "sig_det",
    action_id: "act_det",
    timestamp: "2026-06-29T00:00:00Z",
    verification_url: "https://asqav.com/verify/sig_det",
    algorithm: "ML-DSA-65",
    ...extra,
  };
}

const allowDetector: DetectorPlugin = {
  name: "allow-detector",
  inspect(_actionType, _context): DetectorResult {
    return { allow: true, confidence: 0.9, labels: ["ok"], detector: "allow-detector", reason: "all clear" };
  },
};

const denyDetector: DetectorPlugin = {
  name: "deny-detector",
  inspect(_actionType, _context): DetectorResult {
    return { allow: false, confidence: 1.0, labels: ["pii", "ssn"], detector: "deny-detector", reason: "SSN detected" };
  },
};

const raisingDetector: DetectorPlugin = {
  name: "raising-detector",
  inspect(): DetectorResult {
    throw new Error("detector_internal_error");
  },
};

// ─── Setup / teardown ────────────────────────────────────────────────────────

beforeEach(() => {
  vi.restoreAllMocks();
  _resetForTests();
  clearDetectors();
  init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
});

afterEach(() => {
  clearDetectors();
  vi.restoreAllMocks();
});

// ─── Unit: runDetectors ───────────────────────────────────────────────────────

describe("runDetectors", () => {
  it("returns [] with no detectors registered", async () => {
    expect(await runDetectors("api:call", { x: 1 })).toEqual([]);
  });

  it("allow detector returns a record", async () => {
    registerDetector(allowDetector);
    const records = await runDetectors("api:call", { x: 1 });
    expect(records).toHaveLength(1);
    expect(records[0].allow).toBe(true);
    expect(records[0].detector).toBe("allow-detector");
  });

  it("deny detector throws DetectorBlockedError", async () => {
    registerDetector(denyDetector);
    await expect(runDetectors("api:call", { ssn: "123-45-6789" }))
      .rejects.toBeInstanceOf(DetectorBlockedError);
  });

  it("DetectorBlockedError has correct attributes", async () => {
    registerDetector(denyDetector);
    let err: DetectorBlockedError | undefined;
    try {
      await runDetectors("api:call", { ssn: "xxx" });
    } catch (e) {
      err = e as DetectorBlockedError;
    }
    expect(err).toBeDefined();
    expect(err!.detectorName).toBe("deny-detector");
    expect(err!.result.labels).toContain("pii");
    expect(err!.message).toContain("detector_blocked");
  });

  it("raising detector is fail-closed by default", async () => {
    registerDetector(raisingDetector);
    await expect(runDetectors("api:call", {}))
      .rejects.toBeInstanceOf(DetectorBlockedError);
    await expect(runDetectors("api:call", {}))
      .rejects.toMatchObject({ message: expect.stringContaining("detector_error_blocked") });
  });

  it("raising detector with failOpen proceeds", async () => {
    registerDetector(raisingDetector, { failOpen: true });
    const records = await runDetectors("api:call", {});
    expect(records).toHaveLength(1);
    expect(records[0].allow).toBe(true);
    expect(records[0].reason).toContain("fail-open");
  });

  it("deny stops after first deny (second detector never runs)", async () => {
    const ran: string[] = [];
    registerDetector({
      name: "first-deny",
      inspect() { ran.push("deny"); return { allow: false, confidence: 1, labels: [], detector: "first-deny", reason: "no" }; },
    });
    registerDetector({
      name: "second-allow",
      inspect() { ran.push("allow"); return { allow: true, confidence: 1, labels: [], detector: "second-allow", reason: "" }; },
    });
    await expect(runDetectors("api:call", {})).rejects.toBeInstanceOf(DetectorBlockedError);
    expect(ran).toEqual(["deny"]);
  });

  it("null context treated as empty", async () => {
    registerDetector(allowDetector);
    const records = await runDetectors("api:call", null);
    expect(records).toHaveLength(1);
  });
});

// ─── Integration: agent.sign path ────────────────────────────────────────────

describe("agent.sign with detectors", () => {
  it("allow detector: sign POST is called and _detectors in context", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    // First call: agent create (GET /agents/agt_det_test).
    // Since we use Agent.attach, only the sign POST happens.
    fetchSpy.mockResolvedValueOnce(jsonResponse(okBody()));

    registerDetector(allowDetector);
    const agent = fakeAgent();
    await agent.sign({ actionType: "api:call", context: { model: "gpt-4" }, complianceMode: false });

    expect(fetchSpy).toHaveBeenCalledOnce();
    const [, init] = fetchSpy.mock.calls[0];
    const body = JSON.parse((init as RequestInit).body as string);
    const ctx = body.context ?? {};
    expect(ctx._detectors).toBeDefined();
    expect(ctx._detectors[0].allow).toBe(true);
    expect(ctx._detectors[0].detector).toBe("allow-detector");
  });

  it("deny detector: sign throws DetectorBlockedError, no POST", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockResolvedValue(jsonResponse(okBody()));

    registerDetector(denyDetector);
    const agent = fakeAgent();
    await expect(
      agent.sign({ actionType: "api:call", context: { ssn: "xxx" }, complianceMode: false }),
    ).rejects.toBeInstanceOf(DetectorBlockedError);

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("fail-closed detector error blocks sign (no POST)", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockResolvedValue(jsonResponse(okBody()));

    registerDetector(raisingDetector);
    const agent = fakeAgent();
    await expect(
      agent.sign({ actionType: "api:call", context: {}, complianceMode: false }),
    ).rejects.toBeInstanceOf(DetectorBlockedError);

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("fail-open detector error: sign proceeds, error record in context", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockResolvedValueOnce(jsonResponse(okBody()));

    registerDetector(raisingDetector, { failOpen: true });
    const agent = fakeAgent();
    await agent.sign({ actionType: "api:call", context: { x: 1 }, complianceMode: false });

    expect(fetchSpy).toHaveBeenCalledOnce();
    const [, init] = fetchSpy.mock.calls[0];
    const body = JSON.parse((init as RequestInit).body as string);
    const ctx = body.context ?? {};
    expect(ctx._detectors[0].labels).toContain("error");
  });

  it("additive: no detector registered → _detectors absent from body", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockResolvedValueOnce(jsonResponse(okBody()));

    const agent = fakeAgent();
    await agent.sign({ actionType: "api:call", context: { model: "gpt-4" }, complianceMode: false });

    const [, init] = fetchSpy.mock.calls[0];
    const body = JSON.parse((init as RequestInit).body as string);
    const ctx = body.context ?? {};
    expect(ctx._detectors).toBeUndefined();
  });
});

// ─── Reference: OpaDetector ──────────────────────────────────────────────────

describe("OpaDetector", () => {
  it("returns allow=true when OPA responds {result: true}", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockResolvedValueOnce(jsonResponse({ result: true }));

    const det = new OpaDetector({ opaUrl: "http://fake-opa", policyPath: "test/allow" });
    const result = await det.inspect("api:call", { model: "x" });
    expect(result.allow).toBe(true);
  });

  it("returns allow=false when OPA responds {result: false}", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockResolvedValueOnce(jsonResponse({ result: false }));

    const det = new OpaDetector({ opaUrl: "http://fake-opa", policyPath: "test/allow" });
    const result = await det.inspect("api:call", {});
    expect(result.allow).toBe(false);
    expect(result.labels).toContain("opa_deny");
  });

  it("returns allow=false when result key is missing (fail-closed)", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockResolvedValueOnce(jsonResponse({}));

    const det = new OpaDetector();
    const result = await det.inspect("api:call", {});
    expect(result.allow).toBe(false);
  });

  it("throws on HTTP error", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockResolvedValueOnce(new Response(null, { status: 500 }));

    const det = new OpaDetector();
    await expect(det.inspect("api:call", {})).rejects.toThrow("opa_request_failed");
  });

  it("sends correct input envelope to OPA", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockResolvedValueOnce(jsonResponse({ result: true }));

    const det = new OpaDetector({ opaUrl: "http://opa:8181", policyPath: "my/policy" });
    await det.inspect("tool:call", { model: "gpt-4" });

    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe("http://opa:8181/v1/data/my/policy");
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.input.action_type).toBe("tool:call");
    expect(body.input.context).toEqual({ model: "gpt-4" });
  });
});

// ─── Reference: PresidioDetector ─────────────────────────────────────────────

describe("PresidioDetector", () => {
  it("returns allow=false with labels when Presidio finds PII", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockResolvedValueOnce(
      jsonResponse([{ entity_type: "US_SSN", score: 0.9, start: 0, end: 11 }]),
    );

    const det = new PresidioDetector({ url: "http://presidio:5002", threshold: 0.5 });
    const result = await det.inspect("api:call", { data: "SSN: 123-45-6789" });
    expect(result.allow).toBe(false);
    expect(result.labels).toContain("US_SSN");
    expect(result.confidence).toBeGreaterThanOrEqual(0.9);
  });

  it("returns allow=true when no high-confidence PII found", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockResolvedValueOnce(jsonResponse([]));

    const det = new PresidioDetector({ url: "http://presidio:5002" });
    const result = await det.inspect("api:call", { data: "hello world" });
    expect(result.allow).toBe(true);
  });

  it("returns allow=true when all entities are below threshold", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockResolvedValueOnce(
      jsonResponse([{ entity_type: "PERSON", score: 0.3, start: 0, end: 5 }]),
    );

    const det = new PresidioDetector({ threshold: 0.5 });
    const result = await det.inspect("api:call", { name: "Alice" });
    expect(result.allow).toBe(true);
  });

  it("throws on Presidio HTTP error", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockResolvedValueOnce(new Response(null, { status: 503 }));

    const det = new PresidioDetector({ url: "http://presidio:5002" });
    await expect(det.inspect("api:call", {})).rejects.toThrow("presidio_request_failed");
  });
});
