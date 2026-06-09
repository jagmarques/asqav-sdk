/**
 * Every outbound API call from Node carries the asqav-js User-Agent.
 *
 * The api.asqav.com edge runs a Cloudflare browser-integrity check that
 * 403s anonymous default agents; the SDK must identify itself. Browsers
 * forbid setting User-Agent on fetch, so the helper returns {} there.
 */

import { readFileSync } from "node:fs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { _resetForTests, Agent, init } from "../src/index.js";
import { SDK_VERSION, USER_AGENT, userAgentHeaders } from "../src/userAgent.js";

function mockJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("user agent", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("matches the asqav-js/<version> (+url) shape", () => {
    expect(USER_AGENT).toMatch(/^asqav-js\/\S+ \(\+https:\/\/www\.asqav\.com\)$/);
  });

  it("SDK_VERSION matches package.json version", () => {
    const pkgPath = new URL("../package.json", `file://${__filename}`);
    const pkg = JSON.parse(readFileSync(pkgPath, "utf8")) as { version: string };
    expect(SDK_VERSION).toBe(pkg.version);
  });

  it("userAgentHeaders sets the header under Node", () => {
    expect(userAgentHeaders()).toEqual({ "User-Agent": USER_AGENT });
  });

  it("API requests send the User-Agent header", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockJsonResponse({
        agent_id: "agt_1",
        name: "test",
        public_key: "pk",
        key_id: "kid",
        algorithm: "ml-dsa-65",
        capabilities: ["read"],
        created_at: "2026-01-01T00:00:00Z",
      }),
    );

    await Agent.create({ name: "test" });

    const headers = (fetchSpy.mock.calls[0][1] as RequestInit).headers as Record<string, string>;
    expect(headers["User-Agent"]).toBe(USER_AGENT);
  });
});
