/** verify(id): public no-key verify-by-id with a locally reproduced chainHash. */

import { createHash } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";

import { canonicalJson, verify } from "../src/index.js";

const PAYLOAD = {
  type: "protectmcp:decision",
  issued_at: "2026-05-04T09:14:22.118Z",
  issuer_id: "00000000000000000098",
};

function cloudResponse(): Record<string, unknown> {
  return {
    signature_id: "sig_test_dict",
    agent_id: "agent_dict",
    agent_name: "dict-agent",
    algorithm: "ML-DSA-65",
    verified: true,
    verification_url: "https://www.asqav.com/verify/sig_test_dict",
    payload: PAYLOAD,
  };
}

function mockFetch(status: number, body: unknown) {
  return vi.fn().mockResolvedValue({
    status,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("verify (public, no key)", () => {
  it("returns a plain object with verified / agentId / verificationUrl", async () => {
    vi.stubGlobal("fetch", mockFetch(200, cloudResponse()));
    const r = await verify("sig_test_dict");
    expect(r.verified).toBe(true);
    expect(r.agentId).toBe("agent_dict");
    expect(r.algorithm).toBe("ML-DSA-65");
    expect(r.verificationUrl).toContain("sig_test_dict");
  });

  it("chainHash equals sha256(canonicalJson(payload))", async () => {
    vi.stubGlobal("fetch", mockFetch(200, cloudResponse()));
    const r = await verify("sig_test_dict");
    const expected = createHash("sha256")
      .update(canonicalJson(PAYLOAD))
      .digest("hex");
    expect(r.chainHash).toBe(expected);
  });

  it("chainHash is null when the payload is absent", async () => {
    const body = cloudResponse();
    delete body.payload;
    vi.stubGlobal("fetch", mockFetch(200, body));
    const r = await verify("sig_test_dict");
    expect(r.chainHash).toBeNull();
  });

  it("throws on a 404 (unknown signature id)", async () => {
    vi.stubGlobal("fetch", mockFetch(404, { detail: "Signature not found" }));
    await expect(verify("sig_missing")).rejects.toThrow();
  });
});
