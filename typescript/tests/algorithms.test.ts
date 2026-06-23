// Algorithm agility tests (AG3, AG4). Pins client-side validation plus the
// node:crypto Ed25519/ES256 keypair, sign, and verify helpers, and the
// cloud 400 reject for local-only algorithms on Agent.create.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  Agent,
  APIError,
  AsqavError,
  LOCAL_SIGNING_ALGORITHMS,
  SUPPORTED_ALGORITHMS,
  _resetForTests,
  generateKeypair,
  init,
  isLocalSigningAlgorithm,
  isSupportedAlgorithm,
  signMessage,
  verifyMessage,
} from "../src/index.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("algorithm registry", () => {
  it("includes ML-DSA, Ed25519, and ES256", () => {
    expect(SUPPORTED_ALGORITHMS).toContain("ml-dsa-65");
    expect(SUPPORTED_ALGORITHMS).toContain("ed25519");
    expect(SUPPORTED_ALGORITHMS).toContain("es256");
  });

  it("isSupportedAlgorithm matches the registry", () => {
    for (const a of SUPPORTED_ALGORITHMS) {
      expect(isSupportedAlgorithm(a)).toBe(true);
    }
    expect(isSupportedAlgorithm("rsa")).toBe(false);
    expect(isSupportedAlgorithm("")).toBe(false);
  });

  it("isLocalSigningAlgorithm rejects ML-DSA (server-side only)", () => {
    expect(isLocalSigningAlgorithm("ed25519")).toBe(true);
    expect(isLocalSigningAlgorithm("es256")).toBe(true);
    expect(isLocalSigningAlgorithm("ml-dsa-65")).toBe(false);
  });
});

describe("Agent.create algorithm validation", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("rejects unknown algorithm without making a request", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    await expect(
      Agent.create({ name: "x", algorithm: "rsa-2048" }),
    ).rejects.toThrow(AsqavError);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  // Pins CLIENT-SIDE validation passthrough only: every member clears local
  // validation and is forwarded. Cloud acceptance is a separate matter (the
  // cloud signs only ml-dsa-{44,65,87}; see the 400-mock test below).
  it("passes every SUPPORTED_ALGORITHMS member past client-side validation", async () => {
    for (const alg of SUPPORTED_ALGORITHMS) {
      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse({
          agent_id: "agt_1",
          name: "x",
          public_key: "pk",
          key_id: "kid",
          algorithm: alg,
          capabilities: [],
          created_at: "t",
        }),
      );
      await expect(Agent.create({ name: "x", algorithm: alg })).resolves.toBeDefined();
      const [, calledInit] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(JSON.parse(calledInit.body as string).algorithm).toBe(alg);
      fetchSpy.mockRestore();
    }
  });

  // Pins the docstring: the rejection message enumerates all five.
  it("rejection message lists exactly SUPPORTED_ALGORITHMS", async () => {
    await expect(
      Agent.create({ name: "x", algorithm: "rsa-2048" }),
    ).rejects.toThrow(SUPPORTED_ALGORITHMS.join(", "));
  });

  // Mocked fetch = client-side passthrough only. ed25519 clears local
  // validation and is forwarded verbatim. This does NOT prove the live cloud
  // accepts it (it returns 400; see the next test).
  it("forwards ed25519 past client-side validation (mocked transport)", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        agent_id: "agt_1",
        name: "x",
        public_key: "pk",
        key_id: "kid",
        algorithm: "ed25519",
        capabilities: [],
        created_at: "t",
      }),
    );
    await Agent.create({ name: "x", algorithm: "ed25519" });
    const [, calledInit] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(calledInit.body as string).algorithm).toBe("ed25519");
  });

  // Same as above for es256: client passthrough, not a cloud-acceptance claim.
  it("forwards es256 past client-side validation (mocked transport)", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        agent_id: "agt_1",
        name: "x",
        public_key: "pk",
        key_id: "kid",
        algorithm: "es256",
        capabilities: [],
        created_at: "t",
      }),
    );
    await Agent.create({ name: "x", algorithm: "es256" });
    const [, calledInit] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(calledInit.body as string).algorithm).toBe("es256");
  });

  // Pins the real cloud reality: ed25519 clears client validation, but the
  // cloud signs only ml-dsa-{44,65,87} and returns 400. The SDK surfaces it
  // as an APIError with statusCode 400 (index.ts create comment + READMEs).
  it("surfaces the cloud 400 when ed25519 reaches the cloud", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(
        { error: "ed25519 is local-keygen only; cloud signs ml-dsa-{44,65,87}" },
        400,
      ),
    );
    const err = await Agent.create({ name: "x", algorithm: "ed25519" }).catch(
      (e: unknown) => e,
    );
    expect(err).toBeInstanceOf(APIError);
    expect((err as APIError).statusCode).toBe(400);
  });
});

describe("local Ed25519 keypair + sign + verify round-trip", () => {
  it("signs canonical bytes and verifies them", () => {
    const kp = generateKeypair("ed25519");
    expect(kp.algorithm).toBe("ed25519");
    expect(kp.privateKeyPkcs8B64.length).toBeGreaterThan(0);
    expect(kp.publicKeySpkiB64.length).toBeGreaterThan(0);

    const message = new TextEncoder().encode("hello asqav");
    const sig = signMessage("ed25519", kp.privateKeyPkcs8B64, message);
    expect(typeof sig).toBe("string");

    expect(verifyMessage("ed25519", kp.publicKeySpkiB64, message, sig)).toBe(true);
    // Tamper detection.
    const wrong = new TextEncoder().encode("hello asqav!");
    expect(verifyMessage("ed25519", kp.publicKeySpkiB64, wrong, sig)).toBe(false);
  });
});

describe("local ES256 keypair + sign + verify round-trip", () => {
  it("signs canonical bytes and verifies them in IEEE-P1363 form", () => {
    const kp = generateKeypair("es256");
    expect(kp.algorithm).toBe("es256");

    const message = new TextEncoder().encode("compliance receipt");
    const sig = signMessage("es256", kp.privateKeyPkcs8B64, message);
    // IEEE-P1363 / raw r||s for P-256 = 64 bytes = 88 base64 chars
    // (with optional padding).
    const decoded = Buffer.from(sig, "base64");
    expect(decoded.length).toBe(64);

    expect(verifyMessage("es256", kp.publicKeySpkiB64, message, sig)).toBe(true);
    const wrong = new TextEncoder().encode("compliance receipt!");
    expect(verifyMessage("es256", kp.publicKeySpkiB64, wrong, sig)).toBe(false);
  });
});

describe("local signing rejects ML-DSA", () => {
  it("generateKeypair only knows about local algorithms", () => {
    for (const a of LOCAL_SIGNING_ALGORITHMS) {
      expect(() => generateKeypair(a)).not.toThrow();
    }
  });
});
