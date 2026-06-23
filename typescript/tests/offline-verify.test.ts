/**
 * Offline receipt verification helpers: fetchJwks + verifyReceiptOffline.
 *
 * All network tests are guarded: fetch is mocked where needed, and the main
 * verifyReceiptOffline tests pass NO network calls (all data comes from the
 * conformance-vector files on disk).
 */

import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";

import { ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";

import { fetchJwks, verifyReceiptOffline } from "../src/index.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { verify as oracleVerify } from "../src/verifier/core.js";
import { b64decode } from "../src/verifier/vrShim.js";

const VECTORS = resolve(__dirname, "..", "..", "verifier", "conformance-vectors");
const PASS_DIR = join(VECTORS, "asqav-01-genesis-permit");
const TAMPER_DIR = join(VECTORS, "asqav-04-tamper-sig");

function loadJson(dir: string, file: string): Record<string, unknown> {
  return JSON.parse(readFileSync(join(dir, file), "utf-8")) as Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// fetchJwks: exported, wires to the network
// ---------------------------------------------------------------------------

describe("fetchJwks", () => {
  it("is exported from the top-level SDK module", () => {
    expect(typeof fetchJwks).toBe("function");
  });

  it("calls fetch and returns parsed JSON", async () => {
    const fakeJwks = { keys: [{ kid: "test" }] };
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(fakeJwks),
    });
    vi.stubGlobal("fetch", mockFetch);
    const result = await fetchJwks();
    expect(result).toEqual(fakeJwks);
    expect(mockFetch).toHaveBeenCalledWith(
      "https://api.asqav.com/.well-known/jwks.json",
      expect.objectContaining({ headers: expect.any(Object) }),
    );
    vi.unstubAllGlobals();
  });

  it("throws on HTTP error", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: false, status: 404 });
    vi.stubGlobal("fetch", mockFetch);
    await expect(fetchJwks()).rejects.toThrow("HTTP 404");
    vi.unstubAllGlobals();
  });

  it("accepts a custom URL", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ keys: [] }),
    });
    vi.stubGlobal("fetch", mockFetch);
    await fetchJwks("https://custom.example.com/jwks.json");
    expect(mockFetch).toHaveBeenCalledWith(
      "https://custom.example.com/jwks.json",
      expect.any(Object),
    );
    vi.unstubAllGlobals();
  });
});

// ---------------------------------------------------------------------------
// verifyReceiptOffline: Ed25519 (conformance vector, no network)
// ---------------------------------------------------------------------------

describe("verifyReceiptOffline - Ed25519 path (no network)", () => {
  it("returns PASS for a valid Ed25519-signed receipt", () => {
    const receipt = loadJson(PASS_DIR, "receipt.json");
    const jwks = loadJson(PASS_DIR, "jwks.json");
    const result = verifyReceiptOffline(receipt, jwks);
    expect(result.verdict).toBe("PASS");
    const sigAxis = result.axes.find((a) => a.axis === "signature");
    expect(sigAxis?.result).toBe("PASS");
  });

  it("returns the correct fmt for asqav-native receipts", () => {
    const receipt = loadJson(PASS_DIR, "receipt.json");
    const jwks = loadJson(PASS_DIR, "jwks.json");
    const result = verifyReceiptOffline(receipt, jwks);
    expect(result.fmt).toBe("asqav-native");
  });

  it("result has axes with expected shape", () => {
    const receipt = loadJson(PASS_DIR, "receipt.json");
    const jwks = loadJson(PASS_DIR, "jwks.json");
    const result = verifyReceiptOffline(receipt, jwks);
    expect(Array.isArray(result.axes)).toBe(true);
    for (const ax of result.axes) {
      expect(ax).toHaveProperty("axis");
      expect(ax).toHaveProperty("result");
      expect(ax).toHaveProperty("note");
    }
  });
});

// ---------------------------------------------------------------------------
// verifyReceiptOffline: FAIL on tampered receipts (no network)
// ---------------------------------------------------------------------------

describe("verifyReceiptOffline - tamper detection (no network)", () => {
  it("returns FAIL for a receipt with tampered decision field", () => {
    const receipt = loadJson(TAMPER_DIR, "receipt.json");
    const jwks = loadJson(TAMPER_DIR, "jwks.json");
    const result = verifyReceiptOffline(receipt, jwks);
    expect(result.verdict).toBe("FAIL");
    const sigAxis = result.axes.find((a) => a.axis === "signature");
    expect(sigAxis?.result).toBe("FAIL");
  });

  it("returns FAIL when payload is mutated after signing", () => {
    const receipt = loadJson(PASS_DIR, "receipt.json");
    const jwks = loadJson(PASS_DIR, "jwks.json");
    const tampered = JSON.parse(JSON.stringify(receipt)) as Record<string, unknown>;
    (tampered.payload as Record<string, unknown>).decision = "deny";
    const result = verifyReceiptOffline(tampered, jwks);
    expect(result.verdict).toBe("FAIL");
  });
});

// ---------------------------------------------------------------------------
// verifyReceiptOffline: missing key in JWKS (no network)
// ---------------------------------------------------------------------------

describe("verifyReceiptOffline - missing key (no network)", () => {
  it("never returns PASS when key is absent from JWKS", () => {
    const receipt = loadJson(PASS_DIR, "receipt.json");
    const result = verifyReceiptOffline(receipt, { keys: [] });
    expect(result.verdict).not.toBe("PASS");
    // Signature must be SKIPPED (key not found) not FAIL
    const sigAxis = result.axes.find((a) => a.axis === "signature");
    expect(sigAxis?.result).toBe("SKIPPED");
  });
});

// ---------------------------------------------------------------------------
// ML-DSA-65 path: sign in Node with noble, verify with the oracle
// ---------------------------------------------------------------------------

// NOTE: ML-DSA-65 tests below are same-library interop round-trips (noble sign +
// noble verify). They confirm the wiring works but do NOT prove interop with
// real Asqav-cloud ML-DSA-65 signatures. A real-cloud known-answer test vector
// (payload-mode prod receipt) is a documented follow-up.

describe("verifyReceiptOffline - ML-DSA-65 path (@noble/post-quantum)", () => {
  it("verifies a real ML-DSA-65 receipt PASS (noble sign + noble verify)", () => {
    const { publicKey, secretKey } = ml_dsa65.keygen();
    const msg = new Uint8Array([1, 2, 3, 4, 5]);
    const sig = ml_dsa65.sign(msg, secretKey);
    const ok = ml_dsa65.verify(sig, msg, publicKey);
    expect(ok).toBe(true);
    const bad = ml_dsa65.verify(sig, new Uint8Array([9, 9, 9]), publicKey);
    expect(bad).toBe(false);
  });

  it("verifies a well-formed ML-DSA-65 receipt envelope through the oracle", () => {
    // NOTE: the oracle uses the asqav-native adapter whose signing_input
    // is asqavJcs (canonical bytes). We use a simpler approach here:
    // generate with JSON.stringify+sort which is close but not JCS-exact
    // for complex payloads. The round-trip test below uses the oracle's
    // actual canonical bytes to generate the signature.
    const { publicKey, secretKey } = ml_dsa65.keygen();
    const pk_b64 = Buffer.from(publicKey).toString("base64");
    const kid = "test-mldsa-roundtrip-01";

    const payload: Record<string, unknown> = {
      type: "protectmcp:decision",
      issued_at: "2026-06-19T00:00:00.000000Z",
      issuer_id: kid,
      action_ref: "sha256:" + "a".repeat(64),
      payload_digest: { hash: "b".repeat(64), size: 128 },
      policy_digest: "sha256:" + "c".repeat(64),
      previousReceiptHash: "0".repeat(64),
      decision: "allow",
    };
    const envelope: Record<string, unknown> = {
      payload,
      signature: { alg: "ML-DSA-65", kid, sig: "" },
      anchors: [],
    };

    // Use the adapter's signing_input to get the exact canonical bytes.
    const ad = ADAPTERS.find((a) => a.detect(envelope))!;
    expect(ad.name).toBe("asqav-native");
    const canonicalBytes = ad.signingInput(envelope);
    const sig = ml_dsa65.sign(Buffer.from(canonicalBytes), secretKey);
    const sig_b64 = Buffer.from(sig).toString("base64");

    (envelope.signature as Record<string, unknown>).sig = sig_b64;
    const jwks: Record<string, unknown> = {
      keys: [{ kid, issuer_id: kid, alg: "ML-DSA-65", status: "active", public_key: pk_b64 }],
    };

    const result = oracleVerify(envelope, ADAPTERS, jwks);
    expect(result.verdict).toBe("PASS");
    const sigAxis = result.axes.find((a) => a.axis === "signature");
    expect(sigAxis?.result).toBe("PASS");
  });

  it("returns FAIL for a tampered ML-DSA-65 receipt", () => {
    const { publicKey, secretKey } = ml_dsa65.keygen();
    const pk_b64 = Buffer.from(publicKey).toString("base64");
    const kid = "test-mldsa-tamper-01";

    const payload: Record<string, unknown> = {
      type: "protectmcp:decision",
      issued_at: "2026-06-19T00:00:00.000000Z",
      issuer_id: kid,
      action_ref: "sha256:" + "a".repeat(64),
      payload_digest: { hash: "b".repeat(64), size: 128 },
      policy_digest: "sha256:" + "c".repeat(64),
      previousReceiptHash: "0".repeat(64),
      decision: "allow",
    };
    const envelope: Record<string, unknown> = {
      payload,
      signature: { alg: "ML-DSA-65", kid, sig: "" },
      anchors: [],
    };

    const ad = ADAPTERS.find((a) => a.detect(envelope))!;
    const canonicalBytes = ad.signingInput(envelope);
    const sig = ml_dsa65.sign(Buffer.from(canonicalBytes), secretKey);
    const sig_b64 = Buffer.from(sig).toString("base64");
    (envelope.signature as Record<string, unknown>).sig = sig_b64;

    // Tamper: flip the decision AFTER signing
    (envelope.payload as Record<string, unknown>).decision = "deny";

    const jwks: Record<string, unknown> = {
      keys: [{ kid, issuer_id: kid, alg: "ML-DSA-65", status: "active", public_key: pk_b64 }],
    };
    const result = verifyReceiptOffline(envelope, jwks);
    expect(result.verdict).toBe("FAIL");
    const sigAxis = result.axes.find((a) => a.axis === "signature");
    expect(sigAxis?.result).toBe("FAIL");
  });

  // Real-cloud ML-DSA-65 payload-mode KAT (asqav-06-mldsa65-payload-prod).
  // Uses a receipt + JWKS minted from api.asqav.com with mode=full-payload.
  const KAT_DIR = join(VECTORS, "asqav-06-mldsa65-payload-prod");

  it("verifies a real-cloud ML-DSA-65 payload-mode receipt (KAT - signature axis must be PASS)", () => {
    const receipt = loadJson(KAT_DIR, "receipt.json");
    const jwks = loadJson(KAT_DIR, "jwks.json");
    const result = verifyReceiptOffline(receipt, jwks);
    expect(result.verdict).toBe("PASS");
    const sigAxis = result.axes.find((a) => a.axis === "signature");
    // Must be PASS - SKIPPED or INCOMPLETE means ML-DSA-65 check was bypassed.
    expect(sigAxis?.result).toBe("PASS");
  });

  it("returns FAIL for a tampered real-cloud ML-DSA-65 KAT receipt (anti-vacuous)", () => {
    const receipt = loadJson(KAT_DIR, "receipt.json");
    const jwks = loadJson(KAT_DIR, "jwks.json");
    const tampered = JSON.parse(JSON.stringify(receipt)) as Record<string, unknown>;
    // Flip one payload field - ML-DSA sig over canonical bytes must not match.
    (tampered.payload as Record<string, unknown>).decision = "deny";
    const result = verifyReceiptOffline(tampered, jwks);
    expect(result.verdict).toBe("FAIL");
    const sigAxis = result.axes.find((a) => a.axis === "signature");
    expect(sigAxis?.result).toBe("FAIL");
  });

  it("returns FAIL for a corrupted-sig real-cloud ML-DSA-65 KAT receipt (anti-vacuous)", () => {
    const receipt = loadJson(KAT_DIR, "receipt.json");
    const jwks = loadJson(KAT_DIR, "jwks.json");
    const tampered = JSON.parse(JSON.stringify(receipt)) as Record<string, unknown>;
    const sig = tampered.signature as Record<string, unknown>;
    // Zero-out first 16 bytes of the signature - will not verify.
    const rawSig = b64decode(sig.sig as string);
    const bad = new Uint8Array(rawSig.length);
    bad.set(rawSig);
    bad.fill(0, 0, 16);
    sig.sig = Buffer.from(bad).toString("base64");
    const result = verifyReceiptOffline(tampered, jwks);
    expect(result.verdict).toBe("FAIL");
    const sigAxis = result.axes.find((a) => a.axis === "signature");
    expect(sigAxis?.result).toBe("FAIL");
  });
});
