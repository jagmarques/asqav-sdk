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
// verifyReceiptOffline: revoked-key receipts (CRIT-167 fix)
// ---------------------------------------------------------------------------

describe("verifyReceiptOffline - revoked key (CRIT-167, no network)", () => {
  const REVOKED_DIR = join(VECTORS, "asqav-07-revoked-key");

  it("returns FAIL for a receipt with a valid sig but revoked key status (corpus vector)", () => {
    const receipt = loadJson(REVOKED_DIR, "receipt.json");
    const jwks = loadJson(REVOKED_DIR, "jwks.json");
    const result = verifyReceiptOffline(receipt, jwks);
    expect(result.verdict).toBe("FAIL");
    const keyAxis = result.axes.find((a) => a.axis === "key_status");
    expect(keyAxis, "key_status axis must be present").toBeDefined();
    expect(keyAxis?.result).toBe("FAIL");
  });

  it("key_status axis note names the revoked status", () => {
    const receipt = loadJson(REVOKED_DIR, "receipt.json");
    const jwks = loadJson(REVOKED_DIR, "jwks.json");
    const result = verifyReceiptOffline(receipt, jwks);
    const keyAxis = result.axes.find((a) => a.axis === "key_status");
    expect(keyAxis?.note).toMatch(/revoked/i);
  });

  it("signature axis is PASS (sig is valid - only revocation triggers FAIL)", () => {
    const receipt = loadJson(REVOKED_DIR, "receipt.json");
    const jwks = loadJson(REVOKED_DIR, "jwks.json");
    const result = verifyReceiptOffline(receipt, jwks);
    const sigAxis = result.axes.find((a) => a.axis === "signature");
    expect(sigAxis?.result).toBe("PASS");
  });

  it("returns PASS for the same receipt when key status is active (anti-vacuous: revocation is what fails it)", () => {
    const receipt = loadJson(REVOKED_DIR, "receipt.json");
    const revokedJwks = loadJson(REVOKED_DIR, "jwks.json");
    // Promote the key to active so revocation is the only difference.
    const activeJwks = JSON.parse(JSON.stringify(revokedJwks)) as Record<string, unknown>;
    ((activeJwks.keys as Array<Record<string, unknown>>)[0]).status = "active";
    const result = verifyReceiptOffline(receipt, activeJwks);
    expect(result.verdict).toBe("PASS");
  });

  it("suspended key also FAILs the key_status axis (REVOKED_KEY_STATUSES parity)", () => {
    const receipt = loadJson(REVOKED_DIR, "receipt.json");
    const revokedJwks = loadJson(REVOKED_DIR, "jwks.json");
    const suspJwks = JSON.parse(JSON.stringify(revokedJwks)) as Record<string, unknown>;
    ((suspJwks.keys as Array<Record<string, unknown>>)[0]).status = "suspended";
    const result = verifyReceiptOffline(receipt, suspJwks);
    expect(result.verdict).toBe("FAIL");
    const keyAxis = result.axes.find((a) => a.axis === "key_status");
    expect(keyAxis?.result).toBe("FAIL");
  });

  it("compromised key also FAILs the key_status axis (REVOKED_KEY_STATUSES parity)", () => {
    const receipt = loadJson(REVOKED_DIR, "receipt.json");
    const revokedJwks = loadJson(REVOKED_DIR, "jwks.json");
    const compJwks = JSON.parse(JSON.stringify(revokedJwks)) as Record<string, unknown>;
    ((compJwks.keys as Array<Record<string, unknown>>)[0]).status = "compromised";
    const result = verifyReceiptOffline(receipt, compJwks);
    expect(result.verdict).toBe("FAIL");
    const keyAxis = result.axes.find((a) => a.axis === "key_status");
    expect(keyAxis?.result).toBe("FAIL");
  });

  it("revoked key with revoked_at AFTER issuance INCOMPLETE without anchor (c386 backdating fix)", () => {
    const receipt = loadJson(REVOKED_DIR, "receipt.json");
    const revokedJwks = loadJson(REVOKED_DIR, "jwks.json");
    // issued_at is 2026-06-01T12:00:00+00:00; revoked_at is after that.
    // The vector receipt ships anchors:[] so issued_at is self-attested.
    // Without an anchor a holder of the compromised key can backdate, so the
    // verdict must be INCOMPLETE, never a hiding PASS.
    const futureJwks = JSON.parse(JSON.stringify(revokedJwks)) as Record<string, unknown>;
    ((futureJwks.keys as Array<Record<string, unknown>>)[0]).revoked_at = "2026-12-01T00:00:00+00:00";
    const result = verifyReceiptOffline(receipt, futureJwks);
    expect(result.verdict).toBe("INCOMPLETE");
    const keyAxis = result.axes.find((a) => a.axis === "key_status");
    expect(keyAxis?.result).toBe("SKIPPED");
    expect(keyAxis?.note).toMatch(/no anchor/i);
  });

  it("forged anchor does not upgrade a revoked key offline (revocation-bypass guard)", () => {
    const receipt = loadJson(REVOKED_DIR, "receipt.json");
    const revokedJwks = loadJson(REVOKED_DIR, "jwks.json");
    const futureJwks = JSON.parse(JSON.stringify(revokedJwks)) as Record<string, unknown>;
    ((futureJwks.keys as Array<Record<string, unknown>>)[0]).revoked_at = "2026-12-01T00:00:00+00:00";
    // Anchors sit outside the signed payload and are only shape-checked, so a
    // revoked-key holder can append one. Offline it is not verifiable timing.
    const anchored = JSON.parse(JSON.stringify(receipt)) as Record<string, unknown>;
    anchored.anchors = [{ type: "rfc3161", value: "dGVzdC10c3ItY29va2ll", tsa_url: "https://tsa.example/timestamp" }];
    const result = verifyReceiptOffline(anchored, futureJwks);
    expect(result.verdict).not.toBe("PASS");
    expect(result.verdict).toBe("INCOMPLETE");
    const keyAxis = result.axes.find((a) => a.axis === "key_status");
    expect(keyAxis?.result).toBe("SKIPPED");
    expect(keyAxis?.note).toMatch(/no anchor|self-attested/i);
  });

  it("c386: compromised-key backdated receipt, offline, no anchor, must NOT PASS", () => {
    // A receipt whose key is now compromised (status=compromised) but whose
    // signed issued_at is BEFORE revoked_at. Offline with no anchor the
    // self-attested issued_at is forgeable, so PASS is forbidden.
    const receipt = loadJson(REVOKED_DIR, "receipt.json");
    const compromisedJwks = JSON.parse(JSON.stringify(loadJson(REVOKED_DIR, "jwks.json"))) as Record<string, unknown>;
    ((compromisedJwks.keys as Array<Record<string, unknown>>)[0]).status = "compromised";
    ((compromisedJwks.keys as Array<Record<string, unknown>>)[0]).revoked_at = "2026-12-01T00:00:00+00:00";
    const result = verifyReceiptOffline(receipt, compromisedJwks);
    expect(result.verdict).not.toBe("PASS");
    expect(result.verdict).toBe("INCOMPLETE");
    const keyAxis = result.axes.find((a) => a.axis === "key_status");
    expect(keyAxis?.result).toBe("SKIPPED");
  });

  it("revoked key with revoked_at AT issuance fails (at-or-before rule)", () => {
    const receipt = loadJson(REVOKED_DIR, "receipt.json");
    const revokedJwks = loadJson(REVOKED_DIR, "jwks.json");
    // Same timestamp as issued_at -> rev <= iss -> FAIL
    const atJwks = JSON.parse(JSON.stringify(revokedJwks)) as Record<string, unknown>;
    ((atJwks.keys as Array<Record<string, unknown>>)[0]).revoked_at = "2026-06-01T12:00:00+00:00";
    const result = verifyReceiptOffline(receipt, atJwks);
    expect(result.verdict).toBe("FAIL");
    const keyAxis = result.axes.find((a) => a.axis === "key_status");
    expect(keyAxis?.result).toBe("FAIL");
    expect(keyAxis?.note).toMatch(/on\/before issuance/i);
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

// ---------------------------------------------------------------------------
// Cross-issuer forgery: a real key from the shared JWKS, another org's claim
// ---------------------------------------------------------------------------

describe("verifyReceiptOffline - issuer binding (no network)", () => {
  function crossIssuerForgery(): [Record<string, unknown>, Record<string, unknown>] {
    const attacker = ml_dsa65.keygen();
    const victim = ml_dsa65.keygen();
    const payload: Record<string, unknown> = {
      type: "protectmcp:decision",
      issued_at: "2026-06-19T00:00:00.000000Z",
      issuer_id: "org-victim",
      agent_id: "agt_attacker",
      action_ref: "sha256:" + "a".repeat(64),
      payload_digest: { hash: "b".repeat(64), size: 128 },
      policy_digest: "sha256:" + "c".repeat(64),
      previousReceiptHash: "0".repeat(64),
      decision: "allow",
    };
    // kid points at the attacker's OWN published key, not the victim's.
    const envelope: Record<string, unknown> = {
      payload,
      signature: { alg: "ML-DSA-65", kid: "attacker-key", sig: "" },
      anchors: [],
    };
    const ad = ADAPTERS.find((a) => a.detect(envelope))!;
    const sig = ml_dsa65.sign(Buffer.from(ad.signingInput(envelope)), attacker.secretKey);
    (envelope.signature as Record<string, unknown>).sig = Buffer.from(sig).toString("base64");
    const key = (kid: string, issuerId: string, pub: Uint8Array) => ({
      kid,
      issuer_id: issuerId,
      alg: "ML-DSA-65",
      status: "active",
      public_key: Buffer.from(pub).toString("base64"),
    });
    const jwks: Record<string, unknown> = {
      keys: [
        key("victim-key", "org-victim", victim.publicKey),
        key("attacker-key", "org-attacker", attacker.publicKey),
      ],
    };
    return [envelope, jwks];
  }

  it("refuses a receipt signed by a key published under another issuer", () => {
    const [envelope, jwks] = crossIssuerForgery();
    const result = verifyReceiptOffline(envelope, jwks);
    expect(result.verdict).toBe("FAIL");
    const bind = result.axes.find((a) => a.axis === "issuer_bind");
    expect(bind?.result).toBe("FAIL");
    // The forged signature itself verifies; the bind is what refuses it.
    expect(result.axes.find((a) => a.axis === "signature")?.result).toBe("PASS");
  });

  it("still PASSes a receipt signed by the claimed issuer's own key", () => {
    const { publicKey, secretKey } = ml_dsa65.keygen();
    const kid = "agent-one";
    const payload: Record<string, unknown> = {
      type: "protectmcp:decision",
      issued_at: "2026-06-19T00:00:00.000000Z",
      issuer_id: "org-legit",
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
    const sig = ml_dsa65.sign(Buffer.from(ad.signingInput(envelope)), secretKey);
    (envelope.signature as Record<string, unknown>).sig = Buffer.from(sig).toString("base64");
    const jwks: Record<string, unknown> = {
      keys: [
        {
          kid,
          issuer_id: "org-legit",
          alg: "ML-DSA-65",
          status: "active",
          public_key: Buffer.from(publicKey).toString("base64"),
        },
      ],
    };
    expect(verifyReceiptOffline(envelope, jwks).verdict).toBe("PASS");
  });
});
