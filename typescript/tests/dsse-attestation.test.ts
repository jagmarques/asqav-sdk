/**
 * Offline DSSE attestation: the happy path runs the committed ML-DSA-65 vector and negatives
 * are derived by mutation, so each FAIL is anti-vacuous. No network; disk or in-process only.
 */

import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  buildPae,
  extractSubjectDigest,
  IN_TOTO_PAYLOAD_TYPE,
  verifyAttestation,
} from "../src/verifier/dsse.js";
import { b64decode } from "../src/verifier/vrShim.js";

const VECTORS = resolve(__dirname, "..", "..", "verifier", "conformance-vectors");
const DIR = join(VECTORS, "dsse-attestation-ml-dsa-65");
const COMMIT_SHA = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";

function loadJson(path: string): Record<string, unknown> {
  return JSON.parse(readFileSync(path, "utf-8")) as Record<string, unknown>;
}

function loadVector(): { envelope: Record<string, unknown>; jwks: Record<string, unknown> } {
  return { envelope: loadJson(join(DIR, "receipt.json")), jwks: loadJson(join(DIR, "jwks.json")) };
}

/** Decode the envelope payload to the in-toto Statement object. */
function decodeStatement(envelope: Record<string, unknown>): Record<string, unknown> {
  const bytes = b64decode(envelope.payload as string);
  return JSON.parse(new TextDecoder().decode(bytes)) as Record<string, unknown>;
}

/** Re-encode a (mutated) Statement back into envelope.payload as standard base64. */
function reencodePayload(envelope: Record<string, unknown>, statement: Record<string, unknown>): void {
  envelope.payload = Buffer.from(new TextEncoder().encode(JSON.stringify(statement))).toString("base64");
}

// ---------------------------------------------------------------------------
// Conformance vector: discovered via its per-dir manifest.json

describe("dsse-attestation conformance vector (manifest-driven)", () => {
  it("manifest.json discovers a PASS vector that verifyAttestation confirms", () => {
    const manifest = loadJson(join(DIR, "manifest.json")) as {
      dir: string;
      format: string;
      outcome: string;
      files: { envelope: string; jwks: string; expected: string };
    };
    expect(manifest.dir).toBe("dsse-attestation-ml-dsa-65");
    expect(manifest.format).toBe("dsse-attestation");

    const envelope = loadJson(join(DIR, manifest.files.envelope));
    const jwks = loadJson(join(DIR, manifest.files.jwks));
    const expected = loadJson(join(DIR, manifest.files.expected)) as { outcome: string };

    const result = verifyAttestation(envelope, jwks);
    expect(result.verdict).toBe(expected.outcome);
    expect(result.verdict).toBe("PASS");
    expect(result.axes.signature.result).toBe("PASS");
    expect(result.axes.key_status.result).toBe("PASS");
    expect(result.axes.structure.result).toBe("PASS");
  });
});

// ---------------------------------------------------------------------------
// Happy path

describe("verifyAttestation - valid envelope (no network)", () => {
  it("returns PASS for the committed real ML-DSA-65 DSSE envelope", () => {
    const { envelope, jwks } = loadVector();
    const result = verifyAttestation(envelope, jwks);
    expect(result.verdict).toBe("PASS");
    expect(result.axes.signature.result).toBe("PASS");
    expect(result.axes.signature.note).toMatch(/signature valid/);
  });

  it("exposes the bound subject digest and predicateType on the verdict", () => {
    const { envelope, jwks } = loadVector();
    const result = verifyAttestation(envelope, jwks);
    expect(result.subjectDigest).toBe(COMMIT_SHA);
    expect(result.predicateType).toBe("https://asqav.com/receipt/action/v1");
  });
});

// ---------------------------------------------------------------------------
// Tamper detection

describe("verifyAttestation - tamper detection (no network)", () => {
  it("returns FAIL when subject.digest is mutated after signing", () => {
    const { envelope, jwks } = loadVector();
    const statement = decodeStatement(envelope);
    ((statement.subject as Array<Record<string, unknown>>)[0].digest as Record<string, unknown>).sha256 =
      "deadbeef" + "0".repeat(56);
    reencodePayload(envelope, statement);

    const result = verifyAttestation(envelope, jwks);
    expect(result.verdict).toBe("FAIL");
    // Structure stays well-formed; the signature is what binds the digest.
    expect(result.axes.structure.result).toBe("PASS");
    expect(result.axes.signature.result).toBe("FAIL");
  });

  it("returns FAIL when the signature bytes are corrupted", () => {
    const { envelope, jwks } = loadVector();
    const sig = (envelope.signatures as Array<Record<string, unknown>>)[0];
    const raw = b64decode(sig.sig as string);
    const bad = new Uint8Array(raw.length);
    bad.set(raw);
    bad.fill(0, 0, 16);
    sig.sig = Buffer.from(bad).toString("base64");

    const result = verifyAttestation(envelope, jwks);
    expect(result.verdict).toBe("FAIL");
    expect(result.axes.signature.result).toBe("FAIL");
    // The key is still active; only the signature is broken.
    expect(result.axes.key_status.result).toBe("PASS");
  });
});

// ---------------------------------------------------------------------------
// Key status: fail-closed

describe("verifyAttestation - key status (no network)", () => {
  it("returns FAIL when revoked_at is set (sig still valid)", () => {
    const { envelope, jwks } = loadVector();
    (jwks.keys as Array<Record<string, unknown>>)[0].revoked_at = "2026-12-01T00:00:00+00:00";

    const result = verifyAttestation(envelope, jwks);
    expect(result.verdict).toBe("FAIL");
    expect(result.axes.key_status.result).toBe("FAIL");
    expect(result.axes.key_status.note).toMatch(/revoked/i);
    expect(result.axes.signature.result).toBe("PASS");
  });

  it("returns FAIL for a revoked status", () => {
    const { envelope, jwks } = loadVector();
    (jwks.keys as Array<Record<string, unknown>>)[0].status = "revoked";

    const result = verifyAttestation(envelope, jwks);
    expect(result.verdict).toBe("FAIL");
    expect(result.axes.key_status.result).toBe("FAIL");
  });

  it("returns FAIL when the kid is unknown to the JWKS", () => {
    const { envelope } = loadVector();
    const result = verifyAttestation(envelope, { keys: [] });
    expect(result.verdict).toBe("FAIL");
    expect(result.axes.key_status.result).toBe("FAIL");
    expect(result.axes.signature.result).toBe("FAIL");
    expect(result.axes.key_status.note).toMatch(/no key published/i);
  });

  it("returns FAIL when the envelope kid does not match any JWKS entry", () => {
    const { envelope, jwks } = loadVector();
    (envelope.signatures as Array<Record<string, unknown>>)[0].keyid = "some-other-kid";
    const result = verifyAttestation(envelope, jwks);
    expect(result.verdict).toBe("FAIL");
    expect(result.axes.key_status.result).toBe("FAIL");
  });
});

// ---------------------------------------------------------------------------
// Structure guards

describe("verifyAttestation - structure guards (no network)", () => {
  it("rejects a wrong payloadType", () => {
    const { envelope, jwks } = loadVector();
    envelope.payloadType = "application/json";
    const result = verifyAttestation(envelope, jwks);
    expect(result.verdict).toBe("FAIL");
    expect(result.axes.structure.result).toBe("FAIL");
    expect(result.axes.structure.note).toMatch(/payloadType/);
  });

  it("rejects an empty signatures array", () => {
    const { envelope, jwks } = loadVector();
    envelope.signatures = [];
    const result = verifyAttestation(envelope, jwks);
    expect(result.verdict).toBe("FAIL");
    expect(result.axes.structure.result).toBe("FAIL");
  });

  it("never throws on a malformed envelope", () => {
    expect(() => verifyAttestation(null, { keys: [] })).not.toThrow();
    expect(() => verifyAttestation({ payloadType: IN_TOTO_PAYLOAD_TYPE }, null)).not.toThrow();
    expect(verifyAttestation("garbage", null).verdict).toBe("FAIL");
  });
});

// ---------------------------------------------------------------------------
// PAE byte-exactness vs the backend formula (core/dsse.py pae())

describe("buildPae - DSSEv1 byte-exactness (known-answer)", () => {
  const enc = new TextEncoder();
  // Known-answer hex minted by the verbatim core/dsse.py pae() formula, so this
  // pins buildPae to the backend byte-for-byte with no hand-copied constant.
  const kat = loadJson(join(DIR, "pae_kat.json")) as {
    vectors: Array<{ name: string; payloadType: string; body: string; pae_ascii?: string; pae_hex: string }>;
  };
  const byName = (name: string) => kat.vectors.find((v) => v.name === name)!;

  it("matches the trivial DSSEv1 vector exactly", () => {
    const v = byName("trivial");
    const got = buildPae(v.payloadType, enc.encode(v.body));
    expect(Buffer.from(got).toString("hex")).toBe(v.pae_hex);
    expect(new TextDecoder().decode(got)).toBe(v.pae_ascii ?? "DSSEv1 1 a 1 b");
  });

  it("byte-matches the backend pae() over the in-toto payloadType", () => {
    const v = byName("in-toto");
    expect(v.payloadType).toBe(IN_TOTO_PAYLOAD_TYPE);
    const got = Buffer.from(buildPae(v.payloadType, enc.encode(v.body))).toString("hex");
    expect(got).toBe(v.pae_hex);
  });
});

// ---------------------------------------------------------------------------
// extractSubjectDigest

describe("extractSubjectDigest", () => {
  it("returns the sha256 hex the committed Statement binds", () => {
    const { envelope } = loadVector();
    expect(extractSubjectDigest(decodeStatement(envelope))).toBe(COMMIT_SHA);
  });

  it("lowercases a mixed-case digest", () => {
    const statement = { subject: [{ name: "x", digest: { sha256: "ABCDEF" + "0".repeat(58) } }] };
    expect(extractSubjectDigest(statement)).toBe("abcdef" + "0".repeat(58));
  });

  it("returns null when the digest is absent or malformed", () => {
    expect(extractSubjectDigest(null)).toBeNull();
    expect(extractSubjectDigest({})).toBeNull();
    expect(extractSubjectDigest({ subject: [] })).toBeNull();
    expect(extractSubjectDigest({ subject: [{ name: "x" }] })).toBeNull();
    expect(extractSubjectDigest({ subject: [{ name: "x", digest: {} }] })).toBeNull();
    expect(extractSubjectDigest({ subject: [{ name: "x", digest: { sha256: "" } }] })).toBeNull();
  });
});
