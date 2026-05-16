/**
 * Cross-language conformance: the TypeScript fingerprint helpers must
 * agree with conformance/vectors.json byte-for-byte (which is generated
 * and also validated by the Python tests).
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { canonicalize, hashAction } from "../src/canonicalize.js";

interface Vector {
  name: string;
  input: unknown;
  canonical: string;
  sha256: string;
  capture_topology?: string;
  expected_verify?: boolean;
}

const vectorsPath = resolve(__dirname, "..", "..", "conformance", "vectors.json");
const { vectors } = JSON.parse(readFileSync(vectorsPath, "utf8")) as {
  vectors: Vector[];
};

// Closed Literal mirrored from the cloud SignRequest and IETF draft -04 appendix.
const CAPTURE_TOPOLOGY_VOCABULARY = new Set([
  "in_process_sdk",
  "network_proxy",
  "browser_extension",
  "ebpf_observer",
  "mcp_proxy",
]);

describe("canonicalize() against conformance vectors", () => {
  for (const v of vectors) {
    it(`canonical bytes match for ${v.name}`, () => {
      const actual = new TextDecoder().decode(canonicalize(v.input));
      expect(actual).toBe(v.canonical);
    });

    it(`SHA-256 matches for ${v.name}`, () => {
      const bytes = canonicalize(v.input);
      const hex = createHash("sha256").update(bytes).digest("hex");
      expect(hex).toBe(v.sha256);
    });
  }
});

describe("hashAction()", () => {
  it("matches sha256 for {action_type, context}-shaped vectors", async () => {
    const happy = vectors.filter((v) => {
      const keys = Object.keys(v.input as object);
      return keys.length === 2 && keys.includes("action_type") && keys.includes("context");
    });
    expect(happy.length).toBeGreaterThan(0);
    for (const v of happy) {
      const input = v.input as { action_type: string; context: Record<string, unknown> };
      const result = await hashAction(input.action_type, input.context);
      expect(result).toBe(`sha256:${v.sha256}`);
    }
  });

  it("uses HMAC when a salt is provided", async () => {
    const salt = Buffer.from(
      "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20",
      "hex",
    );
    const plain = await hashAction("api:call", { k: 3 });
    const keyed = await hashAction("api:call", { k: 3 }, new Uint8Array(salt));
    expect(plain).not.toBe(keyed);
    expect(keyed.startsWith("sha256:")).toBe(true);
    expect(keyed.length).toBe("sha256:".length + 64);
  });
});

describe("capture_topology conformance vectors (IETF -04 appendix)", () => {
  const captureVectors = vectors.filter((v) =>
    v.name.startsWith("capture_topology_"),
  );

  it("covers all five values in the closed vocabulary", () => {
    const accepted = new Set(
      captureVectors
        .filter((v) => v.expected_verify === true)
        .map((v) => v.capture_topology),
    );
    expect(accepted).toEqual(CAPTURE_TOPOLOGY_VOCABULARY);
  });

  for (const value of [...CAPTURE_TOPOLOGY_VOCABULARY]) {
    it(`stamps capture_topology=${value} on the manifest, not the signed payload`, () => {
      const matching = captureVectors.filter(
        (v) => v.capture_topology === value && v.expected_verify === true,
      );
      expect(matching).toHaveLength(1);
      const v = matching[0];
      const input = v.input as {
        manifest?: { capture_topology?: string };
        context?: Record<string, unknown>;
      };
      expect(input.manifest?.capture_topology).toBe(value);
      expect(input.context).toBeDefined();
      expect((input.context ?? {}).capture_topology).toBeUndefined();
    });
  }

  it("ships an unknown-value rejection vector", () => {
    const rejected = captureVectors.filter((v) => v.expected_verify === false);
    expect(rejected.length).toBeGreaterThan(0);
    const v = rejected[0];
    expect(CAPTURE_TOPOLOGY_VOCABULARY.has(v.capture_topology ?? "")).toBe(false);
  });
});

describe("canonicalize() invariants", () => {
  it("is deterministic across key insertion order", () => {
    const a = { b: 1, a: 2, c: [{ y: 1, x: 2 }] };
    const b = { a: 2, c: [{ x: 2, y: 1 }], b: 1 };
    expect(canonicalize(a)).toEqual(canonicalize(b));
  });

  it("emits non-ASCII as raw UTF-8", () => {
    const out = canonicalize({ name: "café" });
    const utf8 = new TextEncoder().encode("café");
    const haystack = Buffer.from(out);
    expect(haystack.includes(Buffer.from(utf8))).toBe(true);
    expect(new TextDecoder().decode(out)).not.toContain("\\u");
  });

  it("rejects NaN", () => {
    expect(() => canonicalize({ x: Number.NaN })).toThrow();
  });
});
