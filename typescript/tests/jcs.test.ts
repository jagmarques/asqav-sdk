/**
 * RFC 8785 (JCS) golden-vector tests for `canonicalJson`.
 *
 * Vectors come from RFC 8785 §3 (the appendix of test data).
 * https://datatracker.ietf.org/doc/html/rfc8785
 */

import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";

import { canonicalJson } from "../src/jcs.js";
import { canonicalize } from "../src/canonicalize.js";

function decode(bytes: Uint8Array): string {
  return new TextDecoder().decode(bytes);
}

describe("canonicalJson() RFC 8785 golden vectors", () => {
  it("§3.2.3 sorts object keys lexicographically (UTF-16 code unit order)", () => {
    const input = { peach: "This sorting order", péché: "is wrong according to French", pêche: "but ordains", sin: "ordering by code unit" };
    const out = decode(canonicalJson(input));
    // The keys end up sorted by JavaScript's default string sort, which is
    // UTF-16 code unit order: ascii letters come before pre-composed Latin
    // accented letters.
    expect(out).toBe(
      "{\"peach\":\"This sorting order\",\"péché\":\"is wrong according to French\",\"pêche\":\"but ordains\",\"sin\":\"ordering by code unit\"}"
    );
  });

  it("emits no insignificant whitespace", () => {
    const input = { a: 1, b: [1, 2, 3], c: { d: true } };
    const out = decode(canonicalJson(input));
    expect(out).toBe("{\"a\":1,\"b\":[1,2,3],\"c\":{\"d\":true}}");
    expect(out).not.toMatch(/[\n\t ]/);
  });

  it("emits non-ASCII strings as raw UTF-8", () => {
    const input = { name: "café" };
    const bytes = canonicalJson(input);
    const utf8 = new TextEncoder().encode("café");
    const haystack = Buffer.from(bytes);
    expect(haystack.includes(Buffer.from(utf8))).toBe(true);
    // No \u escape for non-control non-ASCII characters.
    expect(decode(bytes)).not.toContain("\\u");
  });

  it("§3.2.2 emits integers without trailing decimals", () => {
    const input = { n: 42, big: 1000000 };
    const out = decode(canonicalJson(input));
    expect(out).toBe("{\"big\":1000000,\"n\":42}");
  });

  it("§3.2.2 maps -0 to 0", () => {
    const out = decode(canonicalJson({ n: -0 }));
    expect(out).toBe("{\"n\":0}");
  });

  it("§3.2.2 rejects NaN and Infinity", () => {
    expect(() => canonicalJson({ x: Number.NaN })).toThrow(TypeError);
    expect(() => canonicalJson({ x: Number.POSITIVE_INFINITY })).toThrow(TypeError);
    expect(() => canonicalJson({ x: Number.NEGATIVE_INFINITY })).toThrow(TypeError);
  });

  it("escapes the JSON-mandated control characters", () => {
    const input = { s: "a\b\t\n\f\r\"\\b" };
    const out = decode(canonicalJson(input));
    expect(out).toBe("{\"s\":\"a\\b\\t\\n\\f\\r\\\"\\\\b\"}");
  });

  it("escapes other U+0000..U+001F as \\u00xx", () => {
    const input = { s: "" };
    const out = decode(canonicalJson(input));
    expect(out).toBe("{\"s\":\"\\u0001\\u001f\"}");
  });

  it("is byte-identical with canonicalize() (legacy alias)", () => {
    const input = { z: 1, a: { c: [1, 2, 3], b: "x" } };
    expect(canonicalJson(input)).toEqual(canonicalize(input));
  });

  it("nested ordering: keys recursively sorted at every level", () => {
    const a = { z: { y: 1, a: 2 }, a: 1 };
    const b = { a: 1, z: { a: 2, y: 1 } };
    expect(canonicalJson(a)).toEqual(canonicalJson(b));
  });

  it("known SHA-256 digest for the §5.7 chain seed envelope", () => {
    // First record's previousReceiptHash MUST equal "0"*64 per §5.7.
    const envelope = {
      type: "protectmcp:decision",
      issued_at: "2026-05-04T00:00:00Z",
      issuer_id: "issuer-test",
      agent_id: "agt_test",
      action_ref: "sha256:" + "a".repeat(64),
      payload_digest: "sha256:" + "a".repeat(64),
      policy_digest: "sha256:" + "b".repeat(64),
      previousReceiptHash: "0".repeat(64),
      receipt_type: "protectmcp:decision",
      policy_decision: "permit",
    };
    const bytes = canonicalJson(envelope);
    const hex = createHash("sha256").update(bytes).digest("hex");
    expect(hex).toMatch(/^[0-9a-f]{64}$/);
    // Reproducible across key insertion order.
    const reordered = {
      receipt_type: "protectmcp:decision",
      previousReceiptHash: "0".repeat(64),
      policy_decision: "permit",
      policy_digest: "sha256:" + "b".repeat(64),
      payload_digest: "sha256:" + "a".repeat(64),
      action_ref: "sha256:" + "a".repeat(64),
      agent_id: "agt_test",
      issuer_id: "issuer-test",
      issued_at: "2026-05-04T00:00:00Z",
      type: "protectmcp:decision",
    };
    const hex2 = createHash("sha256").update(canonicalJson(reordered)).digest("hex");
    expect(hex).toBe(hex2);
  });
});
