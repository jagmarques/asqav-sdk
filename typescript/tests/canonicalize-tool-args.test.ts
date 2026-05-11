/**
 * Tests for `canonicalizeToolArgs`.
 *
 * The helper pre-validates `tool_args` payloads against the Compliance
 * Receipts canonicalization rule: no floats in the signed canonical
 * scope, because RFC 8785 section 3.2.2 only fixes byte-stable numeric
 * output for safe-range integers. Floats drift across runtimes and
 * break verifier equality.
 */

import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";

import { canonicalize, canonicalizeToolArgs } from "../src/canonicalize.js";

describe("canonicalizeToolArgs() happy path", () => {
  it("returns JCS bytes for dict with int + str + bool + nested dict + list", () => {
    const args = {
      tool: "database.query",
      limit: 100,
      dry_run: true,
      filters: { status: "active", tags: ["a", "b"] },
      nullable: null,
    };
    const out = canonicalizeToolArgs(args);
    expect(out).toEqual(canonicalize(args));
    const left = createHash("sha256").update(out).digest("hex");
    const right = createHash("sha256").update(canonicalize(args)).digest("hex");
    expect(left).toBe(right);
  });

  it("accepts a list at root", () => {
    const out = canonicalizeToolArgs([{ a: 1 }, { b: 2 }]);
    expect(new TextDecoder().decode(out)).toBe('[{"a":1},{"b":2}]');
  });

  it("accepts a decimal-as-string", () => {
    const out = canonicalizeToolArgs({ x: "1.5" });
    expect(new TextDecoder().decode(out)).toBe('{"x":"1.5"}');
  });

  it("does not misclassify booleans as floats", () => {
    const out = canonicalizeToolArgs({ flag: true, count: 0 });
    expect(new TextDecoder().decode(out)).toBe('{"count":0,"flag":true}');
  });
});

describe("canonicalizeToolArgs() rejects floats", () => {
  it("rejects a bare float with the RFC pointer in the message", () => {
    expect(() => canonicalizeToolArgs({ x: 1.5 })).toThrow(/tool_args float at \/x/);
    expect(() => canonicalizeToolArgs({ x: 1.5 })).toThrow(/RFC 8785 section 3\.2\.2/);
  });

  it("reports the JSON pointer for a nested float", () => {
    expect(() => canonicalizeToolArgs({ a: { b: [1, 2.5] } })).toThrow(/\/a\/b\/1/);
  });

  it("rejects NaN", () => {
    expect(() => canonicalizeToolArgs({ x: Number.NaN })).toThrow(/tool_args float at \/x/);
  });

  it("rejects Infinity", () => {
    expect(() => canonicalizeToolArgs({ x: Number.POSITIVE_INFINITY })).toThrow(
      /tool_args float at \/x/,
    );
  });

  it("escapes slash and tilde in JSON pointer tokens", () => {
    expect(() => canonicalizeToolArgs({ "a/b": 1.5 })).toThrow(/\/a~1b/);
    expect(() => canonicalizeToolArgs({ "a~b": 1.5 })).toThrow(/\/a~0b/);
  });
});

describe("canonicalizeToolArgs() negative conformance fixture", () => {
  it("receipt payload.tool_args with a float fails canonicalization", () => {
    const receipt = {
      payload: {
        action_type: "tool:call",
        tool_args: { price: 1.5 },
      },
    };
    expect(() => canonicalizeToolArgs(receipt.payload.tool_args)).toThrow(
      /tool_args float at \/price/,
    );
    expect(() => canonicalizeToolArgs(receipt.payload.tool_args)).toThrow(
      /RFC 8785 section 3\.2\.2/,
    );
  });

  it("same fixture with the float stringified canonicalizes cleanly", () => {
    const receipt = {
      payload: {
        action_type: "tool:call",
        tool_args: { price: "1.5" },
      },
    };
    const out = canonicalizeToolArgs(receipt.payload.tool_args);
    expect(new TextDecoder().decode(out)).toBe('{"price":"1.5"}');
  });
});
