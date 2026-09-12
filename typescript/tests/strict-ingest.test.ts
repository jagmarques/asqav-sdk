/**
 * Strict JSON ingest (criterion 419): a duplicated member at ANY depth is terminal before any
 * hashing or signature check, since last-wins would hash the bytes an attacker kept.
 */
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  DuplicateMemberError,
  ProfileIntegerError,
  RawFloat,
  parseJsonPreservingFloats,
  parseJsonStrict,
  parseProfileJson,
} from "../src/verifier/canonical.js";
import { receiptFromOtelGenaiAttributes, OTEL_RECEIPT_ATTR } from "../src/doors.js";
import { runOne } from "../src/verifier/runner.js";

const CORPUS_ROOT = resolve(__dirname, "..", "..", "verifier", "conformance-vectors");

describe("strict ingest rejects duplicate members at any depth (criterion 419)", () => {
  it("rejects a duplicated member at the top level", () => {
    expect(() => parseJsonPreservingFloats('{"payload": {"a": 1}, "payload": {"a": 2}}')).toThrow(
      DuplicateMemberError,
    );
    expect(() => parseJsonStrict('{"payload": {"a": 1}, "payload": {"a": 2}}')).toThrow(
      DuplicateMemberError,
    );
  });

  it("rejects a duplicated member nested inside an object", () => {
    expect(() =>
      parseJsonPreservingFloats('{"payload": {"digest": {"hash": "x", "hash": "y"}}}'),
    ).toThrow(DuplicateMemberError);
  });

  it("rejects a duplicate five levels down", () => {
    expect(() => parseJsonStrict('{"a": {"b": {"c": {"d": [{"e": 1, "e": 2}]}}}}')).toThrow(
      DuplicateMemberError,
    );
  });

  it("rejects a duplicate inside an array element", () => {
    expect(() => parseJsonStrict('{"list": [{"k": 1}, {"k": 2, "k": 3}]}')).toThrow(
      DuplicateMemberError,
    );
  });

  it("allows the same name in sibling objects (one object scope only)", () => {
    expect(parseJsonStrict('{"list": [{"k": 1}, {"k": 2}]}')).toEqual({
      list: [{ k: 1 }, { k: 2 }],
    });
  });

  it("clean documents still parse to plain values", () => {
    expect(parseJsonStrict('{"a": 1, "b": [true, false, null]}')).toEqual({
      a: 1,
      b: [true, false, null],
    });
  });

  it("DuplicateMemberError is a SyntaxError so existing catch sites stay fail-closed", () => {
    try {
      parseJsonStrict('{"a": 1, "a": 2}');
      expect.unreachable("should have thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(SyntaxError);
      expect((e as Error).name).toBe("DuplicateMemberError");
    }
  });

  it("the OTel GenAI door rejects a duplicate-member receipt string", () => {
    const attrs = { [OTEL_RECEIPT_ATTR]: '{"a": 1, "a": 2}' };
    expect(() => receiptFromOtelGenaiAttributes(attrs)).toThrow(DuplicateMemberError);
  });
});

describe("corpus duplicate-member vectors never verify (criteria 419/418)", () => {
  it("both vectors are terminal parse failures in the raw files", () => {
    for (const dir of ["asqav-11-dup-member-toplevel", "asqav-13-dup-member-nested"]) {
      const raw = readFileSync(join(CORPUS_ROOT, dir, "receipt.json"), "utf-8");
      expect(() => parseJsonPreservingFloats(raw)).toThrow(DuplicateMemberError);
    }
  });

  it("the runner reports them unverified/unverifiable, never verified", () => {
    for (const dir of ["asqav-11-dup-member-toplevel", "asqav-13-dup-member-nested"]) {
      const r = runOne(join(CORPUS_ROOT, dir), "asqav-native", "unverified", "duplicate_member", "unverifiable");
      expect(r.ok, r.detail).toBe(true);
      expect(r.actualVerdict).toBe("unverified");
      expect(r.actualFailureClass).toBe("unverifiable");
      expect(r.detail).toContain("terminal parse failure before any hashing");
    }
  });
});

describe("explicit Asqav profile ingest (shared parser unchanged)", () => {
  it("accepts the safe interval on both signs, nested or not", () => {
    expect(parseProfileJson('{"n":9007199254740991}')).toEqual({ n: 9007199254740991 });
    expect(parseProfileJson('{"a":{"b":[1,{"c":-9007199254740991}]}}')).toEqual({
      a: { b: [1, { c: -9007199254740991 }] },
    });
  });

  it("refuses the excluded boundary on both signs, nested or not", () => {
    for (const text of [
      '{"n":9007199254740992}',
      '{"n":-9007199254740992}',
      '{"a":{"b":[1,{"c":9007199254740992}]}}',
    ]) {
      expect(() => parseProfileJson(text)).toThrow(ProfileIntegerError);
    }
  });

  it("refuses float spellings of the excluded boundary", () => {
    expect(() => parseProfileJson('{"n":9007199254740992.0}')).toThrow(ProfileIntegerError);
    expect(() => parseProfileJson('{"n":9.007199254740992e15}')).toThrow(ProfileIntegerError);
  });

  it("keeps string twins, booleans and fractions unchanged", () => {
    expect(parseProfileJson('{"n":"9007199254740993"}')).toEqual({ n: "9007199254740993" });
    const parsed = parseProfileJson('{"a":true,"b":3.14}') as {
      a: boolean;
      b: RawFloat;
    };
    expect(parsed.a).toBe(true);
    expect(parsed.b).toBeInstanceOf(RawFloat);
    expect(parsed.b.value).toBe(3.14);
  });

  it("shared generic ingest still accepts the excluded boundary", () => {
    expect(parseJsonPreservingFloats('{"n":9007199254740992}')).toEqual({ n: 9007199254740992 });
  });
});

describe("decoded __proto__ keys stay own data members", () => {
  it("keeps __proto__ own and enumerable with its original value", () => {
    const parsed = parseJsonPreservingFloats('{"__proto__":{"v":1},"n":1}') as Record<
      string,
      unknown
    >;
    expect(Object.prototype.hasOwnProperty.call(parsed, "__proto__")).toBe(true);
    expect(Object.entries(parsed).some(([k]) => k === "__proto__")).toBe(true);
    expect(parsed.__proto__).toEqual({ v: 1 });
    expect({}.hasOwnProperty.call(parsed, "v")).toBe(false);
  });

  it("preserves __proto__ through the strict unwrap path", () => {
    const parsed = parseJsonStrict('{"__proto__":{"v":1},"n":1}') as Record<string, unknown>;
    expect(Object.prototype.hasOwnProperty.call(parsed, "__proto__")).toBe(true);
    expect(parsed.__proto__).toEqual({ v: 1 });
  });

  it("still detects a duplicated __proto__ member", () => {
    expect(() => parseJsonPreservingFloats('{"__proto__":1,"__proto__":2}')).toThrow(
      DuplicateMemberError,
    );
  });
});
