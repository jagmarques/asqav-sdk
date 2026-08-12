/**
 * Strict JSON ingest (criterion 419): duplicate members fail closed.
 *
 * Every receipt- and record-parsing path rejects a duplicated JSON member name
 * at ANY nesting depth as a terminal parse failure, before any hashing,
 * canonicalisation, or signature check. Last-wins ingest would hash the bytes an
 * attacker kept and drop the ones they replaced; these tests pin the rejection.
 */
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  DuplicateMemberError,
  parseJsonPreservingFloats,
  parseJsonStrict,
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
