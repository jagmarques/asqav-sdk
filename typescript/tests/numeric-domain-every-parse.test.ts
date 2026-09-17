/**
 * The numeric-domain rule holds at every object parse boundary (criterion 686).
 * Twin of python/tests/test_numeric_domain_every_parse.py: no field inside the
 * receipt can switch the rule off, and the acta split-verdict bytes read the
 * same here as in Python.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { verify, type VerifyResult } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";

const VECTORS = resolve(__dirname, "..", "..", "verifier", "conformance-vectors");
const GENESIS = resolve(VECTORS, "asqav-01-genesis-permit");
const FIXTURES = resolve(__dirname, "fixtures");
const BIG = 9007199254740993;

function loadJson(path: string): Record<string, unknown> {
  return JSON.parse(readFileSync(path, "utf-8")) as Record<string, unknown>;
}

function refused(result: VerifyResult): boolean {
  return result.axes.some((a) => a.result === "FAIL" && a.note.includes("profile range"));
}

describe("numeric domain at every object parse", () => {
  it("control: pristine genesis permit verifies", () => {
    const doc = loadJson(resolve(GENESIS, "receipt.json"));
    const keys = loadJson(resolve(GENESIS, "jwks.json"));
    expect(verify(doc, ADAPTERS, keys).verdict).toBe("verified");
  });

  it("control: v=1 plus an out-of-domain integer is refused by the rule", () => {
    const doc = loadJson(resolve(GENESIS, "receipt.json"));
    const keys = loadJson(resolve(GENESIS, "jwks.json"));
    (doc.payload as Record<string, unknown>).probe = BIG;
    expect((doc.payload as Record<string, unknown>).v).toBe(1);
    expect(refused(verify(doc, ADAPTERS, keys))).toBe(true);
  });

  it("rule applies with v missing", () => {
    const doc = loadJson(resolve(GENESIS, "receipt.json"));
    const keys = loadJson(resolve(GENESIS, "jwks.json"));
    const payload = doc.payload as Record<string, unknown>;
    payload.probe = BIG;
    delete payload.v;
    const result = verify(doc, ADAPTERS, keys);
    expect(result.verdict).toBe("unverified");
    expect(refused(result)).toBe(true);
  });

  it("rule applies with v=2", () => {
    const doc = loadJson(resolve(GENESIS, "receipt.json"));
    const keys = loadJson(resolve(GENESIS, "jwks.json"));
    const payload = doc.payload as Record<string, unknown>;
    payload.probe = BIG;
    payload.v = 2;
    const result = verify(doc, ADAPTERS, keys);
    expect(result.verdict).toBe("unverified");
    expect(refused(result)).toBe(true);
  });

  it("acta big-integer bytes read unverified, matching Python", () => {
    const doc = loadJson(resolve(FIXTURES, "t202-acta-big-receipt.json"));
    const keys = loadJson(resolve(FIXTURES, "t202-acta-big-jwks.json"));
    const result = verify(doc, ADAPTERS, keys);
    expect(result.verdict).toBe("unverified");
  });
});
