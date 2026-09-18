/**
 * Twin of the Python test module, versioned round 2: declared v gets it
 * No version keeps legacy pre-rule behaviour on these acta bytes
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

  it("v missing keeps the legacy verdict with no rule note", () => {
    const doc = loadJson(resolve(GENESIS, "receipt.json"));
    const keys = loadJson(resolve(GENESIS, "jwks.json"));
    const payload = doc.payload as Record<string, unknown>;
    payload.probe = BIG;
    delete payload.v;
    const result = verify(doc, ADAPTERS, keys);
    expect(result.verdict).toBe("unverified");
    expect(result.axes.length).toBe(12);
    expect(refused(result)).toBe(false);
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

  it("acta legacy bytes stay unverified via lossy parse, rule silent", () => {
    const doc = loadJson(resolve(FIXTURES, "t202-acta-big-receipt.json"));
    const keys = loadJson(resolve(FIXTURES, "t202-acta-big-jwks.json"));
    const result = verify(doc, ADAPTERS, keys);
    expect(result.verdict).toBe("unverified");
    expect(refused(result)).toBe(false);
  });

  it("acta bytes with v declared are refused by the rule", () => {
    const doc = loadJson(resolve(FIXTURES, "t202-acta-big-receipt.json"));
    const keys = loadJson(resolve(FIXTURES, "t202-acta-big-jwks.json"));
    (doc.payload as Record<string, unknown>).v = 2;
    const result = verify(doc, ADAPTERS, keys);
    expect(result.verdict).toBe("unverified");
    expect(refused(result)).toBe(true);
  });
});
