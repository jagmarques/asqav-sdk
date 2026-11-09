import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { flatSignedFields } from "../src/verifier/adapters/asqavNative.js";
import { FLAT_SIGNED, FLAT_SIGNED_OPTIONAL } from "../src/verifier/vrShim.js";

interface Contract {
  required: string[];
  optional: string[];
}

function contract(): Contract {
  const path = resolve(__dirname, "..", "..", "contracts", "flat-signed.json");
  return JSON.parse(readFileSync(path, "utf-8")) as Contract;
}

function drift(shape: Contract, required: Iterable<string>, optional: Iterable<string>): string[] {
  const apart = (a: string[], b: string[]) =>
    [...a.filter((n) => !b.includes(n)), ...b.filter((n) => !a.includes(n))].sort();
  return [...apart(shape.required, [...required]), ...apart(shape.optional, [...optional])];
}

describe("flat hash-mode signed members contract", () => {
  it("TypeScript verifier members match the contract", () => {
    const shape = contract();
    expect([...FLAT_SIGNED]).toEqual(shape.required);
    expect([...FLAT_SIGNED_OPTIONAL]).toEqual(shape.optional);
    expect(drift(shape, FLAT_SIGNED, FLAT_SIGNED_OPTIONAL)).toEqual([]);
  });

  it("rebuilt receipt members are the contract members", () => {
    const shape = contract();
    const doc = Object.fromEntries([...shape.required, ...shape.optional].map((n) => [n, "x"]));
    expect(Object.keys(flatSignedFields(doc))).toEqual([...shape.required, ...shape.optional]);
  });

  // NEGCTL: a contract that lost one member is reported as drift
  it("negctl dropped contract member is reported", () => {
    const shape = contract();
    for (const key of ["required", "optional"] as const) {
      const shrunk = { ...shape, [key]: shape[key].slice(1) };
      expect(drift(shrunk, FLAT_SIGNED, FLAT_SIGNED_OPTIONAL)).toEqual([shape[key][0]]);
    }
  });
});
