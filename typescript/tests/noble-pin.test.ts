// @noble/post-quantum is pinned to one exact version, in package.json and
// in the lock root mirror

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

const ROOT = join(__dirname, "..");
const PACKAGE = "@noble/post-quantum";
const EXACT = /^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/;

function readJson(name: string): any {
  return JSON.parse(readFileSync(join(ROOT, name), "utf-8"));
}

export function rangeProblems(spec: unknown): string[] {
  if (typeof spec !== "string") return ["no spec"];
  return EXACT.test(spec) ? [] : [`not an exact version: ${spec}`];
}

describe("@noble/post-quantum pin", () => {
  it("is an exact version in package.json", () => {
    const spec = readJson("package.json").dependencies[PACKAGE];
    expect(rangeProblems(spec)).toEqual([]);
  });

  it("is the same exact version in the lock root mirror", () => {
    const pkg = readJson("package.json").dependencies[PACKAGE];
    const lock = readJson("package-lock.json");
    expect(lock.packages[""].dependencies[PACKAGE]).toBe(pkg);
    expect(lock.packages[`node_modules/${PACKAGE}`].version).toBe(pkg);
  });

  it("NEGCTL: refuses a range operator in the spec", () => {
    for (const spec of ["^0.6.1", "~0.6.1", ">=0.6.1", "0.6.x", "*", "latest", undefined]) {
      expect(rangeProblems(spec)).not.toEqual([]);
    }
    const real = readJson("package.json").dependencies[PACKAGE];
    expect(rangeProblems(`^${real}`)).toEqual([`not an exact version: ^${real}`]);
  });
});
