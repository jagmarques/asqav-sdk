/**
 * Shared adversarial case table, TypeScript half.
 *
 * One JSON table drives both verifiers. Each case pins a verdict and the result
 * of every axis it names, so a rule that drifts in one language fails a suite
 * instead of waiting for a reviewer to notice. The Python half lives in
 * python/tests/test_cross_language_cases.py and reads the same file.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { verifyReceiptOffline } from "../src/index.js";

interface Case {
  name: string;
  receipt: Record<string, unknown>;
  jwks: Record<string, unknown>;
  expect: { verdict: string; axes: Record<string, string> };
}

const CASES_FILE = resolve(__dirname, "..", "..", "verifier", "cross-language-cases.json");
const CASES = (JSON.parse(readFileSync(CASES_FILE, "utf-8")) as { cases: Case[] }).cases;

describe("cross-language adversarial case table", () => {
  it("is populated", () => {
    expect(CASES.length).toBeGreaterThanOrEqual(20);
  });

  for (const c of CASES) {
    it(c.name, () => {
      const result = verifyReceiptOffline(c.receipt, c.jwks);
      const axes: Record<string, string> = {};
      for (const a of result.axes) axes[a.axis] = a.result;
      expect(result.verdict, `${c.name}: axes ${JSON.stringify(axes)}`).toBe(c.expect.verdict);
      for (const [axis, expected] of Object.entries(c.expect.axes)) {
        expect(axes[axis], `${c.name}: axis ${axis}`).toBe(expected);
      }
    });
  }
});
