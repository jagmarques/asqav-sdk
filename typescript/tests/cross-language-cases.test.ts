/**
 * Shared adversarial case table, TS half: one JSON table pins a verdict and every named axis
 * for both verifiers, so a rule drifting in one language fails a suite. Python reads the same file.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { verifyReceiptOffline } from "../src/index.js";

interface Case {
  name: string;
  receipt: Record<string, unknown>;
  jwks: Record<string, unknown>;
  expect: { verdict: string; failure_class: string | null; axes: Record<string, string> };
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
      // invalid and unverifiable are never collapsed (criterion 418).
      expect(result.failureClass, `${c.name}: axes ${JSON.stringify(axes)}`).toBe(
        c.expect.failure_class,
      );
      for (const [axis, expected] of Object.entries(c.expect.axes)) {
        expect(axes[axis], `${c.name}: axis ${axis}`).toBe(expected);
      }
    });
  }
});
