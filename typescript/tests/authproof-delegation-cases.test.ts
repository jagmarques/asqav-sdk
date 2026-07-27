/**
 * Shared authproof delegationId format table, TypeScript half.
 *
 * One JSON table drives both authproof adapters. Each case feeds a delegationId
 * through this language's format-detection validator and pins the boolean, so a
 * regex that drifts between languages fails a suite instead of waiting for a
 * reviewer. The Python half lives in
 * python/tests/test_authproof_delegation_cases.py and reads the same file.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { AuthproofAdapter } from "../src/verifier/adapters/authproof.js";

interface Case {
  name: string;
  value: string;
  expected: boolean;
}

const CASES_FILE = resolve(__dirname, "..", "..", "verifier", "authproof-delegation-cases.json");
const CASES = (JSON.parse(readFileSync(CASES_FILE, "utf-8")) as { cases: Case[] }).cases;

/** A receipt satisfying every detect() condition except the delegationId, so the
 *  detect() boolean isolates the delegationId regex the table pins. */
function skeleton(delegationId: string): Record<string, unknown> {
  return {
    delegationId,
    issuedAt: 1719000000000,
    scope: "session",
    boundaries: { maxActions: 1 },
    timeWindow: { start: 0, end: 1 },
    operatorInstructions: "carry out the agreed task",
    instructionsHash: "sha256:" + "a".repeat(64),
    signerPublicKey: { kty: "EC", crv: "P-256", x: "AA", y: "AA" },
    signature: "ab".repeat(64),
  };
}

describe("authproof delegationId cross-language agreement", () => {
  const adapter = new AuthproofAdapter();
  for (const c of CASES) {
    it(c.name, () => {
      expect(adapter.detect(skeleton(c.value)), `${c.name}: ${JSON.stringify(c.value)}`).toBe(
        c.expected,
      );
    });
  }
});
