/**
 * Fail-clean robustness (rule 3.12): the neutral verifier returns a verdict, never
 * throws, on malformed input (non-object receipt, non-string alg). Parity with Python.
 */
import { describe, expect, it } from "vitest";

import { ADAPTERS } from "../src/verifier/index.js";
import { verify as oracleVerify } from "../src/verifier/core.js";
import { verifySignature } from "../src/verifier/crypto.js";

describe("verifier fail-clean on malformed input", () => {
  it("returns FAIL (no throw) on a non-object top-level receipt", () => {
    for (const doc of [null, "a string", 123, [1, 2]] as unknown[]) {
      const r = oracleVerify(doc as Record<string, unknown>, ADAPTERS);
      expect(r.verdict).toBe("FAIL");
    }
  });

  it("SKIPs (no throw) when alg is not a string", () => {
    const z = new Uint8Array(0);
    for (const alg of [123, null, ["x"], {}] as unknown[]) {
      const r = verifySignature(alg as string, z, z, z);
      expect(r.result).toBe("SKIPPED");
    }
  });
});
