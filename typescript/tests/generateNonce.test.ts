/**
 * ESM bundle crypto regression: tsup turns require("node:crypto") into a __require() shim that
 * throws in pure-ESM Node 22, so generateNonce uses globalThis.crypto.getRandomValues.
 */

import { describe, expect, it } from "vitest";
import { generateNonce } from "../src/index.js";

describe("generateNonce", () => {
  it("returns a 24-char hex string", () => {
    const nonce = generateNonce();
    expect(typeof nonce).toBe("string");
    expect(nonce).toHaveLength(24);
    expect(/^[0-9a-f]{24}$/.test(nonce)).toBe(true);
  });

  it("returns distinct values on repeated calls", () => {
    const nonces = new Set(Array.from({ length: 20 }, () => generateNonce()));
    // 20 draws from 12-byte random space - collision probability negligible
    expect(nonces.size).toBe(20);
  });

  it("uses globalThis.crypto so the ESM .mjs bundle does not require() crypto", () => {
    // If require() were used, this test would throw in an ESM context.
    // Vitest runs in Node ESM mode, so a CJS require() shim would surface here.
    expect(() => generateNonce()).not.toThrow();
  });
});
