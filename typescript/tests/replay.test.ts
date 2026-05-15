/**
 * Offline chain verification under the IETF v2 chain shape.
 *
 * The v2 chain is `previousReceiptHash[i+1] = sha256(canonicalJson(env[i]))`
 * with seed `"0".repeat(64)` on the first record. This test builds a
 * synthetic chain, then mutates each link to confirm tamper detection.
 */

import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";

import { canonicalJson } from "../src/jcs.js";
import {
  FIRST_RECEIPT_SEED,
  deriveChainHash,
  verifyChain,
  type ChainRecord,
} from "../src/replay.js";

function sha256Hex(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function makeEnvelope(seq: number, prev: string): Record<string, unknown> {
  return {
    type: "protectmcp:decision",
    issued_at: `2026-05-04T00:00:0${seq}Z`,
    issuer_id: "Acme Ltd",
    agent_id: "agt_test",
    action_ref: `sha256:${"a".repeat(64)}`,
    payload_digest: `sha256:${"b".repeat(64)}`,
    policy_digest: `sha256:${"c".repeat(64)}`,
    previousReceiptHash: prev,
    receipt_type: "protectmcp:decision",
    policy_decision: "permit",
  };
}

function buildChain(n: number): ChainRecord[] {
  const records: ChainRecord[] = [];
  let prev = FIRST_RECEIPT_SEED;
  for (let i = 0; i < n; i++) {
    const env = makeEnvelope(i, prev);
    records.push({ signedEnvelope: env });
    prev = sha256Hex(canonicalJson(env));
  }
  return records;
}

describe("verifyChain (v2, IETF profile)", () => {
  it("seed is 64 zeros", () => {
    expect(FIRST_RECEIPT_SEED).toBe("0".repeat(64));
    expect(FIRST_RECEIPT_SEED.length).toBe(64);
  });

  it("verifies a freshly-built chain end-to-end", () => {
    const records = buildChain(5);
    const result = verifyChain(records);
    expect(result.chainIntegrity).toBe(true);
    expect(result.steps.length).toBe(5);
    for (const s of result.steps) {
      expect(s.chainValid).toBe(true);
    }
    expect(result.steps[0].expectedPreviousReceiptHash).toBe(FIRST_RECEIPT_SEED);
    expect(result.steps[0].actualPreviousReceiptHash).toBe(FIRST_RECEIPT_SEED);
  });

  it("derives chain hash via sha256(canonicalJson(envelope))", () => {
    const env = makeEnvelope(0, FIRST_RECEIPT_SEED);
    const expected = sha256Hex(canonicalJson(env));
    expect(deriveChainHash(env)).toBe(expected);
  });

  it("detects a tampered envelope mid-chain", () => {
    const records = buildChain(3);
    // Mutate the second record's payload_digest so the [1]->[2] chain link diverges.
    (records[1].signedEnvelope as Record<string, unknown>).payload_digest =
      `sha256:${"f".repeat(64)}`;
    const result = verifyChain(records);
    expect(result.chainIntegrity).toBe(false);
    expect(result.steps[0].chainValid).toBe(true);
    expect(result.steps[1].chainValid).toBe(true); // own prev still seeds correctly
    expect(result.steps[2].chainValid).toBe(false); // [1]'s hash changed
  });

  it("detects a forged previousReceiptHash on the first record", () => {
    const records = buildChain(2);
    (records[0].signedEnvelope as Record<string, unknown>).previousReceiptHash =
      "1".repeat(64);
    const result = verifyChain(records);
    expect(result.chainIntegrity).toBe(false);
    expect(result.steps[0].chainValid).toBe(false);
  });

  it("flags storedChainHashMatches when caller supplied a wrong chain hash", () => {
    const records = buildChain(1);
    records[0].chainHash = "0".repeat(64);
    const result = verifyChain(records);
    expect(result.steps[0].storedChainHashMatches).toBe(false);
    expect(result.chainIntegrity).toBe(false);
  });

  it("accepts snake_case `previous_receipt_hash` for one release", () => {
    const env: Record<string, unknown> = {
      a: 1,
      previous_receipt_hash: FIRST_RECEIPT_SEED,
    };
    const result = verifyChain([{ signedEnvelope: env }]);
    expect(result.chainIntegrity).toBe(true);
  });

  it("throws when given an unsupported option", () => {
    const env = {
      signature_id: "sig_1",
      action_type: "api:call",
      timestamp: 1,
      previousReceiptHash: FIRST_RECEIPT_SEED,
    };
    expect(() =>
      verifyChain([{ signedEnvelope: env }], { unknownFlag: true } as never),
    ).toThrow(TypeError);
  });

  // H12: chain digest scope matches cloud (payload-only, not full envelope).
  it("unwraps bundle-shaped envelopes and hashes only `payload`", () => {
    const p0 = makeEnvelope(0, FIRST_RECEIPT_SEED);
    const h0 = sha256Hex(canonicalJson(p0));
    const p1 = makeEnvelope(1, h0);

    const bundle0: Record<string, unknown> = {
      payload: p0,
      signature: { alg: "ML-DSA-65", kid: "kid_test", sig: "b64sig" },
      anchors: [],
    };
    const bundle1: Record<string, unknown> = {
      payload: p1,
      signature: { alg: "ML-DSA-65", kid: "kid_test", sig: "b64sig" },
      anchors: [],
    };

    const result = verifyChain([
      { signedEnvelope: bundle0 },
      { signedEnvelope: bundle1 },
    ]);
    expect(result.chainIntegrity).toBe(true);
    expect(result.steps[1].expectedPreviousReceiptHash).toBe(h0);
  });

  it("detects a single forged link in a 3-receipt chain", () => {
    const records = buildChain(3);
    // Forge: rewrite the third record's previousReceiptHash to junk.
    (records[2].signedEnvelope as Record<string, unknown>).previousReceiptHash =
      "f".repeat(64);
    const result = verifyChain(records);
    expect(result.chainIntegrity).toBe(false);
    expect(result.steps[0].chainValid).toBe(true);
    expect(result.steps[1].chainValid).toBe(true);
    expect(result.steps[2].chainValid).toBe(false);
  });
});
