import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { verifyReceiptOffline } from "../src/index.js";
import { flatSignedFields } from "../src/verifier/adapters/asqavNative.js";
import { b64decode, checkOuterCopies } from "../src/verifier/vrShim.js";
import {
  FAILURE_INVALID,
  VERDICT_UNVERIFIED,
  VERDICT_VERIFIED,
  type VerifyResult,
} from "../src/verifier/core.js";

const FIXTURE = resolve(__dirname, "..", "..", "python", "tests", "fixtures", "flat_hash_outcome_members.json");
const TWELVE = [
  "v", "mode", "hash", "hash_algo", "metadata", "server_timestamp", "action_id",
  "agent_id", "org_id", "policy_digest", "policy_decision", "previous_receipt_hash",
];
const OPTIONAL: Record<string, string[]> = {
  plain: [],
  result_digest: ["result_digest"],
  action_ref: ["action_ref"],
  invocation_ref: ["invocation_ref"],
  all_three: ["result_digest", "action_ref", "invocation_ref"],
};

interface Fixture {
  jwks: { keys: Array<{ kid: string; public_key: string }> };
  receipts: Record<string, Record<string, unknown>>;
}

function load(name: string, tamper?: (receipt: Record<string, unknown>) => void) {
  const data = JSON.parse(readFileSync(FIXTURE, "utf-8")) as Fixture;
  const receipt = data.receipts[name];
  if (tamper !== undefined) tamper(receipt);
  return { data, receipt };
}

function verify(name: string, tamper?: (receipt: Record<string, unknown>) => void): VerifyResult {
  const { data, receipt } = load(name, tamper);
  return verifyReceiptOffline(receipt, data.jwks);
}

describe("flat hash-mode outcome members", () => {
  for (const [name, members] of Object.entries(OPTIONAL)) {
    it(`${name} receipt verifies and rebuilds the twelve plus its outcome members`, () => {
      expect(verify(name).verdict).toBe(VERDICT_VERIFIED);
      expect(Object.keys(flatSignedFields(load(name).receipt))).toEqual([...TWELVE, ...members]);
    });
  }

  // NEGCTL: each signed outcome member, changed after signing, is refused
  for (const member of OPTIONAL.all_three) {
    it(`tampered ${member} fails`, () => {
      expect(verify("all_three").verdict).toBe(VERDICT_VERIFIED);
      const result = verify("all_three", (receipt) => {
        receipt[member] = "f".repeat(64);
      });
      expect(result.verdict).toBe(VERDICT_UNVERIFIED);
      expect(result.failureClass).toBe(FAILURE_INVALID);
    });
  }

  // NEGCTL: an outcome member the signature never covered is refused
  it("outcome member added after signing fails", () => {
    expect(verify("plain").verdict).toBe(VERDICT_VERIFIED);
    const result = verify("plain", (receipt) => {
      receipt.result_digest = `sha256:${"d".repeat(64)}`;
    });
    expect(result.verdict).toBe(VERDICT_UNVERIFIED);
    expect(result.failureClass).toBe(FAILURE_INVALID);
  });

  it("outer-copies note leaves the signed outcome members out of the unsigned list", () => {
    const { data, receipt } = load("all_three");
    const [key] = data.jwks.keys;
    const note = checkOuterCopies(receipt, new Uint8Array(), b64decode(key.public_key), key.kid)[1];
    for (const member of OPTIONAL.all_three) expect(note).not.toContain(member);
  });
});
