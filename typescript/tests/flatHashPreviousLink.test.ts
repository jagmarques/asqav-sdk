import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { verifyReceiptOffline } from "../src/index.js";
import { flatSignedFields } from "../src/verifier/adapters/asqavNative.js";
import {
  FAILURE_INVALID,
  VERDICT_UNVERIFIED,
  VERDICT_VERIFIED,
  type VerifyResult,
} from "../src/verifier/core.js";

const FIXTURE = resolve(__dirname, "..", "..", "python", "tests", "fixtures", "flat_hash_prev_link.json");

interface Fixture {
  jwks: Record<string, unknown>;
  receipts: Record<string, Record<string, unknown>>;
}

function load(): Fixture {
  return JSON.parse(readFileSync(FIXTURE, "utf-8")) as Fixture;
}

function verify(name: string, tamper?: (receipt: Record<string, unknown>) => void): VerifyResult {
  const data = load();
  const receipt = data.receipts[name];
  if (tamper !== undefined) tamper(receipt);
  return verifyReceiptOffline(receipt, data.jwks);
}

function linkName(): string {
  const data = load();
  const signed = Object.keys(flatSignedFields(data.receipts.linked));
  const legacy = new Set(Object.keys(flatSignedFields(data.receipts.legacy)));
  const [link] = signed.filter((field) => !legacy.has(field));
  expect(link).toBeTypeOf("string");
  return link;
}

describe("flat hash-mode previous link", () => {
  it("cloud hash receipt with previous link verifies", () => {
    for (const name of ["linked", "null_link"]) {
      expect(verify(name).verdict).toBe(VERDICT_VERIFIED);
    }
  });

  it("legacy hash receipt without link verifies", () => {
    expect(verify("legacy").verdict).toBe(VERDICT_VERIFIED);
    const fields = Object.keys(flatSignedFields(load().receipts.legacy));
    expect(fields).not.toContain(linkName());
  });

  // NEGCTL: the linked body's own link, changed, is refused
  it("tampered previous_receipt_hash fails", () => {
    const link = linkName();
    const result = verify("linked", (receipt) => {
      receipt[link] = "f".repeat(64);
    });
    expect(result.verdict).toBe(VERDICT_UNVERIFIED);
    expect(result.failureClass).toBe(FAILURE_INVALID);
  });

  // NEGCTL: a link the 11-field signature never covered is refused
  it("unsigned link on legacy receipt fails", () => {
    const link = linkName();
    const result = verify("legacy", (receipt) => {
      receipt[link] = "f".repeat(64);
    });
    expect(result.verdict).toBe(VERDICT_UNVERIFIED);
    expect(result.failureClass).toBe(FAILURE_INVALID);
  });
});
