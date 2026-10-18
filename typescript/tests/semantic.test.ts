/** Independent verdict semantics tests; tables derive from normative rules */
/** Pins: literals, fold truth table, classification; corpus via parity gate */
import { describe, expect, it } from "vitest";

import { FAIL, PASS, SKIPPED } from "../src/verifier/crypto.js";
import {
  FAILURE_INVALID,
  FAILURE_UNVERIFIABLE,
  VERDICT_UNVERIFIED,
  VERDICT_VERIFIED,
  VERDICT_VERIFIED_KEYED,
  type AxisResult,
} from "../src/verifier/core.js";
import { classifyFailure, foldIndependent } from "../src/verifier/semantic.js";

function ax(axis: string, result: string, failureClass: string | null, note = ""): AxisResult {
  return { axis, result: result as AxisResult["result"], note: note || `${axis} ${result}`, failureClass: failureClass as AxisResult["failureClass"] };
}

const BASE = ["structure", "signature", "chain", "seq"];
const pass = (axes: string[]): AxisResult[] => axes.map((a) => ax(a, PASS, null));

describe("semantic vocabulary pins", () => {
  it("literals match the core surface", () => {
    expect(VERDICT_VERIFIED).toBe("verified");
    expect(VERDICT_VERIFIED_KEYED).toBe("verified_keyed");
    expect(VERDICT_UNVERIFIED).toBe("unverified");
    expect(FAILURE_INVALID).toBe("invalid");
    expect(FAILURE_UNVERIFIABLE).toBe("unverifiable");
  });
});

describe("fold truth table", () => {
  const rows: Array<[AxisResult[], boolean, string, string | null]> = [
    [pass(BASE), false, "verified", null],
    [pass(BASE), true, "verified_keyed", null],
    [[...pass(BASE), ax("expiry", FAIL, FAILURE_INVALID)], false, "verified", null],
    [[...pass(BASE), ax("expiry", FAIL, FAILURE_INVALID)], true, "verified_keyed", null],
    [[...pass(["structure", "signature", "seq"]), ax("chain", SKIPPED, FAILURE_UNVERIFIABLE)], false, "verified", null],
    [[...pass(["structure", "chain", "seq"]), ax("signature", SKIPPED, FAILURE_UNVERIFIABLE)], false, "unverified", "unverifiable"],
    [[...pass(["structure", "chain", "seq"]), ax("signature", FAIL, FAILURE_INVALID)], false, "unverified", "invalid"],
    [[...pass(["structure", "chain", "seq"]), ax("signature", FAIL, FAILURE_UNVERIFIABLE)], false, "unverified", "unverifiable"],
    [
      [...pass(["structure", "seq"]), ax("signature", FAIL, FAILURE_INVALID), ax("chain", FAIL, FAILURE_UNVERIFIABLE)],
      false, "unverified", "invalid",
    ],
    [
      [...pass(["structure", "chain", "seq"]), ax("signature", FAIL, FAILURE_INVALID)],
      true, "unverified", "invalid",
    ],
  ];
  for (const [axes, keyed, verdict, failureClass] of rows) {
    it(`folds ${JSON.stringify(axes.map((a) => `${a.axis}:${a.result}`))} keyed=${keyed}`, () => {
      expect(foldIndependent(axes, keyed)).toEqual([verdict, failureClass]);
    });
  }
});

describe("classification truth table", () => {
  const invalidAxes = [
    "signature", "key_status", "issuer_bind", "key_binding",
    "counterparty", "payload_digest", "outer_copies", "nonce",
    "seq", "parent_signature", "pdp_signature", "anchors",
  ];
  for (const axis of invalidAxes) {
    it(`${axis} FAIL is invalid`, () => {
      expect(classifyFailure(axis, FAIL, "proven break")).toBe(FAILURE_INVALID);
    });
  }
  const rows: Array<[string, string, string]> = [
    ["chain", "chain break: expected ab.. got cd..", "invalid"],
    ["chain", "predecessor is a different receipt format", "invalid"],
    ["chain", "predecessor payload not canonicalisable", "unverifiable"],
    ["skew", "unparseable issued_at 'x'", "unverifiable"],
    ["skew", "issued_at 9999s ahead of wall clock", "invalid"],
    ["structure", "unsupported ACTA alg ed25519", "invalid"],
    ["structure", "key_purpose mismatch", "invalid"],
    ["structure", "signing-key DID != issuer DID", "invalid"],
    ["structure", "proof @context is not a prefix", "invalid"],
    ["structure", "unregistered type foo", "invalid"],
    ["structure", "missing required fields: sig", "unverifiable"],
    ["expiry", "unreadable expires_at 'x'", "unverifiable"],
    ["expiry", "expires_at lapsed 5s ago", "invalid"],
    ["issuer_key", "anything", "unverifiable"],
    ["brand_new_axis", "anything", "unverifiable"],
  ];
  for (const [axis, note, cls] of rows) {
    it(`${axis} note branch maps correctly`, () => {
      expect(classifyFailure(axis, FAIL, note)).toBe(cls);
    });
  }
  it("PASS and SKIPPED never prove a break", () => {
    expect(classifyFailure("signature", PASS, "x")).toBeNull();
    expect(classifyFailure("signature", SKIPPED, "no key")).toBe(FAILURE_UNVERIFIABLE);
  });
});
