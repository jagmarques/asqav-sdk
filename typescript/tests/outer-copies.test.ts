/**
 * Unsigned outer copies of signed fields (criterion 742). A copy that
 * disagrees with the signed payload must fail the verdict, and the shared
 * case table must give the Python half's answer, note included.
 */

import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { verifyReceiptOffline } from "../src/index.js";
import { checkOuterCopies, b64decode } from "../src/verifier/vrShim.js";

const ROOT = resolve(__dirname, "..", "..");
const VECTORS = join(ROOT, "verifier", "conformance-vectors");
const FIXTURES = join(ROOT, "python", "tests", "fixtures");

function load(path: string): Record<string, any> {
  return JSON.parse(readFileSync(path, "utf-8"));
}

function vector(name: string): [Record<string, any>, Record<string, any>] {
  return [load(join(VECTORS, name, "receipt.json")), load(join(VECTORS, name, "jwks.json"))];
}

function prod(): [Record<string, any>, Record<string, any>] {
  return [
    load(join(FIXTURES, "prod_audit_pack_receipt_2026_09_02.json")),
    load(join(FIXTURES, "prod_audit_pack_jwks_2026_09_02.json")),
  ];
}

const CASES = load(join(ROOT, "verifier", "outer-copy-cases.json")).cases as Array<Record<string, any>>;

function axis(result: ReturnType<typeof verifyReceiptOffline>, name: string) {
  return result.axes.find((a) => a.axis === name);
}

describe("outer-copy case table (shared with Python)", () => {
  it("is populated and covers both results", () => {
    expect(CASES.length).toBeGreaterThanOrEqual(50);
    expect(new Set(CASES.map((c) => c.expect.result))).toEqual(new Set(["PASS", "FAIL"]));
  });

  it.each(CASES.map((c) => [c.name, c] as const))("%s", (_name, c) => {
    const key = c.key_b64 === null ? null : b64decode(c.key_b64);
    const [result, note] = checkOuterCopies(c.raw, b64decode(c.message_b64), key);
    expect({ result, note }).toEqual(c.expect);
  });
});

describe("edited outer copies of a real prod export", () => {
  it.each([
    ["signed_at", "2026-03-03T00:00:00"],
    ["agent_id", "agt_someoneElse000000"],
    ["action_id", "act_forgedForgedForgedFo"],
    ["algorithm", "ML-DSA-87"],
    ["compliance_mode", false],
  ])("%s is refused as invalid while the signature still verifies", (member, value) => {
    const [receipt, jwks] = prod();
    receipt[member as string] = value;
    const result = verifyReceiptOffline(receipt, jwks);
    expect(axis(result, "outer_copies")?.result).toBe("FAIL");
    expect(axis(result, "outer_copies")?.failureClass).toBe("invalid");
    expect(axis(result, "signature")?.result).toBe("PASS");
    expect(result.verdict).toBe("unverified");
    expect(result.failureClass).toBe("invalid");
  });

  it("passes the axis on the untouched export and lists what is unattested", () => {
    const [receipt, jwks] = prod();
    const outer = axis(verifyReceiptOffline(receipt, jwks), "outer_copies");
    expect(outer?.result).toBe("PASS");
    expect(outer?.note).toContain("no signed counterpart, do not rely on: action_type, signature_id");
  });

  it("does not treat an unattested or a removed copy as a mismatch", () => {
    const [receipt, jwks] = prod();
    receipt.action_type = "anything:at:all";
    receipt.signature_id = "sig_not_the_real_one";
    delete receipt.signed_at;
    delete receipt.message_b64;
    expect(axis(verifyReceiptOffline(receipt, jwks), "outer_copies")?.result).toBe("PASS");
  });
});

describe("outer-copy conformance vectors asqav-43..48", () => {
  it("the control vector verifies", () => {
    const [receipt, jwks] = vector("asqav-43-outer-copies-agree");
    const result = verifyReceiptOffline(receipt, jwks);
    expect(result.verdict).toBe("verified");
    expect(axis(result, "outer_copies")?.result).toBe("PASS");
  });

  it.each([
    "asqav-44-outer-signed-at-edited",
    "asqav-45-outer-agent-id-edited",
    "asqav-46-outer-action-type-edited",
    "asqav-47-outer-message-copy-edited",
    "asqav-48-outer-public-key-edited",
  ])("%s is refused", (name) => {
    const [receipt, jwks] = vector(name);
    const result = verifyReceiptOffline(receipt, jwks);
    expect(result.verdict).toBe("unverified");
    expect(result.failureClass).toBe("invalid");
    expect(axis(result, "outer_copies")?.result).toBe("FAIL");
    expect(axis(result, "signature")?.result).toBe("PASS");
  });
});
