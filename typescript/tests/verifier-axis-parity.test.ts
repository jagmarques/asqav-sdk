/**
 * Anchor-binding, clock-skew, and nesting-depth parity, TypeScript half.
 *
 * One JSON table drives both verifiers, pinned to the output of the Python
 * `check_anchors` / `check_skew`. The Python half lives in
 * python/tests/test_axis_parity_cases.py and reads the same file.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { verifyReceiptOffline } from "../src/index.js";
import { MAX_NESTING_DEPTH } from "../src/verifier/core.js";
import { checkAnchors, checkSkew, SKEW_BOUND_SECONDS } from "../src/verifier/vrShim.js";

interface AnchorCase {
  name: string;
  envelope: Record<string, unknown>;
  expect: { result: string; note: string };
}

interface SkewCase {
  name: string;
  issued_at: string;
  expect: { result: string; note_contains: string };
}

const CASES_FILE = resolve(__dirname, "..", "..", "verifier", "axis-parity-cases.json");
const TABLE = JSON.parse(readFileSync(CASES_FILE, "utf-8")) as {
  anchors: AnchorCase[];
  skew: SkewCase[];
};

/** A receipt whose only defect is `depth` levels of nesting outside the signed payload. */
function nestedReceipt(depth: number): Record<string, unknown> {
  let node: Record<string, unknown> = { leaf: 1 };
  for (let i = 0; i < depth; i++) node = { n: node };
  return {
    payload: {
      type: "protectmcp:decision",
      issued_at: "2026-06-19T00:00:00.000000Z",
      issuer_id: "f94f66c0-c580-432d-a041-29374f7aee07",
      agent_id: "agt_1",
      action_ref: "sha256:8888888888888888888888888888888888888888888888888888888888888888",
      payload_digest: { hash: "88", size: 512 },
      policy_digest: "sha256:3333333333333333333333333333333333333333333333333333333333333333",
      previousReceiptHash: "0".repeat(64),
      decision: "allow",
    },
    signature: { alg: "ML-DSA-65", kid: "k1", sig: "AAAA" },
    anchors: [],
    junk: node,
  };
}

const JWKS = {
  keys: [
    {
      kid: "k1",
      agent_id: "agt_1",
      issuer_id: "f94f66c0-c580-432d-a041-29374f7aee07",
      alg: "ML-DSA-65",
      status: "active",
      public_key: "AAAA",
    },
  ],
};

describe("anchor-binding axis parity", () => {
  it("is populated", () => {
    expect(TABLE.anchors.length).toBeGreaterThanOrEqual(25);
  });

  for (const c of TABLE.anchors) {
    it(c.name, () => {
      const [result, note] = checkAnchors(c.envelope);
      expect(result, c.name).toBe(c.expect.result);
      // The digest in the note is the JCS of the envelope minus anchors, so an
      // exact match also proves the two canonicalisers agree byte for byte.
      expect(note, c.name).toBe(c.expect.note);
    });
  }
});

describe("clock-skew axis parity", () => {
  it("keeps the Python bound", () => {
    expect(SKEW_BOUND_SECONDS).toBe(300);
    expect(TABLE.skew.length).toBeGreaterThanOrEqual(15);
  });

  for (const c of TABLE.skew) {
    it(c.name, () => {
      const [result, note] = checkSkew(c.issued_at);
      expect(result, `${c.name}: note ${note}`).toBe(c.expect.result);
      expect(note, c.name).toContain(c.expect.note_contains);
    });
  }

  it("rejects a non-string stamp", () => {
    expect(checkSkew(undefined)[0]).toBe("FAIL");
    expect(checkSkew(1700000000)[0]).toBe("FAIL");
  });
});

describe("nesting-depth gate parity", () => {
  it("keeps the Python cap", () => {
    expect(MAX_NESTING_DEPTH).toBe(200);
  });

  // The envelope key and the leaf each add a level, so 198 wrappers sit exactly at
  // the cap and 199 cross it. Python answers the same way for the same two docs.
  it("passes a receipt at the cap", () => {
    const r = verifyReceiptOffline(nestedReceipt(MAX_NESTING_DEPTH - 2), JWKS);
    expect(r.axes.find((a) => a.axis === "structure")?.result).toBe("PASS");
  });

  it("caps a receipt one level past it", () => {
    const r = verifyReceiptOffline(nestedReceipt(MAX_NESTING_DEPTH - 1), JWKS);
    expect(r.verdict).toBe("INCOMPLETE");
    expect(r.axes).toHaveLength(1);
    expect(r.axes[0].axis).toBe("structure");
    expect(r.axes[0].result).toBe("FAIL");
    expect(r.axes[0].note).toBe("receipt nesting exceeds the supported depth (> 200 levels)");
  });

  it("caps an over-nested predecessor", () => {
    const r = verifyReceiptOffline(nestedReceipt(0), JWKS, nestedReceipt(5000));
    expect(r.verdict).toBe("INCOMPLETE");
    expect(r.axes[0].note).toBe("receipt nesting exceeds the supported depth (> 200 levels)");
  });

  it("returns a verdict instead of exhausting the stack", () => {
    // Nesting inside the signed payload reaches the recursive JCS encoder, which
    // throws RangeError out of the call without the gate.
    let node: Record<string, unknown> = { leaf: 1 };
    for (let i = 0; i < 20000; i++) node = { n: node };
    const receipt = nestedReceipt(0);
    (receipt.payload as Record<string, unknown>).junk = node;
    expect(() => verifyReceiptOffline(receipt, JWKS)).not.toThrow();
    expect(verifyReceiptOffline(receipt, JWKS).verdict).toBe("INCOMPLETE");
  });
});
