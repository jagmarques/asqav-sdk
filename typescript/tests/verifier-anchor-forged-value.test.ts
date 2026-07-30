/**
 * A forged anchor value must never read as present, mirroring the Python half
 */

import { describe, expect, it } from "vitest";

import { checkAnchors } from "../src/verifier/vrShim.js";

const ENVELOPE = {
  payload: { type: "protectmcp:decision", issued_at: "2026-06-19T00:00:00+00:00" },
  signature: { alg: "ML-DSA-65", kid: "k1", sig: "AAAA" },
};

const LEGITIMATE = [
  "AA",
  "AAA",
  "AAAA",
  "YQ==",
  "MTIzNA==",
  "AAAA-_AA",
  "AAAA_synthetic_tsr_base64_placeholder_AAAA",
  "A".repeat(64),
];

const FORGED = [
  "!!!!",
  "@@@@",
  "....",
  "****",
  "!",
  "!!",
  "!!!",
  "=",
  "==",
  "===",
  "====",
  " ",
  "\t",
  "\n",
  "\r\n",
  "<script>alert(1)</script>",
  "YQ==!!!!",
  "!!!!YQ==",
  "YQ==YQ==",
  "MTIzNA==\n",
  "\x00",
  "é",
  "中文",
  "😀",
  " ",
];

const NON_STRINGS: unknown[] = [null, undefined, 0, 123, 1.5, true, [], {}, ["AAAA"]];

function axisFor(value: unknown): readonly [string, string] {
  return checkAnchors({ ...ENVELOPE, anchors: [{ type: "rfc3161", value }] });
}

describe("forged anchor value", () => {
  it("refuses the reported all-punctuation value", () => {
    const [state, note] = axisFor("!!!!");
    expect(state).toBe("FAIL");
    expect(note).not.toContain("base64-ok");
  });

  it.each(FORGED)("never reads %j as present", (value) => {
    const [state, note] = axisFor(value);
    expect(state).toBe("FAIL");
    expect(note).not.toContain("base64-ok");
  });

  it.each(LEGITIMATE)("still accepts %j", (value) => {
    const [state, note] = axisFor(value);
    expect(state).toBe("PASS");
    expect(note).toContain("present, base64-ok");
  });

  it.each(NON_STRINGS.map((v) => [v]))("never reads non-string %j as present", (value) => {
    expect(axisFor(value)[0]).toBe("FAIL");
  });

  it("does not let a valid sibling launder a forged anchor", () => {
    const env = {
      ...ENVELOPE,
      anchors: [
        { type: "rfc3161", value: "AAAA" },
        { type: "forged", value: "!!!!" },
      ],
    };
    expect(checkAnchors(env)[0]).toBe("FAIL");
  });
});
