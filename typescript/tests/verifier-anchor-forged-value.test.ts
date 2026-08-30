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

// Codepoints that read like base64 to a human but sit outside the alphabet
const LOOKALIKE = ["ＡＢＣＤ", "АВСЕ", "MTIzNА==", "⁰¹²³", "MTIzNA==​", "﻿MTIzNA==", "MTIz​NA=="];

// Surplus padding on real base64: Python's b64decode(validate=True) accepts
// these on 3.11 and raises on 3.12, so this half pins the answer
const EXCESS_PADDING = [
  "AAAA=",
  "AAAA==",
  "AAAA===",
  "AAAA====",
  "AAAA=====",
  "AAAA======",
  "MTIzNA====",
  "dGVzdA====",
  "/NTpk6v8HIk8U2RJ/JRrGsPlghKY=",
  "/NTpk6v8HIk8U2RJ/JRrGsPlghKY====",
  "YW5j-aG9y-IHBh-eWxv-YWQg-aGVy-ZQ==",
  "YW5j-aG9y-IHBh-eWxv-YWQg-aGVy-ZQ====",
];

// What GNU base64 and openssl base64 emit by default, refused on purpose
function mimeWrap(raw: Buffer): string {
  return `${(raw.toString("base64").match(/.{1,76}/g) ?? []).join("\n")}\n`;
}
const MIME_WRAPPED = [
  mimeWrap(Buffer.from(Array.from({ length: 200 }, (_, i) => i))),
  mimeWrap(Buffer.from(Array.from({ length: 64 }, (_, i) => i))),
  "MTIzNA==\r\nMTIzNA==",
];

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

  // Genuine base64 passes the shape gate, but shape is not proof: since the
  // cryptographic anchor check landed, a non-token value reports unverifiable.
  it.each(LEGITIMATE)("still accepts %j", (value) => {
    const [state, note] = axisFor(value);
    expect(state).toBe("SKIPPED");
    expect(note).toContain("present, base64-ok");
    expect(note).toContain("unverifiable");
  });

  it.each(NON_STRINGS.map((v) => [v]))("never reads non-string %j as present", (value) => {
    expect(axisFor(value)[0]).toBe("FAIL");
  });

  it.each(LOOKALIKE)("refuses lookalike codepoints in %j", (value) => {
    expect(axisFor(value)[0]).toBe("FAIL");
  });

  it.each(EXCESS_PADDING)("refuses surplus padding on %j", (value) => {
    const [state, note] = axisFor(value);
    expect(state).toBe("FAIL");
    expect(note).not.toContain("base64-ok");
  });

  it.each(MIME_WRAPPED)("refuses MIME line-wrapped %j", (value) => {
    expect(axisFor(value)[0]).toBe("FAIL");
  });

  it("keeps every encoding a real signer emits", () => {
    // Every encoding passes the shape gate; the crypto check then reports
    // unverifiable for a non-token, never PASS.
    for (let n = 1; n <= 128; n++) {
      const raw = Buffer.from(Array.from({ length: n }, (_, i) => (n * 31 + i * 7) % 256));
      const std = raw.toString("base64");
      const url = raw.toString("base64url");
      for (const value of [std, std.replace(/=+$/, ""), url, url.replace(/=+$/, "")]) {
        expect(axisFor(value)[0], value).toBe("SKIPPED");
      }
    }
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
