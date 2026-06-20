/**
 * THE GATE - the TypeScript verifier must reproduce the proven Python oracle's
 * verdicts across the whole conformance corpus, byte-match the upstream
 * canonicalization vectors, and verify the real Authproof receipt.
 *
 * These are reproduced artifacts, not assertions: the corpus, the canonicalization
 * vectors, and the real-SDK receipt are the same files the Python oracle is gated
 * on.
 */

import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";

import { asqavJcs, jcsRfc8785, parseJsonPreservingFloats } from "../src/verifier/canonical.js";
import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { runCorpus, tolerated } from "../src/verifier/runner.js";

// typescript/tests -> repo root -> verifier/conformance-vectors
const CORPUS_ROOT = resolve(__dirname, "..", "..", "verifier", "conformance-vectors");

function loadJson(path: string): Record<string, unknown> {
  // Float-preserving so `500.0` literals survive to the canonicaliser.
  return parseJsonPreservingFloats(readFileSync(path, "utf-8")) as Record<string, unknown>;
}

describe("verifier parity gate (THE GATE)", () => {
  it("matches every manifest outcome across all 47 corpus vectors", () => {
    const results = runCorpus(CORPUS_ROOT);
    const mismatches = results.filter((r) => !tolerated(r));
    const passed = results.filter(tolerated).length;

    // Print the same per-vector report the Python runner.main() prints.
    const report = results
      .map((r) => {
        const mark = tolerated(r) ? "ok" : "FAIL";
        const line = `  [${mark.padStart(4)}] ${r.dir.padEnd(38)} expect=${r.expectedOutcome.padEnd(11)} got=${r.actualVerdict}`;
        return tolerated(r) ? line : `${line}\n         ${r.detail}`;
      })
      .join("\n");
    // eslint-disable-next-line no-console
    console.log(`\n${report}\n\n  => ${passed}/${results.length} vectors matched expected outcome\n`);

    expect(results.length).toBe(47);
    expect(mismatches, `mismatched vectors: ${mismatches.map((m) => m.dir).join(", ")}`).toEqual([]);
    expect(passed).toBe(47);
  });
});

describe("jcs_rfc8785 byte-parity vs upstream canonicalization vectors", () => {
  it("byte-matches every upstream canonicalization vector", () => {
    const path = join(CORPUS_ROOT, "agentreceipts-upstream-interop", "canonicalization_vectors.json");
    const data = loadJson(path) as { canonicalization_vectors: Array<{ name: string; input: unknown; canonical: string }> };
    const vectors = data.canonicalization_vectors;
    const mismatches: Array<{ name: string; want: string; got: string }> = [];
    for (const v of vectors) {
      const got = new TextDecoder().decode(jcsRfc8785(v.input));
      if (got !== v.canonical) mismatches.push({ name: v.name, want: v.canonical, got });
    }
    // eslint-disable-next-line no-console
    console.log(`\n  jcs_rfc8785 byte-parity: ${vectors.length - mismatches.length}/${vectors.length} vectors matched\n`);
    expect(mismatches, JSON.stringify(mismatches)).toEqual([]);
    expect(vectors.length).toBeGreaterThanOrEqual(32);
  });
});

describe("real Authproof receipt verifies PASS (ES256 path)", () => {
  it("authproof-01-genesis-real-sdk -> PASS", () => {
    const dir = join(CORPUS_ROOT, "authproof-01-genesis-real-sdk");
    const receipt = loadJson(join(dir, "receipt.json"));
    const result = verify(receipt, ADAPTERS);
    // eslint-disable-next-line no-console
    console.log(`\n  authproof-01-genesis-real-sdk: fmt=${result.fmt} verdict=${result.verdict}\n`);
    expect(result.fmt).toBe("authproof");
    expect(result.verdict).toBe("PASS");
  });
});

describe("canonical-bytes cross-check (TS signing_input sha256 == Python)", () => {
  // Each TS signing_input sha256 is pinned to the digest the Python oracle's adapter
  // produces for the same receipt, so a canonicaliser drift on either side reddens here.
  const pinned: Record<string, { fmt: string; sha: string }> = {
    "aerf-01-genesis": {
      fmt: "aerf",
      sha: "b0c0ed0d2570cdebb9f87868996497f08747063a2717e5b38db79fae9c87b344",
    },
    "acta-01-genesis": {
      fmt: "acta",
      sha: "8cc38bc400870eb0ac5491b60468fe3b3f4a13dbe5dce3da0f0952c109de0456",
    },
    "agentreceipts-01-didkey-genesis": {
      fmt: "agentreceipts",
      sha: "eb5fd119afbd399658e615cd4687c0047c72dd3bb3c59bbf40f69b7a12e66a34",
    },
    "asqav-01-genesis-permit": {
      fmt: "asqav-native",
      sha: "88051bbc8ba5f41fd1626ade433b923d12ba70a1767201ee78c8b407fc56b580",
    },
  };
  for (const [vec, { fmt, sha }] of Object.entries(pinned)) {
    it(`TS signing_input byte-matches Python for ${vec}`, () => {
      const receipt = loadJson(join(CORPUS_ROOT, vec, "receipt.json"));
      const ad = ADAPTERS.find((a) => a.detect(receipt))!;
      expect(ad.name).toBe(fmt);
      const bytes = ad.signingInput(receipt);
      expect(createHash("sha256").update(Buffer.from(bytes)).digest("hex")).toBe(sha);
    });
  }
});

describe("parseJsonPreservingFloats matches JSON.parse structure (Node 18+ safe)", () => {
  // The parser is hand-rolled (no Node 21+ reviver context.source); it must agree
  // with JSON.parse on everything except the deliberate float/big-int preservation.
  const unwrap = (v: unknown): unknown => {
    if (v && typeof v === "object" && "value" in v && Object.keys(v).length === 1) return (v as { value: number }).value;
    if (v && typeof v === "object" && "source" in v && Object.keys(v).length === 1) return Number((v as { source: string }).source);
    if (Array.isArray(v)) return v.map(unwrap);
    if (v && typeof v === "object") {
      return Object.fromEntries(Object.entries(v as Record<string, unknown>).map(([k, x]) => [k, unwrap(x)]));
    }
    return v;
  };
  const cases = [
    '{}',
    '[]',
    '{"a":1,"b":[true,false,null],"c":{"d":"e"}}',
    '{"s":"a \\"quoted\\" \\\\ slash \\/ tab\\t newline\\n unicode \\u00e9 \\ud83d\\ude00"}',
    '{"nested":[[[[1]]]],"mix":[{"x":[2,{"y":3}]}]}',
    '{"num":[0,-0,1,-1,3.14,-2.5,1e3,1E-3,6.022e23],"big":[12345,9007199254740991]}',
    '"bare string"',
    '42',
    'true',
    '  {  "spaced"  :  [ 1 , 2 ]  }  ',
  ];
  for (const c of cases) {
    it(`agrees with JSON.parse for ${c.slice(0, 40)}`, () => {
      expect(unwrap(parseJsonPreservingFloats(c))).toEqual(JSON.parse(c));
    });
  }
  it("rejects malformed JSON, strict RFC 8259 numbers, and over-deep nesting", () => {
    for (const bad of ['{"a":}', "[1,2", "{} junk", "", "undefined", "[1,,2]", "[1,2,]"]) {
      expect(() => parseJsonPreservingFloats(bad), bad).toThrow();
    }
    // strict number grammar, matching Python json.loads
    for (const bad of ['{"a":01}', '{"a":5.}', '{"a":.5}', '{"a":-}', '{"a":1e}', '{"a":+5}', '{"a":1.e5}']) {
      expect(() => parseJsonPreservingFloats(bad), bad).toThrow();
    }
    // recursion-depth guard rejects rather than overflowing the stack
    expect(() => parseJsonPreservingFloats("[".repeat(5000) + "]".repeat(5000))).toThrow();
  });
});

describe("integers beyond IEEE-754 safe range stay distinct (no false-PASS)", () => {
  const dec = new TextDecoder();
  it("emits the exact source digits in every dialect, matching Python str(int)", () => {
    const o = parseJsonPreservingFloats('{"n":9007199254740993}');
    expect(dec.decode(asqavJcs(o))).toBe('{"n":9007199254740993}');
    expect(dec.decode(jcsRfc8785(o))).toBe('{"n":9007199254740993}');
  });
  it("does not collapse 2^53+1 onto 2^53", () => {
    const a = dec.decode(asqavJcs(parseJsonPreservingFloats('{"n":9007199254740993}')));
    const b = dec.decode(asqavJcs(parseJsonPreservingFloats('{"n":9007199254740992}')));
    expect(a).not.toBe(b);
  });
});
