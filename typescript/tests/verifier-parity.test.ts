/**
 * THE GATE: the TS verifier must reproduce the Python oracle's verdicts across the corpus,
 * byte-match the upstream canonicalization vectors, and verify the real Authproof receipt.
 */

import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";

import { asqavJcs, jcsRfc8785, parseJsonPreservingFloats } from "../src/verifier/canonical.js";
import { AXIS_ORDER_PREFIX, verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import {
  keyProviderFor,
  loadJson as runnerLoadJson,
  runCorpus,
  runOne,
  tolerated,
} from "../src/verifier/runner.js";

// typescript/tests -> repo root -> verifier/conformance-vectors
const CORPUS_ROOT = resolve(__dirname, "..", "..", "verifier", "conformance-vectors");

function loadJson(path: string): Record<string, unknown> {
  // Float-preserving so `500.0` literals survive to the canonicaliser.
  return parseJsonPreservingFloats(readFileSync(path, "utf-8")) as Record<string, unknown>;
}

describe("verifier parity gate (THE GATE)", () => {
  it("matches every manifest outcome across all 73 corpus vectors", () => {
    const results = runCorpus(CORPUS_ROOT);
    const mismatches = results.filter((r) => !tolerated(r));
    const passed = results.filter(tolerated).length;

    // Print the same per-vector report the Python runner.main() prints.
    const report = results
      .map((r) => {
        const mark = tolerated(r) ? "ok" : "FAIL";
        const got = r.actualFailureClass !== "" ? `${r.actualVerdict} (${r.actualFailureClass})` : r.actualVerdict;
        const line = `  [${mark.padStart(4)}] ${r.dir.padEnd(38)} expect=${r.expectedOutcome.padEnd(16)} got=${got}`;
        return tolerated(r) ? line : `${line}\n         ${r.detail}`;
      })
      .join("\n");
    // eslint-disable-next-line no-console
    console.log(`\n${report}\n\n  => ${passed}/${results.length} vectors matched expected outcome\n`);

    expect(results.length).toBe(73);
    expect(mismatches, `mismatched vectors: ${mismatches.map((m) => m.dir).join(", ")}`).toEqual([]);
    expect(passed).toBe(73);
  });

  it("pins failure_class byte-for-byte with the Python oracle for every unverified vector", () => {
    // Criteria 418/438: invalid and unverifiable must never collapse, and both languages
    // must agree which class each failing vector lands in. The optional-dep vector is skipped.
    const results = runCorpus(CORPUS_ROOT);
    const unverified = results.filter(
      (r) => r.expectedOutcome === "unverified" && r.reasonCode !== "signature_skipped_no_dilithium",
    );
    expect(unverified.length).toBeGreaterThanOrEqual(2);
    for (const r of unverified) {
      expect(r.actualVerdict, r.dir).toBe("unverified");
      expect(r.expectedFailureClass, `${r.dir} pins a failure_class`).not.toBe("");
      expect(r.actualFailureClass, `${r.dir}: ${r.detail}`).toBe(r.expectedFailureClass);
    }
    // Both criteria-418 classes are exercised by the corpus.
    const classes = new Set(unverified.map((r) => r.actualFailureClass));
    expect(classes).toContain("invalid");
    expect(classes).toContain("unverifiable");
  });

  it("rejects the duplicate-member vectors at ingest; they never verify (criterion 419)", () => {
    for (const dir of ["asqav-11-dup-member-toplevel", "asqav-13-dup-member-nested"]) {
      const r = runOne(join(CORPUS_ROOT, dir), "asqav-native", "unverified", "duplicate_member", "unverifiable");
      expect(r.ok, `${dir}: ${r.detail}`).toBe(true);
      expect(r.actualVerdict).toBe("unverified");
      expect(r.actualFailureClass).toBe("unverifiable");
      expect(r.detail).toContain("terminal parse failure before any hashing");
    }
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

describe("real Authproof receipt verifies (ES256 path)", () => {
  it("authproof-01-genesis-real-sdk -> verified", () => {
    const dir = join(CORPUS_ROOT, "authproof-01-genesis-real-sdk");
    const receipt = loadJson(join(dir, "receipt.json"));
    const result = verify(receipt, ADAPTERS);
    // eslint-disable-next-line no-console
    console.log(`\n  authproof-01-genesis-real-sdk: fmt=${result.fmt} verdict=${result.verdict}\n`);
    expect(result.fmt).toBe("authproof");
    expect(result.verdict).toBe("verified");
    expect(result.failureClass).toBeNull();
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
    "w3c-vc-01-didweb-happy-path": {
      fmt: "w3c-vc",
      sha: "8bc6f6def30bde0132b272f99efdf583d49129f1c0f34291840525ed8802aed6",
    },
    "w3c-vc-08-didkey-happy-path": {
      fmt: "w3c-vc",
      sha: "969b993f84624dd9f47bba5baa2be7601a07c94bcea4369f7059d53502567c29",
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

describe("integers beyond +/-2**53 are refused at ingest (no cross-SDK divergence)", () => {
  const dec = new TextDecoder();

  // These two cases used to assert that the parser PRESERVED such an integer, so that
  // 2^53+1 and 2^53 stayed distinct. Preserving is not enough: the doors path and any
  // caller who reaches us through JSON.parse has ALREADY rounded, and a rounded 2^53 is
  // indistinguishable from a genuine one. Refusal at the parse boundary is the only
  // point where the two SDKs can still be made to agree.
  it("refuses 2^53+1, which has no exact double", () => {
    expect(() => parseJsonPreservingFloats('{"n":9007199254740993}')).toThrow(
      /canonical integer range/,
    );
  });

  it("accepts 2^53 itself, which is exactly representable and pinned upstream", () => {
    const o = parseJsonPreservingFloats('{"n":9007199254740992}');
    expect(dec.decode(asqavJcs(o))).toBe('{"n":9007199254740992}');
    expect(dec.decode(jcsRfc8785(o))).toBe('{"n":9007199254740992}');
  });

  it("accepts the conformant workaround: the same value as a JSON string", () => {
    const o = parseJsonPreservingFloats('{"n":"9007199254740993"}');
    expect(dec.decode(asqavJcs(o))).toBe('{"n":"9007199254740993"}');
  });

  it("refuses an integer at 1e21, where toString would go exponential", () => {
    expect(() => parseJsonPreservingFloats('{"n":1000000000000000000000}')).toThrow(
      /canonical integer range/,
    );
  });

  // The corpus publishes documents it says are refused. Both SDKs must actually refuse
  // them, or the corpus advertises a rule the shipped code does not implement. The Python
  // half of this pairing lives in test_corpus_integrity.py.
  it("refuses every document the corpus pins as refused", () => {
    const path = resolve(__dirname, "..", "..", "conformance", "vectors.json");
    const { vectors } = JSON.parse(readFileSync(path, "utf8")) as {
      vectors: Array<{ name: string; input_text?: string; expected_verify: boolean }>;
    };
    const refused = vectors.filter((v) => typeof v.input_text === "string");
    expect(refused.length).toBeGreaterThan(0);
    for (const v of refused) {
      expect(v.expected_verify).toBe(false);
      expect(() => parseJsonPreservingFloats(v.input_text as string), v.name).toThrow();
    }
  });

  // The boundary the corpus pins as INSIDE the range must actually parse and canonicalize.
  it("accepts every in-range vector the corpus pins, including 2**53", () => {
    const path = resolve(__dirname, "..", "..", "conformance", "vectors.json");
    const { vectors } = JSON.parse(readFileSync(path, "utf8")) as {
      vectors: Array<{ name: string; canonical?: string; input?: unknown }>;
    };
    const boundary = vectors.find((v) => v.name === "asqav-25-number-at-safe-range-boundary");
    expect(boundary, "boundary vector missing from the corpus").toBeDefined();
    const parsed = parseJsonPreservingFloats('{"n":9007199254740992}');
    expect(dec.decode(asqavJcs(parsed))).toBe(boundary!.canonical);
  });
});

// --- A29: ordered first-bad-edge reporting (criterion 490) ---

const FIRST_BAD_EDGE = resolve(__dirname, "..", "..", "verifier", "first-bad-edge-cases.json");

describe("first-bad-edge parity (criterion 490)", () => {
  it("reproduces the pinned first-bad-edge for every corpus vector", () => {
    // The same frozen table the Python gate drives: a verdict alone hides two verifiers
    // agreeing a receipt is unverified while disagreeing about WHICH check failed first.
    const table = JSON.parse(readFileSync(FIRST_BAD_EDGE, "utf-8")).cases as Record<
      string,
      string | null
    >;
    const manifest = JSON.parse(
      readFileSync(join(CORPUS_ROOT, "manifest.json"), "utf-8"),
    ) as { dir: string; format: string }[];

    expect(Object.keys(table).length).toBe(manifest.length);

    const mismatches: string[] = [];
    for (const entry of manifest) {
      const vecDir = join(CORPUS_ROOT, entry.dir);
      let got: string | null;
      try {
        const receipt = runnerLoadJson(join(vecDir, "receipt.json")) ?? {};
        const predecessor = runnerLoadJson(join(vecDir, "predecessor.json"));
        const keyProvider = keyProviderFor(vecDir, entry.format);
        got = verify(receipt, ADAPTERS, keyProvider, predecessor).firstFailingEdge;
      } catch {
        // Terminal at ingest. Both halves must agree on WHICH vectors these are,
        // so a parser-strictness divergence fails here too.
        got = "__ingest_error__";
      }
      const want = table[entry.dir];
      if (got !== want) mismatches.push(`${entry.dir}: want ${want}, got ${got}`);
    }
    expect(mismatches, `first-bad-edge divergence from Python:\n  ${mismatches.join("\n  ")}`).toEqual([]);
  });

  it("names an edge for exactly the unverified verdicts", () => {
    // Held independently on the TS side: firstFailingEdge's exclusions (expiry never folds,
    // a SKIPPED chain does not block) must track foldVerdict's, or an expired receipt names one.
    const manifest = JSON.parse(
      readFileSync(join(CORPUS_ROOT, "manifest.json"), "utf-8"),
    ) as { dir: string; format: string }[];
    let checked = 0;
    for (const entry of manifest) {
      const vecDir = join(CORPUS_ROOT, entry.dir);
      let result;
      try {
        const receipt = runnerLoadJson(join(vecDir, "receipt.json")) ?? {};
        const predecessor = runnerLoadJson(join(vecDir, "predecessor.json"));
        result = verify(receipt, ADAPTERS, keyProviderFor(vecDir, entry.format), predecessor);
      } catch {
        continue;
      }
      checked += 1;
      const namesEdge = result.firstFailingEdge !== null;
      const isUnverified = result.verdict === "unverified";
      expect(
        namesEdge,
        `${entry.dir}: verdict=${result.verdict} firstFailingEdge=${result.firstFailingEdge}`,
      ).toBe(isUnverified);
    }
    expect(checked).toBeGreaterThan(60);
  });

  it("pins the shared axis prefix order", () => {
    expect([...AXIS_ORDER_PREFIX]).toEqual(["structure", "signature", "chain", "seq"]);
  });
});
