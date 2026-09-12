/**
 * THE GATE: the TS verifier must reproduce the Python oracle's verdicts across the corpus,
 * byte-match the upstream canonicalization vectors, and verify the real Authproof receipt.
 */

import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";

import {
  ProfileIntegerError,
  asqavJcs,
  jcsRfc8785,
  parseJsonPreservingFloats,
  parseProfileJson,
} from "../src/verifier/canonical.js";
import { AXIS_ORDER_PREFIX, verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { AsqavNativeAdapter } from "../src/verifier/adapters/asqavNative.js";
import { AerfAdapter } from "../src/verifier/adapters/aerf.js";
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
  it("matches every manifest outcome across all 81 corpus vectors", () => {
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

    expect(results.length).toBe(81);
    expect(mismatches, `mismatched vectors: ${mismatches.map((m) => m.dir).join(", ")}`).toEqual([]);
    expect(passed).toBe(81);
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
      sha: "5614b581dabab8e71114137531342161ac0512cf326fe002130aa69906e078da",
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

  it("keeps 2^53 accepted on the shared generic path (upstream compatibility)", () => {
    const o = parseJsonPreservingFloats('{"n":9007199254740992}');
    expect(dec.decode(asqavJcs(o))).toBe('{"n":9007199254740992}');
    expect(dec.decode(jcsRfc8785(o))).toBe('{"n":9007199254740992}');
  });

  it("refuses 2^53 on the explicit Asqav profile entry", () => {
    expect(() => parseProfileJson('{"n":9007199254740992}')).toThrow(ProfileIntegerError);
    expect(() => parseProfileJson('{"n":-9007199254740992}')).toThrow(
      /Asqav profile range/,
    );
    expect(parseProfileJson('{"n":9007199254740991}')).toEqual({ n: 9007199254740991 });
  });

  it("refuses float spellings of the excluded boundary on the profile entry", () => {
    expect(() => parseProfileJson('{"n":9007199254740992.0}')).toThrow(ProfileIntegerError);
    expect(() => parseProfileJson('{"n":9.007199254740992e15}')).toThrow(
      ProfileIntegerError,
    );
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

  // Refused corpus documents must fail via the explicit profile entry (Python
  // half: test_corpus_integrity.py), or corpus advertises an unimplemented rule
  it("refuses every document the corpus pins as refused", () => {
    const path = resolve(__dirname, "..", "..", "conformance", "vectors.json");
    const { vectors } = JSON.parse(readFileSync(path, "utf8")) as {
      vectors: Array<{ name: string; input_text?: string; expected_verify: boolean }>;
    };
    const refused = vectors.filter((v) => typeof v.input_text === "string");
    expect(refused.length).toBeGreaterThan(0);
    for (const v of refused) {
      expect(v.expected_verify).toBe(false);
      expect(() => parseProfileJson(v.input_text as string), v.name).toThrow();
    }
  });

  // The boundary the corpus pins as INSIDE the profile range must parse there
  it("accepts the in-range boundary vector the corpus pins, at 2**53 - 1", () => {
    const path = resolve(__dirname, "..", "..", "conformance", "vectors.json");
    const { vectors } = JSON.parse(readFileSync(path, "utf8")) as {
      vectors: Array<{ name: string; canonical?: string; input?: unknown }>;
    };
    const boundary = vectors.find((v) => v.name === "asqav-25-number-at-safe-range-boundary");
    expect(boundary, "boundary vector missing from the corpus").toBeDefined();
    expect(boundary!.input).toEqual({ n: 9007199254740991 });
    const parsed = parseProfileJson('{"n":9007199254740991}');
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
        got = verify(receipt, ADAPTERS, keyProvider, predecessor, { originatingEnvelope: runnerLoadJson(join(vecDir, "originating_envelope.json")) }).firstFailingEdge;
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
        result = verify(receipt, ADAPTERS, keyProviderFor(vecDir, entry.format), predecessor, { originatingEnvelope: runnerLoadJson(join(vecDir, "originating_envelope.json")) });
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

describe("Asqav profile safe-integer precheck (current v=1 refuses before crypto)", () => {
  const load = (vec: string, name = "receipt.json"): Record<string, unknown> =>
    loadJson(join(CORPUS_ROOT, vec, name));
  const providerFor = (vec: string) => keyProviderFor(join(CORPUS_ROOT, vec), "asqav-native");
  const bomb = (..._args: never[]): never => {
    throw new Error("crypto callback must not run after a profile refusal");
  };
  function spiedAdapter(): AsqavNativeAdapter {
    const ad = new AsqavNativeAdapter();
    ad.signingInput = bomb;
    ad.resolveKey = bomb;
    ad.schema = bomb;
    ad.chainStep = bomb;
    return ad;
  }
  const notes = (res: { axes: Array<{ note: string }> }) => res.axes.map((a) => a.note);

  it("refuses a nested excluded number before any crypto callback", () => {
    const ad = spiedAdapter();
    const doc = load("asqav-01-genesis-permit");
    (doc.payload as Record<string, unknown>).score = 2 ** 53;
    const res = verify(doc, [ad], providerFor("asqav-01-genesis-permit"));
    expect(res.fmt).toBe("asqav-native");
    expect(res.verdict).toBe("unverified");
    expect(res.failureClass).toBe("unverifiable");
    expect(res.axes).toHaveLength(1);
    expect(res.axes[0].axis).toBe("structure");
    expect(res.axes[0].note).toContain("profile range +/-(2**53 - 1)");
    expect(res.firstFailingEdge).toBe("structure");
  });

  it("still refuses when the nested payload carries inner hash mode", () => {
    const doc = load("asqav-01-genesis-permit");
    const payload = doc.payload as Record<string, unknown>;
    payload.mode = "hash";
    payload.score = -(2 ** 53);
    const res = verify(doc, ADAPTERS);
    expect(res.verdict).toBe("unverified");
    expect(res.axes[0].note).toContain("profile range +/-(2**53 - 1)");
  });

  it("refuses a bare current payload", () => {
    const res = verify(
      { previousReceiptHash: "1".repeat(64), issuer_id: "kid-x", v: 1, score: 2 ** 53 },
      ADAPTERS,
    );
    expect(res.fmt).toBe("asqav-native");
    expect(res.axes[0].note).toContain("profile range +/-(2**53 - 1)");
  });

  it("refuses flat excluded metadata before any crypto callback", () => {
    const ad = spiedAdapter();
    const doc = load("asqav-05-hash-mode-prod");
    doc.metadata = { batch: 2 ** 53 };
    const res = verify(doc, [ad], providerFor("asqav-05-hash-mode-prod"));
    expect(res.verdict).toBe("unverified");
    expect(res.axes[0].note).toContain("profile range +/-(2**53 - 1)");
  });

  it("flat missing-v keeps the schema outcome, not a profile refusal", () => {
    const doc = load("asqav-05-hash-mode-prod");
    delete doc.v;
    doc.metadata = { batch: 2 ** 53 };
    const res = verify(doc, ADAPTERS);
    expect(notes(res).some((n) => n.includes("profile range"))).toBe(false);
    expect(res.axes.find((a) => a.axis === "structure")!.result).toBe("FAIL");
  });

  it("a clean v=1 receipt reaches the signing-input callback", () => {
    const ad = new AsqavNativeAdapter();
    const calls: unknown[][] = [];
    const orig = AsqavNativeAdapter.prototype.signingInput;
    ad.signingInput = (doc) => {
      calls.push([doc]);
      return orig.call(ad, doc);
    };
    verify(load("asqav-01-genesis-permit"), [ad], providerFor("asqav-01-genesis-permit"));
    expect(calls.length).toBeGreaterThan(0);
  });

  it.each([[undefined], [2], [true], ["1"]])(
    "non-current version %p skips the precheck",
    (version) => {
      const doc = load("asqav-01-genesis-permit");
      const payload = doc.payload as Record<string, unknown>;
      if (version === undefined) delete payload.v;
      else payload.v = version as number;
      payload.score = 2 ** 53;
      const res = verify(doc, ADAPTERS);
      expect(notes(res).some((n) => n.includes("profile range"))).toBe(false);
    },
  );

  it("outer v cannot activate a nested payload missing v", () => {
    const doc = load("asqav-01-genesis-permit");
    const payload = doc.payload as Record<string, unknown>;
    delete payload.v;
    payload.score = 2 ** 53;
    doc.v = 1;
    const res = verify(doc, ADAPTERS);
    expect(notes(res).some((n) => n.includes("profile range"))).toBe(false);
  });

  it("a bad predecessor refuses even without its own v", () => {
    const doc = load("asqav-03-chain-link");
    const pred = load("asqav-03-chain-link", "predecessor.json");
    const payload = pred.payload as Record<string, unknown>;
    payload.score = 2 ** 53;
    delete payload.v;
    const res = verify(doc, ADAPTERS, null, pred);
    expect(res.axes[0].note).toContain("profile range +/-(2**53 - 1)");
  });

  it("genesis skips predecessor numbers", () => {
    const doc = load("asqav-01-genesis-permit");
    const pred = load("asqav-03-chain-link", "predecessor.json");
    (pred.payload as Record<string, unknown>).score = 2 ** 53;
    const res = verify(doc, ADAPTERS, null, pred);
    expect(notes(res).some((n) => n.includes("profile range"))).toBe(false);
  });

  it("a foreign predecessor keeps the chain result", () => {
    const doc = load("asqav-03-chain-link");
    const pred = loadJson(join(CORPUS_ROOT, "aerf-01-genesis", "receipt.json"));
    const res = verify(doc, ADAPTERS, null, pred);
    expect(notes(res).some((n) => n.includes("profile range"))).toBe(false);
    expect(res.axes.find((a) => a.axis === "chain")!.result).toBe("FAIL");
    expect(res.failureClass).toBe("invalid");
  });

  it("unsigned top-level metadata and JWKS numbers are not checked", () => {
    const doc = load("asqav-01-genesis-permit");
    doc.export_seq = 2 ** 53;
    const provider = providerFor("asqav-01-genesis-permit") as Record<string, unknown>;
    provider.max_seen = 2 ** 53;
    const res = verify(doc, ADAPTERS, provider);
    expect(notes(res).some((n) => n.includes("profile range"))).toBe(false);
  });

  it("foreign formats have no profile precheck", () => {
    const doc = loadJson(join(CORPUS_ROOT, "aerf-01-genesis", "receipt.json"));
    expect(new AerfAdapter().profilePrecheck(doc)).toBeNull();
  });
});

describe("profile selection reads own members only (no prototype activation)", () => {
  const td = new TextDecoder();
  const load = (vec: string, name = "receipt.json"): Record<string, unknown> =>
    loadJson(join(CORPUS_ROOT, vec, name));
  const notes = (res: { axes: Array<{ note: string }> }) => res.axes.map((a) => a.note);

  it("an inherited v=1 cannot select the neutral precheck", () => {
    const versionText = `{"payload":{"__proto__":{"v":1},"previousReceiptHash":"${"0".repeat(64)}","issuer_id":"review","n":9007199254740992},"signature":{"sig":"AAAA"}}`;
    const parsed = parseJsonPreservingFloats(versionText) as Record<string, unknown>;
    const payload = parsed.payload as Record<string, unknown>;
    expect(Object.prototype.hasOwnProperty.call(parsed, "v")).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(payload, "v")).toBe(false);
    expect(() => parseProfileJson(versionText)).toThrow(ProfileIntegerError);
    const res = verify(parsed, ADAPTERS);
    expect(notes(res).some((n) => n.includes("profile range"))).toBe(false);
  });

  it("a direct nested object with prototype v stays unselected", () => {
    const payload = Object.assign(Object.create({ v: 1 }), {
      previousReceiptHash: "1".repeat(64),
      issuer_id: "kid-x",
      score: 2 ** 53,
    });
    const res = verify({ payload, signature: { sig: "AAAA" } }, ADAPTERS);
    expect(notes(res).some((n) => n.includes("profile range"))).toBe(false);
  });

  it("a direct flat object with prototype v stays unselected", () => {
    const doc = Object.assign(Object.create({ v: 1 }), {
      mode: "hash",
      metadata: { batch: 2 ** 53 },
    });
    const res = verify(doc, ADAPTERS);
    expect(notes(res).some((n) => n.includes("profile range"))).toBe(false);
  });

  it("an own v=1 still selects after the guard", () => {
    const doc = load("asqav-01-genesis-permit");
    (doc.payload as Record<string, unknown>).score = 2 ** 53;
    const res = verify(doc, ADAPTERS);
    expect(res.axes[0].note).toContain("profile range +/-(2**53 - 1)");
  });

  it("the upstream 2^53 vector stays generic while the profile refuses it", () => {
    const vec = loadJson(join(CORPUS_ROOT, "agentreceipts-upstream-interop", "canonicalization_vectors.json")) as {
      canonicalization_vectors: Array<{ name: string; input: unknown; canonical: string }>;
    };
    const pinned = vec.canonicalization_vectors.find((v) => v.name === "number_2_to_53")!;
    expect(td.decode(jcsRfc8785(pinned.input))).toBe(pinned.canonical);
    expect(() => parseProfileJson(JSON.stringify(pinned.input))).toThrow(ProfileIntegerError);
  });

  it("a dual-detect predecessor follows the registry, foreign first", () => {
    const doc = load("asqav-03-chain-link");
    for (const value of [1, 2 ** 53]) {
      const pred = {
        type: "notarised_evidence",
        evidence_hash_sha512: "0".repeat(128),
        previousReceiptHash: "1".repeat(64),
        issuer_id: "review",
        n: value,
      };
      expect(new AsqavNativeAdapter().detect(pred)).toBe(true);
      const res = verify(doc, [new AerfAdapter(), new AsqavNativeAdapter()], null, pred);
      expect(notes(res).some((n) => n.includes("profile range"))).toBe(false);
      expect(res.axes.find((a) => a.axis === "chain")!.note).toBe(
        "predecessor is a different receipt format",
      );
      expect(res.failureClass).toBe("invalid");
    }
  });

  it("a dual-detect predecessor follows the registry, native first", () => {
    const doc = load("asqav-03-chain-link");
    const pred = {
      type: "notarised_evidence",
      evidence_hash_sha512: "0".repeat(128),
      previousReceiptHash: "1".repeat(64),
      issuer_id: "review",
      n: 2 ** 53,
    };
    const res = verify(doc, [new AsqavNativeAdapter(), new AerfAdapter()], null, pred);
    expect(res.axes[0].note).toContain("profile range +/-(2**53 - 1)");
    expect(res.failureClass).toBe("unverifiable");
  });
});
