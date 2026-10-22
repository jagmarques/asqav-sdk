// Anchor, skew, expiry and nesting-depth parity, TS half. One JSON table drives
// both verifiers; python/tests/test_axis_parity_cases.py reads the same file

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { verifyReceiptOffline } from "../src/index.js";
import { checkCounterpartyBinding } from "../src/verifier/vrShim.js";
// Imported from the published `@asqav/sdk/verifier` entry, not the inner
// module, so a documented caller reaches every piece the Python surface has
import {
  checkAnchors,
  checkExpiry,
  checkSkew,
  counterpartyEnvelopeHash,
  envelopeMinusAnchorsJcs,
  MAX_NESTING_DEPTH,
  normaliseEnvelope,
  parseJsonPreservingFloats,
  RawFloat,
  sha256Hex,
  SKEW_BOUND_SECONDS,
} from "../src/verifier/index.js";

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

interface NormaliseCase {
  name: string;
  raw: Record<string, unknown>;
  expect: { digest: string; anchors_axis: string };
}

interface ExpiryCase {
  name: string;
  payload: Record<string, unknown>;
  expect: { result: string; note_contains: string };
}

interface ChainPrevCase {
  name: string;
  value?: unknown;
  omit?: boolean;
  expect: { verdict: string; chain: string; signature: string };
}

interface NestingDepthCase {
  name: string;
  wrappers: number;
  leaf: string;
  expect: { structure: string; verdict?: string; failure_class?: string; note?: string };
}

interface CounterpartyDepthCase {
  name: string;
  wrappers: number;
  leaf: string;
  envelope_hash: string;
  expect: { result: string; note_contains: string };
}

interface NestingDepthTable {
  envelope: Record<string, unknown>;
  jwks: Record<string, unknown>;
  cases: NestingDepthCase[];
  counterparty: {
    envelope: Record<string, unknown>;
    payload: Record<string, unknown>;
    cases: CounterpartyDepthCase[];
  };
}

interface DivergenceCase {
  name: string;
  envelope: Record<string, unknown>;
  expect_python: { result: string; note_contains: string };
  expect_ts: { result: string; note_contains: string };
}

const CASES_FILE = resolve(__dirname, "..", "..", "verifier", "axis-parity-cases.json");
const TABLE = JSON.parse(readFileSync(CASES_FILE, "utf-8")) as {
  anchors: AnchorCase[];
  skew: SkewCase[];
  normalise: NormaliseCase[];
  expiry: ExpiryCase[];
  chain_prev_hash: ChainPrevCase[];
  anchors_divergence: DivergenceCase[];
  nesting_depth: NestingDepthTable;
};

const PIPELOCK_VECTOR = resolve(
  __dirname, "..", "..", "verifier", "conformance-vectors", "pipelock-ev2-01-proxy-decision",
);

function loadVector(file: string): Record<string, unknown> {
  return JSON.parse(readFileSync(resolve(PIPELOCK_VECTOR, file), "utf-8")) as Record<string, unknown>;
}

/** A receipt whose only defect is `depth` nesting levels outside the payload */
function nestedReceipt(depth: number): Record<string, unknown> {
  let node: Record<string, unknown> = { leaf: 1 };
  for (let i = 0; i < depth; i++) node = { n: node };
  return {
    payload: {
      type: "protectmcp:decision",
      v: 1,
      mode: "payload",
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
      // exact match also proves the two canonicalisers agree byte for byte
      expect(note, c.name).toBe(c.expect.note);
    });
  }
});

// Rows where the halves disagree: this shim has no anchor cryptography
// (criterion 446); the Python side is asserted in its parity test
describe("anchor divergence from the Python verifier", () => {
  it("the divergence table is populated and two-sided", () => {
    expect(TABLE.anchors_divergence.length).toBeGreaterThanOrEqual(2);
    const py = new Set(TABLE.anchors_divergence.map((c) => c.expect_python.result));
    expect([...py].sort()).toEqual(["FAIL", "PASS"]);
  });

  for (const c of TABLE.anchors_divergence) {
    it(c.name, () => {
      const [result, note] = checkAnchors(c.envelope);
      expect(result, c.name).toBe(c.expect_ts.result);
      expect(note, c.name).toContain(c.expect_ts.note_contains);
      // The divergence itself: Python reaches a verdict this shim cannot
      expect(result, c.name).not.toBe(c.expect_python.result);
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

describe("signed-expiry axis parity", () => {
  it("is populated in both directions", () => {
    expect(TABLE.expiry.length).toBeGreaterThanOrEqual(15);
    // A table of only-PASS or only-FAIL cases cannot tell a working axis from a
    // stuck one, so both directions have to be present
    expect(new Set(TABLE.expiry.map((c) => c.expect.result))).toEqual(new Set(["PASS", "FAIL"]));
  });

  for (const c of TABLE.expiry) {
    it(c.name, () => {
      const [result, note] = checkExpiry(c.payload);
      expect(result, `${c.name}: note ${note}`).toBe(c.expect.result);
      expect(note, c.name).toContain(c.expect.note_contains);
    });
  }

  it("never PASSes a stamp the Python table refuses", () => {
    const permissive = TABLE.expiry
      .filter((c) => c.expect.result === "FAIL" && checkExpiry(c.payload)[0] === "PASS")
      .map((c) => c.name);
    expect(permissive, `TS accepts these where Python refuses: ${permissive.join(", ")}`).toEqual([]);
  });

  it("reads only the signed bytes", () => {
    // anchors and the envelope keys are unsigned, so an expires_at beside the
    // payload must not move a lapsed receipt's window
    const signed = { expires_at: "2020-01-01T00:00:00Z" };
    expect(checkExpiry(signed)[0]).toBe("FAIL");
    const raw = { payload: { ...signed }, signature: "AAAA", expires_at: "2099-01-01T00:00:00Z" };
    const env = normaliseEnvelope(raw);
    expect(checkExpiry(env.payload)[0]).toBe("FAIL");
    expect(checkExpiry(env)[0]).toBe("PASS");
  });
});

describe("pipelock chain_prev_hash parity", () => {
  it("carries both a genesis PASS and a non-genesis SKIPPED", () => {
    const chains = new Set(TABLE.chain_prev_hash.map((c) => c.expect.chain));
    expect(chains).toEqual(new Set(["PASS", "SKIPPED"]));
  });

  for (const c of TABLE.chain_prev_hash) {
    it(c.name, () => {
      const doc = loadVector("receipt.json");
      if (c.omit) delete doc.chain_prev_hash;
      else if ("value" in c) doc.chain_prev_hash = c.value;
      const result = verifyReceiptOffline(doc, loadVector("keys.json"));
      const axes = Object.fromEntries(result.axes.map((a) => [a.axis, a.result]));
      expect(result.verdict, `${c.name}: ${JSON.stringify(axes)}`).toBe(c.expect.verdict);
      expect(axes.chain, c.name).toBe(c.expect.chain);
      expect(axes.signature, c.name).toBe(c.expect.signature);
    });
  }

  it("never reads a producer-set value as an absent link", () => {
    // A non-string chain_prev_hash narrowed to null would read as genesis and
    // PASS the chain axis on a receipt Python leaves unchecked
    const laundered = TABLE.chain_prev_hash
      .filter((c) => c.expect.chain === "SKIPPED")
      .filter((c) => {
        const doc = loadVector("receipt.json");
        doc.chain_prev_hash = c.value;
        const r = verifyReceiptOffline(doc, loadVector("keys.json"));
        return r.axes.find((a) => a.axis === "chain")?.result === "PASS";
      })
      .map((c) => c.name);
    expect(laundered, `TS reads these as genesis where Python does not: ${laundered.join(", ")}`).toEqual([]);
  });
});

describe("envelope normalisation parity", () => {
  it("is populated", () => {
    expect(TABLE.normalise.length).toBeGreaterThanOrEqual(5);
  });

  for (const c of TABLE.normalise) {
    it(c.name, () => {
      const env = normaliseEnvelope(c.raw);
      // The digest is what the anchors axis binds: a match proves both halves
      // normalise and canonicalise the same envelope
      expect(sha256Hex(envelopeMinusAnchorsJcs(env)), c.name).toBe(c.expect.digest);
      expect(checkAnchors(env)[0], c.name).toBe(c.expect.anchors_axis);
    });
  }

  it("never launders a malformed anchors value into an empty list", () => {
    for (const anchors of [{}, "rfc3161", 0]) {
      const raw = { payload: { type: "protectmcp:decision" }, signature: "AAAA", anchors };
      expect(checkAnchors(normaliseEnvelope(raw))[0], JSON.stringify(anchors)).toBe("FAIL");
    }
  });
});

describe("the verifier entry point publishes every axis helper", () => {
  // Callers reach these through `@asqav/sdk/verifier`. Drop the subpath and the
  // in-repo imports still resolve while an installed package sees nothing
  it("keeps the ./verifier subpath in the exports map", () => {
    const pkg = JSON.parse(
      readFileSync(resolve(__dirname, "..", "package.json"), "utf-8"),
    ) as { exports: Record<string, Record<string, string>> };
    const sub = pkg.exports["./verifier"];
    expect(sub, "exports map dropped ./verifier").toBeDefined();
    expect(sub.types).toBe("./dist/verifier/index.d.ts");
    expect(sub.import).toBe("./dist/verifier/index.mjs");
    expect(sub.require).toBe("./dist/verifier/index.js");
  });

  it("re-exports the helpers the Python standalone surface runs", async () => {
    const entry = (await import("../src/verifier/index.js")) as Record<string, unknown>;
    for (const name of ["normaliseEnvelope", "checkAnchors", "checkSkew", "checkExpiry"]) {
      expect(typeof entry[name], `${name} missing from the verifier entry`).toBe("function");
    }
    expect(entry.SKEW_BOUND_SECONDS).toBe(300);
  });
});

describe("nesting-depth gate parity", () => {
  it("keeps the Python cap", () => {
    expect(MAX_NESTING_DEPTH).toBe(200);
  });

  // The envelope key and the leaf each add a level, so 198 wrappers sit at
  // the cap and 199 cross it. Python answers the same for the same two docs
  it("passes a receipt at the cap", () => {
    const r = verifyReceiptOffline(nestedReceipt(MAX_NESTING_DEPTH - 2), JWKS);
    expect(r.axes.find((a) => a.axis === "structure")?.result).toBe("PASS");
  });

  it("caps a receipt one level past it", () => {
    const r = verifyReceiptOffline(nestedReceipt(MAX_NESTING_DEPTH - 1), JWKS);
    expect(r.verdict).toBe("unverified");
    expect(r.failureClass).toBe("unverifiable");
    expect(r.axes).toHaveLength(1);
    expect(r.axes[0].axis).toBe("structure");
    expect(r.axes[0].result).toBe("FAIL");
    expect(r.axes[0].note).toBe("receipt nesting exceeds the supported depth (> 200 levels)");
  });

  it("caps an over-nested predecessor", () => {
    const r = verifyReceiptOffline(nestedReceipt(0), JWKS, nestedReceipt(5000));
    expect(r.verdict).toBe("unverified");
    expect(r.failureClass).toBe("unverifiable");
    expect(r.axes[0].note).toBe("receipt nesting exceeds the supported depth (> 200 levels)");
  });

  it("the depth table is populated and two-sided", () => {
    const outcomes = new Set(TABLE.nesting_depth.cases.map((c) => c.expect.structure));
    expect([...outcomes].sort()).toEqual(["FAIL", "PASS"]);
    const leaves = new Set(TABLE.nesting_depth.cases.map((c) => /[.eE]/.test(c.leaf)));
    expect(leaves.size, "needs both decimal and integer leaves").toBe(2);
  });

  // The parser wraps each decimal in a RawFloat, a plain float in Python, so
  // the wrapper must not add a level to the depth walk
  const PARSERS: Array<[string, (text: string) => unknown]> = [
    ["parseJsonPreservingFloats", parseJsonPreservingFloats],
    ["JSON.parse", (text) => JSON.parse(text)],
  ];
  const nest = (wrappers: number, leaf: string): string =>
    '{"n":'.repeat(wrappers) + `{"leaf":${leaf}}` + "}".repeat(wrappers);

  for (const [parserName, parse] of PARSERS) {
    for (const c of TABLE.nesting_depth.cases) {
      it(`shared case (${parserName}): ${c.name}`, () => {
        const base = JSON.stringify(TABLE.nesting_depth.envelope).slice(0, -1);
        const receipt = parse(`${base},"junk":${nest(c.wrappers, c.leaf)}}`) as Record<string, unknown>;
        const r = verifyReceiptOffline(receipt, TABLE.nesting_depth.jwks);
        expect(r.axes.find((a) => a.axis === "structure")?.result, c.name).toBe(c.expect.structure);
        if (c.expect.structure === "FAIL") {
          expect(r.verdict).toBe(c.expect.verdict);
          expect(r.failureClass).toBe(c.expect.failure_class);
          expect(r.axes).toHaveLength(1);
          expect(r.axes[0].note).toBe(c.expect.note);
        }
      });
    }
  }

  // The counterparty hash walks the originating envelope under its own cap
  for (const [parserName, parse] of PARSERS) {
    for (const c of TABLE.nesting_depth.counterparty.cases) {
      it(`counterparty case (${parserName}): ${c.name}`, () => {
        const env = TABLE.nesting_depth.counterparty.envelope as { payload: object; signature: object };
        const payloadText = JSON.stringify(env.payload).slice(0, -1) + `,"junk":${nest(c.wrappers, c.leaf)}}`;
        const originator = parse(`{"payload":${payloadText},"signature":${JSON.stringify(env.signature)}}`);
        const payload = JSON.parse(JSON.stringify(TABLE.nesting_depth.counterparty.payload)) as {
          counterparty_binding: Record<string, unknown>;
        };
        payload.counterparty_binding.envelope_hash = c.envelope_hash;
        const [result, note] = checkCounterpartyBinding(payload, originator);
        expect(result, `${c.name}: ${note}`).toBe(c.expect.result);
        expect(note, c.name).toContain(c.expect.note_contains);
      });
    }
  }

  // Look-alikes are containers, so they must still meet the cap
  const deep = (levels: number): Record<string, unknown> => {
    let node: Record<string, unknown> = { leaf: 1 };
    for (let i = 0; i < levels; i++) node = { n: node };
    return node;
  };
  const forgeries: Array<[string, () => unknown]> = [
    ["prototype swapped onto an object with an extra key", () =>
      Object.setPrototypeOf({ value: 1, n: deep(500) }, RawFloat.prototype)],
    ["prototype swapped onto a bare nested object", () =>
      Object.setPrototypeOf(deep(500), RawFloat.prototype)],
    ["a nested object passed as the value", () => new RawFloat(deep(500) as unknown as number)],
    ["a nested object passed as the lexeme", () =>
      new RawFloat(1.5, deep(500) as unknown as string)],
  ];
  for (const [name, make] of forgeries) {
    it(`caps a forged RawFloat: ${name}`, () => {
      const receipt = nestedReceipt(0);
      receipt.junk = make();
      const r = verifyReceiptOffline(receipt, JWKS);
      expect(r.verdict).toBe("unverified");
      expect(r.failureClass).toBe("unverifiable");
      expect(r.axes[0].note).toBe("receipt nesting exceeds the supported depth (> 200 levels)");
    });
    it(`caps a forged RawFloat in the originator: ${name}`, () => {
      expect(() => counterpartyEnvelopeHash({
        payload: { junk: make() },
        signature: { alg: "a", kid: "k", sig: "s" },
      })).toThrow(/nesting exceeds 200/);
    });
  }

  it("keeps a genuine RawFloat a leaf at the cap", () => {
    for (const leaf of [new RawFloat(1.5), new RawFloat(1.5, "1.5")]) {
      const receipt = nestedReceipt(0);
      let node: Record<string, unknown> = { leaf };
      for (let i = 0; i < MAX_NESTING_DEPTH - 2; i++) node = { n: node };
      receipt.junk = node;
      const r = verifyReceiptOffline(receipt, JWKS);
      expect(r.axes.find((a) => a.axis === "structure")?.result).toBe("PASS");
    }
  });

  it("parses a decimal leaf to a RawFloat, so the shared cases are not vacuous", () => {
    const parsed = parseJsonPreservingFloats('{"leaf":1.5}') as Record<string, unknown>;
    expect(parsed.leaf).toBeInstanceOf(RawFloat);
  });

  it("returns a verdict instead of exhausting the stack", () => {
    // Nesting inside the signed payload reaches the recursive JCS encoder,
    // which throws RangeError out of the call without the gate
    let node: Record<string, unknown> = { leaf: 1 };
    for (let i = 0; i < 20000; i++) node = { n: node };
    const receipt = nestedReceipt(0);
    (receipt.payload as Record<string, unknown>).junk = node;
    expect(() => verifyReceiptOffline(receipt, JWKS)).not.toThrow();
    expect(verifyReceiptOffline(receipt, JWKS).verdict).toBe("unverified");
  });
});
