/**
 * Anchor-binding, clock-skew, signed-expiry, and nesting-depth parity, TypeScript half.
 *
 * One JSON table drives both verifiers, pinned to the output of the Python
 * `check_anchors` / `check_skew` / `check_expiry`. The Python half lives in
 * python/tests/test_axis_parity_cases.py and reads the same file.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { verifyReceiptOffline } from "../src/index.js";
// Imported from the published `@asqav/sdk/verifier` entry, not the inner module, so
// a caller following the docs reaches every piece the Python surface exposes.
import {
  checkAnchors,
  checkExpiry,
  checkSkew,
  envelopeMinusAnchorsJcs,
  MAX_NESTING_DEPTH,
  normaliseEnvelope,
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
};

const PIPELOCK_VECTOR = resolve(
  __dirname, "..", "..", "verifier", "conformance-vectors", "pipelock-ev2-01-proxy-decision",
);

function loadVector(file: string): Record<string, unknown> {
  return JSON.parse(readFileSync(resolve(PIPELOCK_VECTOR, file), "utf-8")) as Record<string, unknown>;
}

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

// Cases where the two halves legitimately disagree, because this shim carries no
// anchor cryptography (criterion 446). The Python side of the same rows is
// asserted in python/tests/test_axis_parity_cases.py.
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
      // The divergence itself: Python reaches a verdict this shim cannot.
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
    // stuck one, so both directions have to be present.
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
    // payload must not move a lapsed receipt's window.
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
    // PASS the chain axis on a receipt Python leaves unchecked.
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
      // The digest is the bytes the anchors axis binds, so an exact match proves
      // both halves normalise and canonicalise the same envelope.
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
  // in-repo imports still resolve while an installed package sees nothing.
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

  // The envelope key and the leaf each add a level, so 198 wrappers sit exactly at
  // the cap and 199 cross it. Python answers the same way for the same two docs.
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

  it("returns a verdict instead of exhausting the stack", () => {
    // Nesting inside the signed payload reaches the recursive JCS encoder, which
    // throws RangeError out of the call without the gate.
    let node: Record<string, unknown> = { leaf: 1 };
    for (let i = 0; i < 20000; i++) node = { n: node };
    const receipt = nestedReceipt(0);
    (receipt.payload as Record<string, unknown>).junk = node;
    expect(() => verifyReceiptOffline(receipt, JWKS)).not.toThrow();
    expect(verifyReceiptOffline(receipt, JWKS).verdict).toBe("unverified");
  });
});
