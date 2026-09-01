/**
 * Seq continuity axis, TS half of test_seq_continuity_axis.py. The no-SKIPPED gate matters
 * most: foldVerdict blocks on a non-chain SKIPPED, so a counter-less receipt would flip.
 */

import { readdirSync, existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { ADAPTERS, verify } from "../src/verifier/index.js";
import { PASS, FAIL, SKIPPED } from "../src/verifier/crypto.js";

const CORPUS = resolve(__dirname, "..", "..", "verifier", "conformance-vectors");

function load(vec: string, name: string): Record<string, unknown> {
  return JSON.parse(readFileSync(resolve(CORPUS, vec, name), "utf8"));
}

function withSeq(doc: Record<string, unknown>, seq: unknown): Record<string, unknown> {
  // Injection breaks the signature, so assertions read the seq axis directly.
  const out = JSON.parse(JSON.stringify(doc));
  (out.payload as Record<string, unknown>).seq = seq;
  return out;
}

function seqAxisOf(doc: Record<string, unknown>, pred: Record<string, unknown> | null = null) {
  const axis = verify(doc, ADAPTERS, null, pred).axes.find((a) => a.axis === "seq");
  expect(axis, "the seq axis must always be emitted").toBeDefined();
  return axis!;
}

describe("seq continuity axis", () => {
  it("leaves a receipt carrying no counter verified", () => {
    const doc = load("asqav-01-genesis-permit", "receipt.json");
    const res = verify(doc, ADAPTERS, load("asqav-01-genesis-permit", "jwks.json"), null);
    expect(res.verdict).toBe("verified");
    expect(res.failureClass).toBeNull();
    const axis = res.axes.find((a) => a.axis === "seq")!;
    expect(axis.result).toBe(PASS);
    expect(axis.note).toContain("not part of a counted series");
  });

  it("never reports SKIPPED across the whole corpus", () => {
    const offenders: string[] = [];
    for (const vec of readdirSync(CORPUS, { withFileTypes: true })) {
      if (!vec.isDirectory()) continue;
      const receipt = resolve(CORPUS, vec.name, "receipt.json");
      if (!existsSync(receipt)) continue;
      const predPath = resolve(CORPUS, vec.name, "predecessor.json");
      const pred = existsSync(predPath)
        ? (JSON.parse(readFileSync(predPath, "utf8")) as Record<string, unknown>)
        : null;
      const axis = verify(JSON.parse(readFileSync(receipt, "utf8")), ADAPTERS, null, pred).axes.find(
        (a) => a.axis === "seq",
      );
      if (axis && axis.result === SKIPPED) offenders.push(`${vec.name}: ${axis.note}`);
    }
    expect(offenders).toEqual([]);
  });

  it("passes a contiguous counter", () => {
    const pred = withSeq(load("asqav-03-chain-link", "predecessor.json"), 7);
    const axis = seqAxisOf(withSeq(load("asqav-03-chain-link", "receipt.json"), 8), pred);
    expect(axis.result).toBe(PASS);
    expect(axis.note).toContain("seq 8 follows predecessor 7");
  });

  it("fails a gap and counts the withheld receipts", () => {
    const pred = withSeq(load("asqav-03-chain-link", "predecessor.json"), 7);
    const doc = withSeq(load("asqav-03-chain-link", "receipt.json"), 11);
    const res = verify(doc, ADAPTERS, null, pred);
    const axis = res.axes.find((a) => a.axis === "seq")!;
    expect(axis.result).toBe(FAIL);
    expect(axis.note).toContain("3 receipt(s) withheld between 7 and 11");
    // A proven omission is a binding failure, not an incomplete recompute.
    expect(axis.failureClass).toBe("invalid");
    expect(res.verdict).toBe("unverified");
  });

  it("fails a repeated or rewound counter", () => {
    for (const seq of [7, 6, 1]) {
      const pred = withSeq(load("asqav-03-chain-link", "predecessor.json"), 7);
      const axis = seqAxisOf(withSeq(load("asqav-03-chain-link", "receipt.json"), seq), pred);
      expect(axis.result, `seq ${seq} after 7 must fail`).toBe(FAIL);
      expect(axis.note).toContain("not monotonic");
    }
  });

  it("refuses a malformed counter but accepts legal absence", () => {
    // true matters because a loose truthiness check would read it as the counter 1.
    for (const bad of [true, false, "8", 0, -1, 1.5, [8], { n: 8 }]) {
      const axis = seqAxisOf(withSeq(load("asqav-01-genesis-permit", "receipt.json"), bad));
      expect(axis.result, `${JSON.stringify(bad)} must not pass as a counter`).toBe(FAIL);
      expect(axis.note).toContain("malformed seq");
    }
    expect(seqAxisOf(withSeq(load("asqav-01-genesis-permit", "receipt.json"), null)).result).toBe(PASS);
  });

  it("passes with a note when no predecessor is supplied", () => {
    const axis = seqAxisOf(withSeq(load("asqav-01-genesis-permit", "receipt.json"), 4));
    expect(axis.result).toBe(PASS);
    expect(axis.note).toContain("no predecessor supplied");
  });

  it("passes with a note when the predecessor carries no counter", () => {
    const pred = load("asqav-03-chain-link", "predecessor.json");
    expect((pred.payload as Record<string, unknown>).seq).toBeUndefined();
    const axis = seqAxisOf(withSeq(load("asqav-03-chain-link", "receipt.json"), 2), pred);
    expect(axis.result).toBe(PASS);
    expect(axis.note).toContain("predecessor carries no seq");
  });

  it("fails a malformed predecessor counter", () => {
    const pred = withSeq(load("asqav-03-chain-link", "predecessor.json"), "seven");
    const axis = seqAxisOf(withSeq(load("asqav-03-chain-link", "receipt.json"), 8), pred);
    expect(axis.result).toBe(FAIL);
    expect(axis.note).toContain("malformed predecessor seq");
  });

  it("never compares counters across formats", () => {
    // A foreign receipt's counter is not this series; comparing would fake a gap.
    const pred = load("acta-01-genesis", "receipt.json");
    pred.seq = 99;
    const axis = seqAxisOf(withSeq(load("asqav-01-genesis-permit", "receipt.json"), 2), pred);
    expect(axis.result).toBe(PASS);
    expect(axis.note).toContain("different receipt format");
  });

  it("refuses a counter pasted onto a hash-mode receipt", () => {
    // Hash mode signs flat fields only, so a pasted seq is an unsigned claim.
    const doc = load("asqav-05-hash-mode-prod", "receipt.json");
    const jwks = load("asqav-05-hash-mode-prod", "jwks.json");
    const clean = verify(doc, ADAPTERS, jwks, null);
    expect(clean.axes.find((a) => a.axis === "structure")!.result).toBe(PASS);
    expect(clean.axes.find((a) => a.axis === "seq")!.result).toBe(PASS);

    const forged = JSON.parse(JSON.stringify(doc));
    forged.seq = 5;
    const res = verify(forged, ADAPTERS, jwks, null);
    const structure = res.axes.find((a) => a.axis === "structure")!;
    expect(structure.result).toBe(FAIL);
    expect(structure.note).toContain("signature does not cover");
    const axis = res.axes.find((a) => a.axis === "seq")!;
    expect(axis.result).toBe(PASS);
    expect(axis.note).toContain("not part of a counted series");
    expect(res.verdict).toBe("unverified");
  });
});
