// The three anchors wire shapes through verify(): the member absent and the member
// present as an empty array are one conformant fact; a JSON null is a malformed third
// shape and FAILs the structure axis, reported unverified/unverifiable. Mirrors
// python/tests/test_anchor_shapes.py; the corpus twins are asqav-27 and asqav-28.

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { verify, type VerifyResult } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";

const VECTORS = resolve(__dirname, "..", "..", "verifier", "conformance-vectors");

// The one sentence every engine reports for the malformed shape, pinned verbatim
// so a wording drift in either language fails here and in the Python half.
const ANCHORS_NULL_NOTE =
  "anchors is null: malformed; absent or [] is the conformant spelling of no anchors";

function vector(name: string) {
  const dir = resolve(VECTORS, name);
  const receipt = JSON.parse(readFileSync(resolve(dir, "receipt.json"), "utf-8"));
  const jwks = JSON.parse(readFileSync(resolve(dir, "jwks.json"), "utf-8"));
  const predDoc = JSON.parse(readFileSync(resolve(dir, "predecessor.json"), "utf-8"));
  const predecessor =
    predDoc && typeof predDoc === "object" && predDoc.payload ? predDoc.payload : predDoc;
  return { receipt, jwks, predecessor };
}

// The skew note embeds live wall-clock seconds; everything else is deterministic.
const maskSkew = (note: string): string => note.replace(/-?\d+s\b/g, "<n>s");

function shape(r: VerifyResult): unknown[][] {
  return r.axes.map((a) => [a.axis, a.result, maskSkew(a.note), a.failureClass]);
}

describe("anchors wire shapes", () => {
  it("absent and [] are indistinguishable in every axis result", () => {
    const a17 = vector("asqav-17-seq-contiguous");
    const a27 = vector("asqav-27-anchors-absent");
    const r17 = verify(a17.receipt, ADAPTERS, a17.jwks, a17.predecessor);
    const r27 = verify(a27.receipt, ADAPTERS, a27.jwks, a27.predecessor);
    expect(shape(r27)).toEqual(shape(r17));
    expect(r27.verdict).toBe(r17.verdict);
    expect(r27.verdict).toBe("verified");
    expect(r27.failureClass).toBe(r17.failureClass);
    expect(r27.firstFailingEdge).toBe(r17.firstFailingEdge);
  });

  it("a JSON null anchors member fails the structure axis, unverifiable", () => {
    const a28 = vector("asqav-28-anchors-null-malformed");
    const r = verify(a28.receipt, ADAPTERS, a28.jwks, a28.predecessor);
    const structure = r.axes.find((a) => a.axis === "structure")!;
    expect(structure.result).toBe("FAIL");
    expect(structure.note).toBe(ANCHORS_NULL_NOTE);
    expect(structure.failureClass).toBe("unverifiable");
    expect(r.verdict).toBe("unverified");
    expect(r.failureClass).toBe("unverifiable");
    expect(r.firstFailingEdge).toBe("structure");
    // The null never reads as "no anchors": no axis claims the skip.
    expect(r.axes.some((a) => a.note === "no anchors on this receipt")).toBe(false);
  });
});
