// Anchor non-coverage rows stay declared until a TS port exists
// Deleting either row must turn the suite RED

import { describe, expect, it } from "vitest";

import { NOT_CHECKED } from "../src/verifier/not-checked.js";

function row(check: string): Record<string, unknown> {
  const found = NOT_CHECKED.find((e) => e.check === check);
  expect(found, `${check} missing from NOT_CHECKED`).toBeDefined();
  return { ...found } as Record<string, unknown>;
}

describe("anchor non-coverage declaration", () => {
  it("declares the merkle path gap unconditionally", () => {
    const entry = row("opentimestamps_merkle_path");
    expect(entry["condition"]).toBeNull();
  });

  it("declares the block placement gap unconditionally", () => {
    const entry = row("opentimestamps_block_placement");
    expect(entry["condition"]).toBeNull();
  });
});
