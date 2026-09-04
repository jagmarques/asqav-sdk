/**
 * The ACTA chain recompute follows the form of the carried link (ACTA -03 §6.7).
 * Mirrors python/tests/test_acta_chain_form.py.
 */

import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { ActaAdapter } from "../src/verifier/adapters/acta.js";
import { verify } from "../src/verifier/core.js";

const CORPUS = resolve(__dirname, "..", "..", "verifier", "conformance-vectors");

function acta06(): [Record<string, unknown>, Record<string, unknown>, Record<string, unknown>] {
  const dir = join(CORPUS, "acta-06-chain-link-03-prefixed");
  return [
    JSON.parse(readFileSync(join(dir, "receipt.json"), "utf-8")),
    JSON.parse(readFileSync(join(dir, "predecessor.json"), "utf-8")),
    JSON.parse(readFileSync(join(dir, "acta-keys.json"), "utf-8")),
  ];
}

describe("ACTA chain link forms", () => {
  it("recompute matches the carried form", () => {
    const [receipt, pred] = acta06();
    const adapter = new ActaAdapter();
    const step = adapter.chainStep(receipt);

    const prefixed = step.recompute(pred);
    expect(prefixed.startsWith("sha256:")).toBe(true);

    const barePayload = { ...(receipt.payload as Record<string, unknown>) };
    barePayload.previousReceiptHash = prefixed.slice("sha256:".length);
    const bareStep = adapter.chainStep({ payload: barePayload, signature: receipt.signature });
    const bareRecompute = bareStep.recompute(pred);
    expect(bareRecompute.startsWith("sha256:")).toBe(false);
    expect(bareRecompute).toBe(prefixed.slice("sha256:".length));
  });

  it("an unknown prefix fails the chain axis rather than being normalised", () => {
    const [receipt, pred, keys] = acta06();
    const mutated = JSON.parse(JSON.stringify(receipt)) as Record<string, unknown>;
    const payload = mutated.payload as Record<string, unknown>;
    const carried = payload.previousReceiptHash as string;
    expect(carried.startsWith("sha256:")).toBe(true);
    payload.previousReceiptHash = "sha512:" + carried.slice("sha256:".length);

    const result = verify(mutated, [new ActaAdapter()], keys, pred);
    const chain = result.axes.find((a) => a.axis === "chain");
    expect(chain?.result).toBe("FAIL");
    expect(result.verdict).toBe("unverified");
  });
});
