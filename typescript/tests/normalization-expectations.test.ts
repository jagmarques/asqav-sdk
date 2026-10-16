/** Normalization expectations, TS half (488 C6): JSON-table check. */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { verify } from "../src/verifier/core.js";
import { ADAPTERS, type VerificationContext } from "../src/verifier/index.js";

interface Case {
  name: string;
  patch: { payload: Record<string, unknown>; remove?: string[] };
  context: Record<string, unknown>;
  expect: { axes: Record<string, string>; context_echo: Record<string, unknown> };
}

const TABLE = JSON.parse(
  readFileSync(resolve(__dirname, "..", "..", "verifier", "normalization-expectations.json"), "utf-8"),
) as { base_vector: string; cases: Case[] };

const SNAKE_TO_CAMEL: Record<string, string> = {
  evaluation_time: "evaluationTime",
  trust_roots: "trustRoots",
  revocation_snapshot: "revocationSnapshot",
  policy_revision: "policyRevision",
  evidence_knowledge_cutoff: "evidenceKnowledgeCutoff",
  evaluator: "evaluator",
};

function loadBase(): Record<string, unknown> {
  const raw = readFileSync(
    resolve(
      __dirname,
      "..",
      "..",
      "verifier",
      "conformance-vectors",
      TABLE.base_vector,
      "receipt.json",
    ),
    "utf-8",
  );
  return JSON.parse(raw) as Record<string, unknown>;
}

describe("normalization expectations (488 C6)", () => {
  it("table stays populated", () => {
    expect(TABLE.cases.length).toBeGreaterThanOrEqual(8);
  });

  for (const c of TABLE.cases) {
    it(c.name, () => {
      const doc = loadBase();
      Object.assign(doc["payload"] as Record<string, unknown>, c.patch.payload);
      for (const key of c.patch.remove ?? []) {
        delete (doc["payload"] as Record<string, unknown>)[key];
      }
      const context: VerificationContext = {};
      for (const [k, v] of Object.entries(c.context)) {
        (context as Record<string, unknown>)[SNAKE_TO_CAMEL[k] ?? k] = v;
      }
      const result = verify(doc, ADAPTERS, null, null, context);
      const axes: Record<string, string> = {};
      for (const a of result.axes) axes[a.axis] = a.result;
      for (const [axis, expected] of Object.entries(c.expect.axes)) {
        expect(axes[axis], `${c.name}: axis ${axis} (all ${JSON.stringify(axes)})`).toBe(expected);
      }
      expect(result.context, `${c.name}: no resolved context`).toBeDefined();
      const echoed = result.context as Record<string, unknown>;
      for (const [snake, expected] of Object.entries(c.expect.context_echo)) {
        const camel = SNAKE_TO_CAMEL[snake] ?? snake;
        expect(echoed[camel], `${c.name}: context.${camel}`).toEqual(expected);
      }
    });
  }
});
