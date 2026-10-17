// Type resolution per draft anchor `type`
// protectmcp-absent is non-conformance = invalid; foreign/gap unverifiable
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { axisFailureClass } from "../src/verifier/core.js";
import {
  REGISTRY_EDITION,
  REGISTRY_NAMESPACES,
  checkStructure,
} from "../src/verifier/vrShim.js";

const NS_FILE = resolve(__dirname, "..", "..", "verifier", "registry", "type-namespaces.json");
const DOC = JSON.parse(readFileSync(NS_FILE, "utf-8")) as {
  registry_edition: string;
  namespaces: { namespace: string }[];
};

const DRAFT_TEN = new Set([
  "protectmcp:decision",
  "protectmcp:restraint",
  "protectmcp:lifecycle",
  "protectmcp:lifecycle:configuration_change",
  "protectmcp:lifecycle:risk_acceptance",
  "protectmcp:lifecycle:oversight_ruling",
  "protectmcp:lifecycle:code_authorship",
  "protectmcp:acknowledgment",
  "protectmcp:observation",
  "protectmcp:observation:result_bound",
]);

function payload(type: unknown): Record<string, unknown> {
  return {
    type,
    v: 1,
    mode: "payload",
    issuer_id: "iss_1",
    agent_id: "ag_1",
    org_id: "org_1",
    previousReceiptHash: "0".repeat(64),
    issued_at: "2026-09-01T12:00:00+00:00",
    action_ref: "a".repeat(64),
    payload_digest: { hash: "b".repeat(64), size: 1 },
    policy_digest: "pd",
    decision: "allow",
  };
}

const ROWS: [unknown, string, string | null, string][] = [
  ["protectmcp:decision", "PASS", null, "required fields present"],
  ["protectmcp:nope", "FAIL", "invalid", "unregistered type "],
  ["foreign:thing", "SKIPPED", "unverifiable", "outside the known namespace"],
  [
    "protectmcp:lifecycle:oversight_ruling",
    "SKIPPED",
    "unverifiable",
    "does not implement it",
  ],
  [42, "FAIL", "invalid", "malformed type "],
  [null, "FAIL", "invalid", "malformed type "],
];

describe("type resolution rows", () => {
  for (const [type, res, cls, needle] of ROWS) {
    it(`type ${String(type)} -> ${res}/${String(cls)}`, () => {
      const [got, note] = checkStructure(payload(type));
      expect(got, note).toBe(res);
      expect(note).toContain(needle);
      if (typeof type === "string") expect(note).toContain(type);
      expect(axisFailureClass("structure", got, note)).toBe(cls);
    });
  }

  it("namespace set matches the draft ten", () => {
    expect(new Set(DOC.namespaces.map((e) => e.namespace))).toEqual(DRAFT_TEN);
  });

  it("literals match the registry file", () => {
    expect(REGISTRY_NAMESPACES).toEqual(new Set(DOC.namespaces.map((e) => e.namespace)));
    expect(REGISTRY_EDITION).toBe(DOC.registry_edition);
  });
});
