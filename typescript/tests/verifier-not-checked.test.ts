/**
 * Non-coverage is declared on every `verify()` result, passing ones included.
 * A port of python/tests/test_not_checked_declaration.py.
 */

import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import * as ts from "typescript";
import { describe, expect, it } from "vitest";

import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import {
  AXIS_ORDER,
  NOT_CHECKED,
  coverageDeclaration,
  notCheckedDeclaration,
} from "../src/verifier/not-checked.js";

const VECTORS = resolve(__dirname, "..", "..", "verifier", "conformance-vectors");
const REQUIRED_KEYS = ["check", "condition", "reason", "requirement"];

function loadVector(name: string): [Record<string, unknown>, Record<string, unknown>] {
  const dir = join(VECTORS, name);
  return [
    JSON.parse(readFileSync(join(dir, "receipt.json"), "utf-8")) as Record<string, unknown>,
    JSON.parse(readFileSync(join(dir, "jwks.json"), "utf-8")) as Record<string, unknown>,
  ];
}

describe("notChecked declaration", () => {
  it("is non-empty and well-formed", () => {
    expect(NOT_CHECKED.length, "an empty declaration claims total coverage").toBeGreaterThan(0);
    for (const entry of NOT_CHECKED) {
      expect(Object.keys(entry).sort(), entry.check).toEqual(REQUIRED_KEYS);
      expect(entry.check).toBeTruthy();
      expect(entry.check).toBe(entry.check.toLowerCase());
      expect(entry.reason.trim()).toBeTruthy();
      // `condition` is deliberately nullable: null means never performed.
      expect(entry.condition === null || entry.condition.trim().length > 0).toBe(true);
    }
  });

  it("names no check twice", () => {
    const names = NOT_CHECKED.map((e) => e.check);
    expect(new Set(names).size, names.join(",")).toBe(names.length);
  });

  it("rides a rejected input", () => {
    const result = verify({ hello: "world" }, ADAPTERS, null);
    expect(result.verdict).toBe("unverified");
    expect(result.notChecked).toHaveLength(NOT_CHECKED.length);
  });

  it("rides a passing result", () => {
    const [receipt, jwks] = loadVector("asqav-01-genesis-permit");
    const result = verify(receipt, ADAPTERS, jwks);
    const signature = result.axes.find((a) => a.axis === "signature");
    // PASS here proves the crypto ran, so the declaration rides a real verified.
    expect(signature?.result).toBe("PASS");
    expect(result.verdict).toBe("verified");
    expect(result.notChecked).toHaveLength(NOT_CHECKED.length);
  });

  it("cannot be narrowed by a caller mutating an earlier result", () => {
    const first = verify({ hello: "world" }, ADAPTERS, null);
    first.notChecked.length = 0;
    const firstEntry = notCheckedDeclaration()[0];
    firstEntry.reason = "mutated";
    expect(verify({ hello: "world" }, ADAPTERS, null).notChecked).toHaveLength(NOT_CHECKED.length);
    expect(notCheckedDeclaration()[0].reason).not.toBe("mutated");
  });

  it("is on every return path of verify(), parsed from source", () => {
    // Parsed rather than exercised: the path no test reaches would ship undeclared.
    const source = readFileSync(resolve(__dirname, "..", "src", "verifier", "core.ts"), "utf-8");
    const file = ts.createSourceFile("core.ts", source, ts.ScriptTarget.Latest, true);
    let verifyBody: ts.Block | undefined;
    ts.forEachChild(file, (node) => {
      if (ts.isFunctionDeclaration(node) && node.name?.text === "verify" && node.body) {
        verifyBody = node.body;
      }
    });
    expect(verifyBody, "verify() not found; the gate is vacuous").toBeDefined();

    const returns: ts.ReturnStatement[] = [];
    const visit = (node: ts.Node): void => {
      if (ts.isReturnStatement(node)) returns.push(node);
      ts.forEachChild(node, visit);
    };
    ts.forEachChild(verifyBody as ts.Block, visit);
    expect(returns.length, "verify() has no returns; the gate is vacuous").toBeGreaterThan(0);

    for (const ret of returns) {
      const expr = ret.expression;
      const line = file.getLineAndCharacterOfPosition(ret.getStart()).line + 1;
      expect(expr !== undefined && ts.isObjectLiteralExpression(expr), `line ${line}: not an object`).toBe(true);
      const names = (expr as ts.ObjectLiteralExpression).properties
        .filter(ts.isPropertyAssignment)
        .map((p) => (ts.isIdentifier(p.name) || ts.isStringLiteral(p.name) ? p.name.text : ""));
      expect(names, `verify() returns at line ${line} without notChecked`).toContain("notChecked");
    }
  });
});

// The axes after the structure gate on the normal path, pinned verbatim so a
// reorder of the verifier's sequence fails here instead of editing the pin.
const AXES_AFTER_STRUCTURE = [
  "signature",
  "chain",
  "seq",
  "expiry",
  "nonce",
  "key_binding",
  "counterparty",
  "payload_digest",
  "skew",
  "key_status",
  "issuer_bind",
];

describe("coverage declaration", () => {
  it("rides a full run with stopped_at null and the not_implemented table", () => {
    const [receipt, jwks] = loadVector("asqav-01-genesis-permit");
    const result = verify(receipt, ADAPTERS, jwks);
    // The pin is the live axis sequence itself: the coverage constant must track it.
    expect(result.axes.map((a) => a.axis)).toEqual(["structure", ...AXES_AFTER_STRUCTURE]);
    expect(result.coverage.stopped_at).toBeNull();
    const entries = result.coverage.checks_not_evaluated;
    expect(entries.map((e) => e.id)).toEqual(NOT_CHECKED.map((e) => e.check));
    for (const entry of entries) {
      expect(entry.reason).toBe("not_implemented");
      expect(entry.status).toBe("not_implemented");
      expect(Object.keys(entry).slice(0, 3)).toEqual(["id", "reason", "status"]);
      expect(entry.requirement).toBeTruthy();
      expect("condition" in entry).toBe(true);
    }
  });

  it("marks the axes after a stop at the structure gate as not_reached", () => {
    const result = verify("nope" as unknown as Record<string, unknown>, ADAPTERS, null);
    expect(result.coverage.stopped_at).toBe("structure");
    const entries = result.coverage.checks_not_evaluated;
    const notImpl = entries.filter((e) => e.reason === "not_implemented");
    const notReached = entries.filter((e) => e.reason === "not_reached");
    // not_implemented entries come first, in NOT_CHECKED table order.
    expect(notImpl.map((e) => e.id)).toEqual(NOT_CHECKED.map((e) => e.check));
    expect(notReached.map((e) => e.id)).toEqual(AXES_AFTER_STRUCTURE);
    for (const entry of notReached) {
      expect(entry.status).toBe("implemented");
      expect(Object.keys(entry)).toEqual(["id", "reason", "status"]);
    }
  });

  it("keeps the not_implemented ids equal to this language's table, in order", () => {
    for (const axes of [[], [{ axis: "issuer_bind" }]] as Array<Array<{ axis: string }>>) {
      const entries = coverageDeclaration(axes).checks_not_evaluated;
      expect(entries.filter((e) => e.reason === "not_implemented").map((e) => e.id)).toEqual(
        NOT_CHECKED.map((e) => e.check),
      );
    }
  });

  it("cannot be narrowed by a caller mutating an earlier result's block", () => {
    const first = verify({ hello: "world" }, ADAPTERS, null);
    first.coverage.checks_not_evaluated.length = 0;
    const fresh = verify({ hello: "world" }, ADAPTERS, null);
    expect(fresh.coverage.checks_not_evaluated.length).toBe(
      NOT_CHECKED.length + AXES_AFTER_STRUCTURE.length,
    );
  });

  it("is on every return path of verify(), parsed from source", () => {
    // Parsed rather than exercised: the path no test reaches would ship undeclared.
    const source = readFileSync(resolve(__dirname, "..", "src", "verifier", "core.ts"), "utf-8");
    const file = ts.createSourceFile("core.ts", source, ts.ScriptTarget.Latest, true);
    let verifyBody: ts.Block | undefined;
    ts.forEachChild(file, (node) => {
      if (ts.isFunctionDeclaration(node) && node.name?.text === "verify" && node.body) {
        verifyBody = node.body;
      }
    });
    expect(verifyBody, "verify() not found; the gate is vacuous").toBeDefined();

    const returns: ts.ReturnStatement[] = [];
    const visit = (node: ts.Node): void => {
      if (ts.isReturnStatement(node)) returns.push(node);
      ts.forEachChild(node, visit);
    };
    ts.forEachChild(verifyBody as ts.Block, visit);
    expect(returns.length, "verify() has no returns; the gate is vacuous").toBeGreaterThan(0);

    for (const ret of returns) {
      const expr = ret.expression;
      const line = file.getLineAndCharacterOfPosition(ret.getStart()).line + 1;
      expect(expr !== undefined && ts.isObjectLiteralExpression(expr), `line ${line}: not an object`).toBe(true);
      const names = (expr as ts.ObjectLiteralExpression).properties
        .filter(ts.isPropertyAssignment)
        .map((p) => (ts.isIdentifier(p.name) || ts.isStringLiteral(p.name) ? p.name.text : ""));
      expect(names, `verify() returns at line ${line} without coverage`).toContain("coverage");
    }
  });
});
