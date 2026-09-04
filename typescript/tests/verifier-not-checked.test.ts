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
import { NOT_CHECKED, notCheckedDeclaration } from "../src/verifier/not-checked.js";

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
