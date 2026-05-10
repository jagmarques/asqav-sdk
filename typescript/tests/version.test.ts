/**
 * Version consistency: CLI_VERSION in src/cli.ts must match version in package.json.
 *
 * Prevents the drift bug where the CLI prints a different version than the
 * package actually publishes (mirrors the Python __version__/pyproject drift
 * that yanked asqav 0.3.7 from PyPI).
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { CLI_VERSION } from "../src/cli.js";

describe("version consistency", () => {
  it("CLI_VERSION matches package.json version", () => {
    const pkgPath = new URL("../package.json", `file://${__filename}`);
    const pkg = JSON.parse(readFileSync(pkgPath, "utf8")) as {
      version: string;
    };
    expect(CLI_VERSION).toBe(pkg.version);
  });
});
