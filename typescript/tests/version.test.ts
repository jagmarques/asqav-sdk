/**
 * CLI_VERSION in src/cli.ts must match package.json, so the CLI cannot print a version the
 * package does not publish (the Python __version__ drift that yanked asqav 0.3.7).
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
