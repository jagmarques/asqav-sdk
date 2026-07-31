/** Pins the packaged LICENSE text to package.json's declared license. */

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const PKG_DIR = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(PKG_DIR, "..");

interface PackedFile {
  path: string;
}
interface PackResult {
  files: PackedFile[];
}

/** Asks npm which files it would ship, so a fix to the wrong path can't pass */
function packedFiles(): PackedFile[] {
  const out = execFileSync("npm", ["pack", "--dry-run", "--json"], {
    cwd: PKG_DIR,
    encoding: "utf8",
  });
  const [result] = JSON.parse(out) as PackResult[];
  return result.files;
}

describe("license consistency", () => {
  it("package.json declares the Elastic license and lists LICENSE in files", () => {
    const pkg = JSON.parse(
      readFileSync(path.join(PKG_DIR, "package.json"), "utf8"),
    ) as { license: string; files: string[] };
    expect(pkg.license).toBe("LicenseRef-Elastic-License-2.0");
    expect(pkg.files).toContain("LICENSE");
  });

  it(
    "the LICENSE file npm actually packs matches the repo's Elastic LICENSE, not MIT",
    () => {
      const packed = packedFiles();
      const licenseEntry = packed.find((f) => f.path.toUpperCase() === "LICENSE");
      expect(licenseEntry).toBeDefined();

      const packagedLicenseText = readFileSync(path.join(PKG_DIR, "LICENSE"), "utf8");
      const canonicalLicenseText = readFileSync(path.join(REPO_ROOT, "LICENSE"), "utf8");

      // Pins the exact pre-relicense grant this test guards against
      expect(packagedLicenseText).not.toMatch(/^MIT License/);
      expect(packagedLicenseText).not.toMatch(/Permission is hereby granted, free of charge/);

      // Packaged copy must carry the same text as the repo's canonical LICENSE
      expect(packagedLicenseText).toBe(canonicalLicenseText);
    },
    // A real `npm pack` subprocess under a full parallel run can exceed the
    // 5s default when the CI box is busy running 50+ other test files
    20000,
  );
});
