// Reproducible npm pack of the SDK, corpus included; runs under node --test
// and under vitest. NEGCTL: a file rewritten between packs must fail the check

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import {
  CORPUS_DIR,
  PKG_DIR,
  checkReproducible,
  packDigest,
  stagePackage,
} from "../scripts/pack-reproducible.mjs";

const runner = process.env.VITEST ? await import("vitest") : await import("node:test");
const test = (name, fn) => (process.env.VITEST ? runner.test(name, fn, 120_000) : runner.test(name, fn));

const LICENSE_GATE = path.join(PKG_DIR, "..", "python", "scripts", "check_distribution_licenses.py");
const CORPUS_MIN_EXPECTED = Number(
  /^CORPUS_MIN_EXPECTED = (\d+)$/m.exec(readFileSync(LICENSE_GATE, "utf8"))[1],
);

function withTemp(stage, fn) {
  const dir = stage();
  try {
    return fn(dir);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

function fixturePackage() {
  const dir = mkdtempSync(path.join(tmpdir(), "asqav-pack-fixture-"));
  writeFileSync(path.join(dir, "package.json"), JSON.stringify({ name: "fixture", version: "1.0.0", files: ["a.txt"] }));
  writeFileSync(path.join(dir, "a.txt"), "first\n");
  return dir;
}

test("two packs give one sha256", () => {
  const result = withTemp(stagePackage, (pkgDir) => checkReproducible(pkgDir));
  assert.equal(result.ok, true);
  assert.equal(result.digests[0], result.digests[1]);
  const other = withTemp(stagePackage, (pkgDir) => packDigest(pkgDir).sha256);
  assert.equal(other, result.digests[0]);
  const vectors = result.entries.filter((p) => p.startsWith(`${CORPUS_DIR}/`) && p.endsWith("/expected.json"));
  assert.ok(CORPUS_MIN_EXPECTED > 0);
  assert.ok(vectors.length >= CORPUS_MIN_EXPECTED, `${vectors.length} vectors packed`);
});

test("the files list keeps dist, README.md, LICENSE and adds the corpus", () => {
  const { files } = JSON.parse(readFileSync(path.join(PKG_DIR, "package.json"), "utf8"));
  for (const entry of ["dist", "README.md", "LICENSE", "NOTICE", CORPUS_DIR]) assert.ok(files.includes(entry), entry);
});

test("NEGCTL: a packed file rewritten between packs fails the check", () => {
  const result = withTemp(fixturePackage, (dir) =>
    checkReproducible(dir, { between: (d) => writeFileSync(path.join(d, "a.txt"), "second\n") }),
  );
  assert.equal(result.ok, false);
  assert.notEqual(result.digests[0], result.digests[1]);
});

test("NEGCTL: a dir npm cannot pack throws, never an empty digest", () => {
  withTemp(
    () => mkdtempSync(path.join(tmpdir(), "asqav-pack-empty-")),
    (dir) => assert.throws(() => packDigest(dir), /npm pack exited/),
  );
});
