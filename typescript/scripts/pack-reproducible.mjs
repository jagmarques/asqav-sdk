// Packs the SDK twice and compares tarball sha256; the corpus is staged in
// from ../verifier, since npm pack never reaches outside the package dir

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const PKG_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const CORPUS_DIR = "conformance-vectors";
export const CORPUS_PAPERS = ["NOTICE", "LICENSE"];

const filesOf = (pkgDir) => JSON.parse(readFileSync(path.join(pkgDir, "package.json"), "utf8")).files ?? [];

/** Copy the corpus and papers into dest when package.json files lists it. */
export function stageCorpus(pkgDir, dest = pkgDir) {
  if (!filesOf(pkgDir).includes(CORPUS_DIR)) return null;
  const repo = path.resolve(pkgDir, "..");
  const src = path.join(repo, "verifier", CORPUS_DIR);
  if (!existsSync(src)) throw new Error(`corpus source missing: ${src}`);
  const out = path.join(dest, CORPUS_DIR);
  rmSync(out, { recursive: true, force: true });
  cpSync(src, out, { recursive: true });
  for (const name of CORPUS_PAPERS) cpSync(path.join(repo, "conformance", name), path.join(out, name));
  return out;
}

/** A fresh temp copy of what npm would pack from pkgDir, corpus included. */
export function stagePackage(pkgDir = PKG_DIR) {
  const dest = mkdtempSync(path.join(tmpdir(), "asqav-pack-src-"));
  for (const entry of ["package.json", ...filesOf(pkgDir)]) {
    const src = path.join(pkgDir, entry);
    if (entry !== CORPUS_DIR && existsSync(src)) cpSync(src, path.join(dest, entry), { recursive: true });
  }
  stageCorpus(pkgDir, dest);
  return dest;
}

/** sha256 and entry list of `npm pack --ignore-scripts` run in dir. */
export function packDigest(dir) {
  const dest = mkdtempSync(path.join(tmpdir(), "asqav-pack-out-"));
  try {
    const run = spawnSync("npm", ["pack", "--ignore-scripts", "--json", "--pack-destination", dest], {
      cwd: dir,
      encoding: "utf8",
    });
    if (run.error) throw run.error;
    if (run.status !== 0) throw new Error(`npm pack exited ${run.status}: ${run.stderr.trim()}`);
    const [info] = JSON.parse(run.stdout);
    const tarball = path.join(dest, info.filename);
    if (!existsSync(tarball)) throw new Error(`npm pack wrote no tarball at ${tarball}`);
    const sha256 = createHash("sha256").update(readFileSync(tarball)).digest("hex");
    return { sha256, filename: info.filename, entries: info.files.map((f) => f.path) };
  } finally {
    rmSync(dest, { recursive: true, force: true });
  }
}

/** Pack dir twice, calling between(dir) in the middle; ok when both match. */
export function checkReproducible(dir, { between } = {}) {
  const first = packDigest(dir);
  if (between) between(dir);
  const second = packDigest(dir);
  return { ok: first.sha256 === second.sha256, digests: [first.sha256, second.sha256], entries: first.entries };
}

function main(argv) {
  if (argv[0] === "--stage-corpus") return stageCorpus(PKG_DIR) ? 0 : 1;
  if (argv[0] === "--unstage-corpus") {
    rmSync(path.join(PKG_DIR, CORPUS_DIR), { recursive: true, force: true });
    return 0;
  }
  const staged = [stagePackage(), stagePackage()];
  try {
    const result = checkReproducible(staged[0]);
    const other = packDigest(staged[1]).sha256;
    const ok = result.ok && other === result.digests[0];
    console.log(`${ok ? "OK" : "FAIL"}: sha256 ${[...result.digests, other].join(" ")}`);
    return ok ? 0 : 1;
  } finally {
    for (const dir of staged) rmSync(dir, { recursive: true, force: true });
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = main(process.argv.slice(2));
}
