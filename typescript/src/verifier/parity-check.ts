/**
 * Reproducible equivalence gate - run ALL corpus vectors through the TS verifier
 * and compare each verdict to the EXPECTED manifest outcome, exactly as the Python
 * `runner.main()` does. ML-DSA vectors are tolerated as INCOMPLETE->PASS only for
 * reason_code `signature_skipped_no_dilithium` (Node has no ML-DSA, matching the
 * Python verifier without dilithium-py).
 *
 * Run directly:  node --import tsx typescript/src/verifier/parity-check.ts
 * or via the test `tests/verifier-parity.test.ts`, which asserts 41/41.
 */

import { resolve } from "node:path";

import { runCorpus, tolerated, type VectorOutcome } from "./runner.js";

/** Default corpus root: `<repo>/verifier/conformance-vectors`. */
export function defaultCorpusRoot(): string {
  // src/verifier -> ../../.. -> repo root -> verifier/conformance-vectors
  return resolve(__dirname, "..", "..", "..", "verifier", "conformance-vectors");
}

export interface ParityReport {
  results: VectorOutcome[];
  passed: number;
  total: number;
  lines: string[];
}

export function runParity(corpusRoot = defaultCorpusRoot()): ParityReport {
  const results = runCorpus(corpusRoot);
  const lines: string[] = [];
  for (const r of results) {
    const ok = tolerated(r);
    const mark = ok ? "ok" : "FAIL";
    lines.push(`  [${mark.padStart(4)}] ${r.dir.padEnd(38)} expect=${r.expectedOutcome.padEnd(11)} got=${r.actualVerdict}`);
    if (!ok) lines.push(`         ${r.detail}`);
  }
  const passed = results.filter(tolerated).length;
  lines.push("");
  lines.push(`  => ${passed}/${results.length} vectors matched expected outcome`);
  return { results, passed, total: results.length, lines };
}

function main(): number {
  // Corpus root from arg / env, else the in-tree default.
  const root = process.argv[2] || process.env.ASQAV_CORPUS_ROOT || defaultCorpusRoot();
  const report = runParity(root);
  for (const line of report.lines) console.log(line);
  return report.passed === report.total ? 0 : 1;
}

// Run when invoked as a script (not when imported by the test).
if (typeof require !== "undefined" && require.main === module) {
  process.exit(main());
}
