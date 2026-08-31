/**
 * Reproducible equivalence gate: run every corpus vector through the TS verifier and compare each verdict
 * to the manifest, as Python's `runner.main()` does. ML-DSA vectors tolerate the no-dilithium skip.
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
    const got = r.actualFailureClass !== "" ? `${r.actualVerdict} (${r.actualFailureClass})` : r.actualVerdict;
    lines.push(`  [${mark.padStart(4)}] ${r.dir.padEnd(38)} expect=${r.expectedOutcome.padEnd(16)} got=${got}`);
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
