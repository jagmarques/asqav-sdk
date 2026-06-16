/**
 * Conformance-vector runner - drive the oracle over the vector corpus.
 * A port of the Python oracle's `verifier/oracle/runner.py`.
 *
 * The corpus uses the AERF dir-per-vector layout: a top-level `manifest.json`
 * lists `{dir, format, outcome, reason_code, notes}` and each `<NN-name>/`
 * directory carries `receipt.json`, an `expected.json`, optional
 * `predecessor.json`, and optional key material (`jwks.json` for Asqav-native,
 * `keys.json` for AERF, `acta-keys.json` for ACTA, `did_map.json` for agentreceipts).
 *
 * `outcome` maps to the verdict one-for-one (PASS/FAIL/INCOMPLETE). INCOMPLETE
 * lets the corpus carry a vector whose signature axis cannot be checked (a real
 * ML-DSA-65 receipt without an ML-DSA library), pinning that the verifier
 * downgrades rather than false-PASSing.
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

import { ADAPTERS } from "./index.js";
import { verify, type Verdict } from "./core.js";
import { parseJsonPreservingFloats } from "./canonical.js";
import type { KeyProvider } from "./adapter.js";

const OUTCOME_TO_VERDICT: Record<string, Verdict> = {
  PASS: "PASS",
  FAIL: "FAIL",
  INCOMPLETE: "INCOMPLETE",
};

/** The result of running one corpus vector against the oracle. */
export interface VectorOutcome {
  dir: string;
  expectedOutcome: string;
  actualVerdict: Verdict | string;
  ok: boolean;
  reasonCode: string;
  detail: string;
}

interface ManifestEntry {
  dir: string;
  format: string;
  outcome: string;
  reason_code?: string;
  notes?: string;
}

function loadJson(path: string): Record<string, unknown> | null {
  // Receipts and predecessors are parsed float-preserving so a `500.0` literal
  // survives to the canonicaliser (JSON.parse otherwise collapses it to `500`).
  return existsSync(path)
    ? (parseJsonPreservingFloats(readFileSync(path, "utf-8")) as Record<string, unknown>)
    : null;
}

function keyProviderFor(vecDir: string, fmt: string): KeyProvider {
  if (fmt === "asqav-native") return loadJson(join(vecDir, "jwks.json"));
  if (fmt === "aerf") return loadJson(join(vecDir, "keys.json"));
  if (fmt === "acta") return loadJson(join(vecDir, "acta-keys.json"));
  if (fmt === "agentreceipts") return loadJson(join(vecDir, "did_map.json"));
  if (fmt === "pipelock-evidence-v2") return loadJson(join(vecDir, "keys.json"));
  return null;
}

/** Run a single vector directory and compare against its expected outcome. */
export function runOne(
  vecDir: string,
  fmt: string,
  expectedOutcome: string,
  reasonCode = "",
): VectorOutcome {
  const receipt = loadJson(join(vecDir, "receipt.json")) ?? {};
  const predecessor = loadJson(join(vecDir, "predecessor.json"));
  const keyProvider = keyProviderFor(vecDir, fmt);
  const result = verify(receipt, ADAPTERS, keyProvider, predecessor);

  const wantVerdict = OUTCOME_TO_VERDICT[expectedOutcome];
  const ok = wantVerdict !== undefined && result.verdict === wantVerdict;
  const detail = result.axes.map((a) => `${a.axis}=${a.result}(${a.note})`).join("; ");
  const name = vecDir.split(/[/\\]/).pop() ?? vecDir;
  return { dir: name, expectedOutcome, actualVerdict: result.verdict, ok, reasonCode, detail };
}

/** Run every vector named in `corpusRoot/manifest.json`. */
export function runCorpus(corpusRoot: string): VectorOutcome[] {
  const manifest = JSON.parse(readFileSync(join(corpusRoot, "manifest.json"), "utf-8")) as ManifestEntry[];
  return manifest.map((entry) =>
    runOne(join(corpusRoot, entry.dir), entry.format, entry.outcome, entry.reason_code ?? ""),
  );
}

/** Accept the stronger PASS only for the optional-dep ML-DSA skip vector, never broadly. */
export function tolerated(outcome: VectorOutcome): boolean {
  return (
    outcome.ok ||
    (outcome.expectedOutcome === "INCOMPLETE" &&
      outcome.actualVerdict === "PASS" &&
      outcome.reasonCode === "signature_skipped_no_dilithium")
  );
}
