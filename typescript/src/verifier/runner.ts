/**
 * Conformance-vector runner - drive the oracle over the vector corpus.
 * A port of the Python oracle's `verifier/oracle/runner.py`.
 *
 * The corpus uses the AERF dir-per-vector layout: a top-level `manifest.json`
 * lists `{dir, format, outcome, failure_class, reason_code, notes}` and each
 * `<NN-name>/` directory carries `receipt.json`, an `expected.json`, optional
 * `predecessor.json`, and optional key material (`jwks.json` for Asqav-native,
 * `keys.json` for AERF, `acta-keys.json` for ACTA, `did_map.json` for
 * agentreceipts).
 *
 * `outcome` speaks the public verdict vocabulary (criteria 418/438):
 * verified / verified_keyed / unverified, and an `unverified` entry pins its
 * `failure_class` (invalid / unverifiable) so the two classes are never
 * collapsed. Every receipt/record file is parsed with duplicate-member
 * rejection (criterion 419); a receipt that fails to parse is a terminal
 * unverified/unverifiable outcome, never verified.
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

import { ADAPTERS } from "./index.js";
import { verify, VERDICT_UNVERIFIED, VERDICT_VERIFIED, FAILURE_UNVERIFIABLE } from "./core.js";
import { parseJsonPreservingFloats, parseJsonStrict } from "./canonical.js";
import type { KeyProvider } from "./adapter.js";

/** The result of running one corpus vector against the oracle. */
export interface VectorOutcome {
  dir: string;
  expectedOutcome: string;
  actualVerdict: string;
  ok: boolean;
  reasonCode: string;
  detail: string;
  expectedFailureClass: string;
  actualFailureClass: string;
}

interface ManifestEntry {
  dir: string;
  format: string;
  outcome: string;
  failure_class?: string;
  reason_code?: string;
  notes?: string;
}

function loadJson(path: string): Record<string, unknown> | null {
  // Receipts and predecessors are parsed float-preserving so a `500.0` literal
  // survives to the canonicaliser (JSON.parse otherwise collapses it to `500`).
  // The same parser rejects a duplicated member name at any depth (419).
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

function parseFailureOutcome(
  name: string,
  expectedOutcome: string,
  expectedFailureClass: string,
  reasonCode: string,
  exc: unknown,
): VectorOutcome {
  // A terminal ingest failure: nothing was hashed, checked, or verified (419).
  const detail = `ingest=FAIL(terminal parse failure before any hashing: ${(exc as Error).message})`;
  const ok = expectedOutcome === VERDICT_UNVERIFIED && (expectedFailureClass === "" || expectedFailureClass === FAILURE_UNVERIFIABLE);
  return {
    dir: name,
    expectedOutcome,
    actualVerdict: VERDICT_UNVERIFIED,
    ok,
    reasonCode,
    detail,
    expectedFailureClass,
    actualFailureClass: FAILURE_UNVERIFIABLE,
  };
}

/** Run a single vector directory and compare against its expected outcome. */
export function runOne(
  vecDir: string,
  fmt: string,
  expectedOutcome: string,
  reasonCode = "",
  expectedFailureClass = "",
): VectorOutcome {
  const name = vecDir.split(/[/\\]/).pop() ?? vecDir;
  let receipt: Record<string, unknown>;
  let predecessor: Record<string, unknown> | null;
  let keyProvider: KeyProvider;
  try {
    receipt = loadJson(join(vecDir, "receipt.json")) ?? {};
    predecessor = loadJson(join(vecDir, "predecessor.json"));
    keyProvider = keyProviderFor(vecDir, fmt);
  } catch (exc) {
    if (exc instanceof SyntaxError) {
      // Terminal ingest failure (419): never hash, check, or verify the bytes.
      return parseFailureOutcome(name, expectedOutcome, expectedFailureClass, reasonCode, exc);
    }
    throw exc;
  }
  const result = verify(receipt, ADAPTERS, keyProvider, predecessor);

  let ok = result.verdict === expectedOutcome;
  const actualFailureClass = result.failureClass ?? "";
  if (ok && expectedFailureClass !== "") {
    ok = actualFailureClass === expectedFailureClass;
  }
  const detail = result.axes.map((a) => `${a.axis}=${a.result}(${a.note})`).join("; ");
  return {
    dir: name,
    expectedOutcome,
    actualVerdict: result.verdict,
    ok,
    reasonCode,
    detail,
    expectedFailureClass,
    actualFailureClass,
  };
}

/** Run every vector named in `corpusRoot/manifest.json`. */
export function runCorpus(corpusRoot: string): VectorOutcome[] {
  // The manifest rides the same strict parser (419): duplicates fail the run.
  const manifest = parseJsonStrict(
    readFileSync(join(corpusRoot, "manifest.json"), "utf-8"),
  ) as ManifestEntry[];
  return manifest.map((entry) =>
    runOne(
      join(corpusRoot, entry.dir),
      entry.format,
      entry.outcome,
      entry.reason_code ?? "",
      entry.failure_class ?? "",
    ),
  );
}

/** Accept the stronger verified only for the optional-dep ML-DSA skip vector. */
export function tolerated(outcome: VectorOutcome): boolean {
  return (
    outcome.ok ||
    (outcome.expectedOutcome === VERDICT_UNVERIFIED &&
      outcome.actualVerdict === VERDICT_VERIFIED &&
      outcome.reasonCode === "signature_skipped_no_dilithium")
  );
}
