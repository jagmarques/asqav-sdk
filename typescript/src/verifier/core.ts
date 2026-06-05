/**
 * The shared verification core - format detection, dispatch, and the verdict.
 * A port of the Python oracle's `verifier/oracle/core.py`.
 *
 * `verify(doc, ...)` detects the format, then drives the adapter through the
 * shared axes: structure, signature, chain. It proves only what the bytes prove -
 * a valid signature over the canonical bytes, a reproducible chain link, and
 * structural presence at time T. It never attests the behaviour or correctness of
 * the recorded action.
 */

import type { FormatAdapter, KeyProvider } from "./adapter.js";
import { FAIL, PASS, SKIPPED, verifySignature, type VerifyState } from "./crypto.js";

/** One verification axis outcome. */
export interface AxisResult {
  /** Which check (structure / signature / chain / extra). */
  axis: string;
  /** PASS / FAIL / SKIPPED. */
  result: VerifyState;
  /** Human-readable detail for the report. */
  note: string;
}

export type Verdict = "PASS" | "FAIL" | "INCOMPLETE";

/** The aggregate outcome of verifying one receipt. */
export interface VerifyResult {
  /** The matched adapter name, or "unknown". */
  fmt: string;
  axes: AxisResult[];
  /**
   * PASS only when every non-skipped axis passed AND the signature was actually
   * checked; a skipped signature downgrades to INCOMPLETE, never a PASS.
   */
  verdict: Verdict;
}

/** Return the first adapter whose structural fingerprint matches `doc`. */
export function detect(
  doc: Record<string, unknown>,
  adapters: FormatAdapter[],
): FormatAdapter | null {
  return adapters.find((a) => a.detect(doc)) ?? null;
}

function signatureAxis(
  ad: FormatAdapter,
  doc: Record<string, unknown>,
  keyProvider: KeyProvider,
): AxisResult {
  const sm = ad.extractSignature(doc);
  const [pk, note] = ad.resolveKey(doc, keyProvider);
  if (pk === null) {
    return { axis: "signature", result: SKIPPED, note: `no key: ${note}` };
  }
  const msg = ad.signingInput(doc);
  const { result, note: why } = verifySignature(sm.alg, pk, msg, sm.sig);
  return { axis: "signature", result, note: why };
}

function chainAxis(
  ad: FormatAdapter,
  doc: Record<string, unknown>,
  adapters: FormatAdapter[],
  predecessor: Record<string, unknown> | null,
): AxisResult {
  const step = ad.chainStep(doc);
  if (step.isGenesis) {
    return { axis: "chain", result: PASS, note: "genesis receipt (no predecessor link)" };
  }
  if (predecessor === null) {
    return { axis: "chain", result: SKIPPED, note: "no predecessor supplied" };
  }
  // A chain link must stay within one format; a cross-format predecessor is not a valid link.
  const predAd = detect(predecessor, adapters);
  if (predAd === null || predAd.name !== ad.name) {
    return { axis: "chain", result: FAIL, note: "predecessor is a different receipt format" };
  }
  const actual = step.recompute(predecessor);
  if (actual === step.prevField) {
    return { axis: "chain", result: PASS, note: "chain link rederives from predecessor" };
  }
  const exp = String(step.prevField).slice(0, 16);
  return { axis: "chain", result: FAIL, note: `chain break: expected ${exp}.. got ${actual.slice(0, 16)}..` };
}

/** Verify one parsed receipt and return a structured `VerifyResult`. */
export function verify(
  doc: Record<string, unknown>,
  adapters: FormatAdapter[],
  keyProvider: KeyProvider = null,
  predecessor: Record<string, unknown> | null = null,
): VerifyResult {
  const ad = detect(doc, adapters);
  if (ad === null) {
    return {
      fmt: "unknown",
      axes: [{ axis: "structure", result: FAIL, note: "no adapter recognises this receipt" }],
      verdict: "FAIL",
    };
  }

  const [structResult, structNote] = ad.schema(doc);
  const axes: AxisResult[] = [
    { axis: "structure", result: structResult, note: structNote },
    signatureAxis(ad, doc, keyProvider),
    chainAxis(ad, doc, adapters, predecessor),
  ];
  for (const [name, res, note] of ad.extraAxes(doc, keyProvider)) {
    axes.push({ axis: name, result: res, note });
  }

  const hasFail = axes.some((a) => a.result === FAIL);
  // Any skipped axis except a missing-predecessor chain blocks PASS: an unchecked
  // signature / counter-sign / PDP layer means INCOMPLETE, never a hiding PASS.
  const blockingSkip = axes.some((a) => a.result === SKIPPED && a.axis !== "chain");
  let verdict: Verdict;
  if (hasFail) {
    verdict = "FAIL";
  } else if (blockingSkip) {
    verdict = "INCOMPLETE";
  } else {
    verdict = "PASS";
  }
  return { fmt: ad.name, axes, verdict };
}
