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
import { PASS, UNVERIFIABLE, INVALID, verifySignature, type VerifyState } from "./crypto.js";
import { CLASSIFICATION, deriveVerdict, NO_REASON, type Classification } from "./taxonomy.js";

/** One verification axis outcome. */
export interface AxisResult {
  /** Which check (structure / signature / chain / extra). */
  axis: string;
  /** PASS / INVALID / UNVERIFIABLE / SKIPPED (criterion 418). */
  result: VerifyState;
  /** Human-readable detail for the report. */
  note: string;
  /** Closed failure-class token; "none" when nothing failed. */
  reasonCode: string;
}

export type Verdict = "PASS" | "INVALID" | "UNVERIFIABLE";

/** The aggregate outcome of verifying one receipt. */
export interface VerifyResult {
  /** The matched adapter name, or "unknown". */
  fmt: string;
  axes: AxisResult[];
  /**
   * Criterion 418: PASS only when every applicable axis passed and no
   * recomputation was left incomplete. One INVALID axis dominates; otherwise
   * any UNVERIFIABLE axis downgrades. A receipt is never PASS while a
   * recomputation failed. The expiry axis never folds the verdict (426).
   */
  verdict: Verdict;
  /** Wire classification mirroring the verdict: valid/invalid/unverifiable. */
  classification: Classification;
  // In-body origin attestation (v:2 signer) from the signed payload.
  // null for v:1. Never gates the verdict.
  signer: string | null;
}

/** Return the first adapter whose structural fingerprint matches `doc`. */
export function detect(
  doc: Record<string, unknown>,
  adapters: FormatAdapter[],
): FormatAdapter | null {
  if (typeof doc !== "object" || doc === null || Array.isArray(doc)) return null;
  return adapters.find((a) => a.detect(doc)) ?? null;
}

function signatureAxis(
  ad: FormatAdapter,
  doc: Record<string, unknown>,
  keyProvider: KeyProvider,
): AxisResult {
  const sm = ad.extractSignature(doc);
  const [pk, note, reason] = ad.resolveKey(doc, keyProvider);
  if (pk === null) {
    // The signature cannot be recomputed without its key; never a PASS
    return { axis: "signature", result: UNVERIFIABLE, note: `no key: ${note}`, reasonCode: reason };
  }
  const msg = ad.signingInput(doc);
  const { result, note: why, reasonCode } = verifySignature(sm.alg, pk, msg, sm.sig);
  return { axis: "signature", result, note: why, reasonCode };
}

function chainAxis(
  ad: FormatAdapter,
  doc: Record<string, unknown>,
  adapters: FormatAdapter[],
  predecessor: Record<string, unknown> | null,
): AxisResult {
  const step = ad.chainStep(doc);
  if (step.isGenesis) {
    return { axis: "chain", result: PASS, note: "genesis receipt (no predecessor link)", reasonCode: NO_REASON };
  }
  if (predecessor === null) {
    // Recomputation cannot complete without the predecessor; blocks PASS
    return {
      axis: "chain",
      result: UNVERIFIABLE,
      note: "no predecessor supplied",
      reasonCode: "chain_predecessor_missing",
    };
  }
  // A chain link must stay within one format; a cross-format predecessor is not a valid link.
  const predAd = detect(predecessor, adapters);
  if (predAd === null || predAd.name !== ad.name) {
    return {
      axis: "chain",
      result: INVALID,
      note: "predecessor is a different receipt format",
      reasonCode: "chain_mismatch",
    };
  }
  let actual: string;
  try {
    actual = step.recompute(predecessor);
  } catch (exc) {
    return {
      axis: "chain",
      result: UNVERIFIABLE,
      note: `predecessor not canonicalisable: ${(exc as Error).message}`,
      reasonCode: "canonicalization_failed",
    };
  }
  if (actual === step.prevField) {
    return { axis: "chain", result: PASS, note: "chain link rederives from predecessor", reasonCode: NO_REASON };
  }
  const exp = String(step.prevField).slice(0, 16);
  return {
    axis: "chain",
    result: INVALID,
    note: `chain break: expected ${exp}.. got ${actual.slice(0, 16)}..`,
    reasonCode: "chain_mismatch",
  };
}

/** Max nesting the recursive JCS encoder tolerates, mirrors the Python core cap. */
export const MAX_NESTING_DEPTH = 200;

const TOO_DEEP_NOTE = `receipt nesting exceeds the supported depth (> ${MAX_NESTING_DEPTH} levels)`;

/**
 * True when `obj` nests deeper than `maxDepth`, walked with an explicit stack.
 *
 * No recursion here, so the check never overflows before it can cap a receipt the
 * JCS canonicaliser would crash on. `JSON.parse` applies no depth limit, so this
 * gate is the only cap an already-parsed object meets.
 */
function exceedsDepth(obj: unknown, maxDepth: number): boolean {
  const stack: Array<[unknown, number]> = [[obj, 0]];
  while (stack.length > 0) {
    const [cur, depth] = stack.pop() as [unknown, number];
    if (depth > maxDepth) return true;
    if (Array.isArray(cur)) {
      for (const v of cur) stack.push([v, depth + 1]);
    } else if (cur !== null && typeof cur === "object") {
      for (const v of Object.values(cur as Record<string, unknown>)) stack.push([v, depth + 1]);
    }
  }
  return false;
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
    // No format claims this receipt, so no check can even start: the two
    // failure classes stay distinct - this is the unverifiable one
    return {
      fmt: "unknown",
      axes: [{
        axis: "structure",
        result: UNVERIFIABLE,
        note: "no adapter recognises this receipt",
        reasonCode: "format_unrecognized",
      }],
      verdict: "UNVERIFIABLE",
      classification: "unverifiable",
      signer: null,
    };
  }
  // An over-nested receipt would crash the recursive JCS encoder. Cap it here and
  // report UNVERIFIABLE, never a PASS, matching the Python core's shape gate.
  if (
    exceedsDepth(doc, MAX_NESTING_DEPTH) ||
    (predecessor !== null && exceedsDepth(predecessor, MAX_NESTING_DEPTH))
  ) {
    return {
      fmt: ad.name,
      axes: [{
        axis: "structure",
        result: UNVERIFIABLE,
        note: TOO_DEEP_NOTE,
        reasonCode: "canonicalization_failed",
      }],
      verdict: "UNVERIFIABLE",
      classification: "unverifiable",
      signer: null,
    };
  }

  const [structResult, structNote, structReason] = ad.schema(doc);
  const axes: AxisResult[] = [
    { axis: "structure", result: structResult, note: structNote, reasonCode: structReason },
    signatureAxis(ad, doc, keyProvider),
    chainAxis(ad, doc, adapters, predecessor),
  ];
  for (const [name, res, note, reason] of ad.extraAxes(doc, keyProvider)) {
    axes.push({ axis: name, result: res, note, reasonCode: reason });
  }

  // Expiry reports on its own axis and never folds the verdict (criterion 426);
  // INVALID dominates UNVERIFIABLE and either one blocks PASS (criterion 418).
  // SKIPPED axes do not apply and never block; UNVERIFIABLE ones always do.
  const verdict = deriveVerdict(axes.map((a) => [a.axis, a.result] as const));
  const signerVal = ad.attestation(doc).signer;
  const signer = typeof signerVal === "string" ? signerVal : null;
  return { fmt: ad.name, axes, verdict, classification: CLASSIFICATION[verdict], signer };
}
