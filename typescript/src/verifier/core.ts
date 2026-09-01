/**
 * The shared verification core - format detection, dispatch and the verdict, a port of the Python
 * oracle's `core.py`. It proves only what the bytes prove, never the behaviour of the recorded action.
 */

import type { FormatAdapter, KeyProvider } from "./adapter.js";
import { FAIL, PASS, SKIPPED, verifySignature, type VerifyState } from "./crypto.js";

/**
 * Public verdict vocabulary (criteria 418/438). The per-axis PASS/FAIL/SKIPPED
 * tokens stay internal; the surface a caller reads speaks these three only.
 */
export const VERDICT_VERIFIED = "verified";
export const VERDICT_VERIFIED_KEYED = "verified_keyed";
export const VERDICT_UNVERIFIED = "unverified";
export type Verdict = "verified" | "verified_keyed" | "unverified";

/**
 * Failure classes carried by every unverified verdict (criterion 418); the two
 * are never collapsed - a proven binding failure is not an incomplete check.
 */
export const FAILURE_INVALID = "invalid";
export const FAILURE_UNVERIFIABLE = "unverifiable";
export type FailureClass = "invalid" | "unverifiable";

/** One verification axis outcome. */
export interface AxisResult {
  /** Which check (structure / signature / chain / extra). */
  axis: string;
  /** PASS / FAIL / SKIPPED (internal token). */
  result: VerifyState;
  /** Human-readable detail for the report. */
  note: string;
  /** invalid / unverifiable for a non-PASS axis, null on PASS. */
  failureClass: FailureClass | null;
}

/** The aggregate outcome of verifying one receipt. */
export interface VerifyResult {
  /** The matched adapter name, or "unknown". */
  fmt: string;
  axes: AxisResult[];
  /**
   * `verified` only when every non-skipped axis passed AND the signature was checked; `verified_keyed`
   * when the digest is keyed. Defaults fail closed: an unfolded result reads unverified.
   */
  verdict: Verdict;
  /** invalid / unverifiable when unverified, else null. */
  failureClass: FailureClass | null;
  // In-body origin attestation (v:2 signer) from the signed payload.
  // null for v:1. Never gates the verdict.
  signer: string | null;
  /**
   * Earliest axis in report order that did not PASS, else null. Reported so two verifiers
   * disagreeing about WHICH check failed first is as visible as disagreeing about the verdict.
   */
  firstFailingEdge: string | null;
}

/** Axes whose FAIL proves a cryptographic/policy binding failure (invalid). */
const INVALID_FAIL_AXES = new Set([
  "signature",
  "anchors",
  "issuer_bind",
  "key_status",
  "key_binding",
  "counterparty",
  "payload_digest",
  "nonce",
  "parent_signature",
  "pdp_signature",
  // A gap proves receipts were withheld and a malformed counter was signed as-is;
  // both are proven defects, not a recompute that could not finish.
  "seq",
]);

/**
 * Map one axis outcome to its failure class (criterion 418). A FAIL is invalid when a binding was proven
 * broken, unverifiable when the recompute could not finish; an unlisted FAIL reads unverifiable.
 */
export function axisFailureClass(axis: string, result: VerifyState, note: string): FailureClass | null {
  if (result === PASS) return null;
  if (result === SKIPPED) return FAILURE_UNVERIFIABLE;
  if (INVALID_FAIL_AXES.has(axis)) return FAILURE_INVALID;
  if (axis === "chain") {
    // A mismatched link or a cross-format predecessor is a proven break; a
    // predecessor the canonicaliser cannot walk stops the recompute instead.
    if (note.startsWith("chain break:") || note === "predecessor is a different receipt format") {
      return FAILURE_INVALID;
    }
    return FAILURE_UNVERIFIABLE;
  }
  if (axis === "skew") {
    if (note.startsWith("unparseable issued_at")) return FAILURE_UNVERIFIABLE;
    return FAILURE_INVALID;
  }
  if (axis === "structure") {
    if (note.startsWith("unsupported ACTA alg") || note.startsWith("unsupported signature algorithm")) {
      return FAILURE_INVALID;
    }
    if (note.startsWith("key_purpose mismatch")) return FAILURE_INVALID;
    if (note.includes("signing-key DID != issuer DID")) return FAILURE_INVALID;
    if (note.startsWith("proof @context is not a prefix")) return FAILURE_INVALID;
    return FAILURE_UNVERIFIABLE;
  }
  if (axis === "expiry") {
    if (note.startsWith("unreadable expires_at")) return FAILURE_UNVERIFIABLE;
    return FAILURE_INVALID;
  }
  // issuer_key / ingest and anything unlisted: the recompute could not complete.
  return FAILURE_UNVERIFIABLE;
}

function axis(axis: string, result: VerifyState, note: string): AxisResult {
  return { axis, result, note, failureClass: axisFailureClass(axis, result, note) };
}

/**
 * Fixed leading axis order every adapter walks, before its format-specific extras.
 * Pinned so a reorder fails a gate instead of quietly renaming which edge is "first".
 */
export const AXIS_ORDER_PREFIX = ["structure", "signature", "chain", "seq"] as const;

/**
 * Name the earliest axis in report order that drove the verdict away from verified.
 * Exclusions mirror `foldVerdict` (expiry never folds; SKIPPED chain is tolerated, others are not).
 */
export function firstFailingEdge(axes: AxisResult[]): string | null {
  for (const a of axes) {
    if (a.result === FAIL && a.axis !== "expiry") return a.axis;
    if (a.result === SKIPPED && a.axis !== "chain") return a.axis;
  }
  return null;
}

/**
 * Fold per-axis outcomes into the public verdict + failure class (418/438).
 * A port of the Python `fold_verdict`.
 */
export function foldVerdict(
  axes: AxisResult[],
  keyed: boolean,
): readonly [Verdict, FailureClass | null] {
  const failed = axes.filter((a) => a.result === FAIL && a.axis !== "expiry");
  const blockingSkip = axes.some((a) => a.result === SKIPPED && a.axis !== "chain");
  if (failed.length > 0) {
    // A proven binding failure dominates a malformed-member failure: the
    // receipt is invalid on the strongest ground the axes established.
    const failureClass = failed.some((a) => a.failureClass === FAILURE_INVALID)
      ? FAILURE_INVALID
      : FAILURE_UNVERIFIABLE;
    return [VERDICT_UNVERIFIED, failureClass];
  }
  if (blockingSkip) return [VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE];
  if (keyed) return [VERDICT_VERIFIED_KEYED, null];
  return [VERDICT_VERIFIED, null];
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
  const [pk, note] = ad.resolveKey(doc, keyProvider);
  if (pk === null) {
    return axis("signature", SKIPPED, `no key: ${note}`);
  }
  const msg = ad.signingInput(doc);
  const { result, note: why } = verifySignature(sm.alg, pk, msg, sm.sig);
  return axis("signature", result, why);
}

function chainAxis(
  ad: FormatAdapter,
  doc: Record<string, unknown>,
  adapters: FormatAdapter[],
  predecessor: Record<string, unknown> | null,
): AxisResult {
  const step = ad.chainStep(doc);
  if (step.isGenesis) {
    return axis("chain", PASS, "genesis receipt (no predecessor link)");
  }
  if (predecessor === null) {
    return axis("chain", SKIPPED, "no predecessor supplied");
  }
  // A chain link must stay within one format; a cross-format predecessor is not a valid link.
  const predAd = detect(predecessor, adapters);
  if (predAd === null || predAd.name !== ad.name) {
    return axis("chain", FAIL, "predecessor is a different receipt format");
  }
  const actual = step.recompute(predecessor);
  if (actual === step.prevField) {
    return axis("chain", PASS, "chain link rederives from predecessor");
  }
  const exp = String(step.prevField).slice(0, 16);
  return axis("chain", FAIL, `chain break: expected ${exp}.. got ${actual.slice(0, 16)}..`);
}

// A gap is omission evidence. Never SKIPPED: foldVerdict blocks on a non-chain
// SKIPPED, so a receipt with no counter would regress to unverified.
function seqAxis(
  ad: FormatAdapter,
  doc: Record<string, unknown>,
  adapters: FormatAdapter[],
  predecessor: Record<string, unknown> | null,
): AxisResult {
  const seq = ad.seqOf(doc);
  if (seq === null || seq === undefined) {
    return axis("seq", PASS, "no seq member; receipt is not part of a counted series");
  }
  if (!isCounter(seq)) return axis("seq", FAIL, `malformed seq: ${JSON.stringify(seq)}`);
  if (predecessor === null) return axis("seq", PASS, `seq ${seq}; no predecessor supplied`);
  // A counter only means anything within one format's own series.
  const predAd = detect(predecessor, adapters);
  if (predAd === null || predAd.name !== ad.name) {
    return axis("seq", PASS, `seq ${seq}; predecessor is a different receipt format`);
  }
  const prev = ad.seqOf(predecessor);
  if (prev === null || prev === undefined) {
    return axis("seq", PASS, `seq ${seq}; predecessor carries no seq`);
  }
  if (!isCounter(prev)) {
    return axis("seq", FAIL, `malformed predecessor seq: ${JSON.stringify(prev)}`);
  }
  if (seq === prev + 1) return axis("seq", PASS, `seq ${seq} follows predecessor ${prev}`);
  if (seq <= prev) {
    return axis("seq", FAIL, `seq not monotonic: ${seq} after predecessor ${prev}`);
  }
  return axis("seq", FAIL, `seq gap: ${seq - prev - 1} receipt(s) withheld between ${prev} and ${seq}`);
}

// A counter is a positive whole number; booleans and 1.5 are not counters.
function isCounter(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v) && v >= 1;
}

/** Max nesting the recursive JCS encoder tolerates, mirrors the Python core cap. */
export const MAX_NESTING_DEPTH = 200;

const TOO_DEEP_NOTE = `receipt nesting exceeds the supported depth (> ${MAX_NESTING_DEPTH} levels)`;

/**
 * True when `obj` nests deeper than `maxDepth`, walked with an explicit stack so the check itself never
 * overflows. `JSON.parse` applies no depth limit, so this is the only cap a parsed object meets.
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
    const axes = [axis("structure", FAIL, "no adapter recognises this receipt")];
    const [verdict, failureClass] = foldVerdict(axes, false);
    return {
      fmt: "unknown",
      axes,
      verdict,
      failureClass,
      signer: null,
      firstFailingEdge: firstFailingEdge(axes),
    };
  }
  // An over-nested receipt would crash the recursive JCS encoder. Cap it here and
  // report unverified/unverifiable, never a verified, matching the Python gate.
  if (
    exceedsDepth(doc, MAX_NESTING_DEPTH) ||
    (predecessor !== null && exceedsDepth(predecessor, MAX_NESTING_DEPTH))
  ) {
    const axes = [axis("structure", FAIL, TOO_DEEP_NOTE)];
    const [verdict, failureClass] = foldVerdict(axes, false);
    return { fmt: ad.name, axes, verdict, failureClass, signer: null, firstFailingEdge: firstFailingEdge(axes) };
  }

  const [structResult, structNote] = ad.schema(doc);
  const axes: AxisResult[] = [
    axis("structure", structResult, structNote),
    signatureAxis(ad, doc, keyProvider),
    chainAxis(ad, doc, adapters, predecessor),
    seqAxis(ad, doc, adapters, predecessor),
  ];
  for (const [name, res, note] of ad.extraAxes(doc, keyProvider)) {
    axes.push(axis(name, res, note));
  }

  // Expiry reports on its own axis and never folds the verdict (criterion 426);
  // a keyed digest reports verified_keyed, never plain verified (criterion 438).
  const [verdict, failureClass] = foldVerdict(axes, ad.keyedDigest(doc));
  const signerVal = ad.attestation(doc).signer;
  const signer = typeof signerVal === "string" ? signerVal : null;
  return { fmt: ad.name, axes, verdict, failureClass, signer, firstFailingEdge: firstFailingEdge(axes) };
}
