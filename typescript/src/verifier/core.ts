/** Shared core: detection, dispatch and the verdict */
/** Verdict rules live in semantic.ts; proves only what the bytes prove */

import type { FormatAdapter, KeyProvider, VerificationContext } from "./adapter.js";
import { exceedsDepth } from "./canonical.js";
import { FAIL, PASS, SKIPPED, verifySignature, type VerifyState } from "./crypto.js";
import { REGISTRY_EDITION, REGISTRY_VERSION } from "./vrShim.js";
import { notCheckedDeclaration, coverageDeclaration, type Coverage, type NotCheckedEntry } from "./not-checked.js";
import { classifyFailure, foldIndependent } from "./semantic.js";

// The surface a caller reads speaks these three only
/** Public verdict vocabulary; per-axis PASS/FAIL/SKIPPED stay internal */
export const VERDICT_VERIFIED = "verified";
export const VERDICT_VERIFIED_KEYED = "verified_keyed";
export const VERDICT_UNVERIFIED = "unverified";
export type Verdict = "verified" | "verified_keyed" | "unverified";

// Never collapsed: a proven binding failure is not an incomplete check
/** Failure classes carried by every unverified verdict */
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
/** Versioned trust identity, mirrored from Python's trust_bundle. */
export interface TrustBundleIdentity {
  /** Bundle version the result was verified under. */
  version: number | string | null;
  /** Recorded bundle digest the run verified against. */
  digest: string | null;
  /** Where the bundle came from (e.g. bundle file name). */
  source: string | null;
  /** Whether the bundle self-digest verified. */
  valid: boolean;
}

export interface VerifyResult {
  /** The matched adapter name, or "unknown". */
  fmt: string;
  axes: AxisResult[];
  // `verified_keyed` when the digest is keyed; defaults fail closed
  /** `verified` needs every non-skipped axis PASS and a checked signature */
  verdict: Verdict;
  /** invalid / unverifiable when unverified, else null. */
  failureClass: FailureClass | null;
  // In-body origin attestation (v:2 signer) from the signed payload
  // null for v:1, never gates the verdict
  signer: string | null;
  // Makes two verifiers disagreeing on WHICH check failed first as visible as
  // disagreeing on the verdict
  /** Earliest axis in report order that did not PASS, else null */
  firstFailingEdge: string | null;
  /** The checks this result does not claim, beside the claim it makes. */
  notChecked: NotCheckedEntry[];
  /** Trust identity; null here, TS evaluates no bundles. Parity field. */
  trustBundle: TrustBundleIdentity | null;
  /** Registry edition + upstream version this result resolved against (625). */
  registryEdition: string;
  registryVersion: string;
  /** Coverage block mirrored from the reviewer's tool, snake_case on purpose */
  coverage: Coverage;
  /** Resolved invocation context, so a disagreement attributes to an input. */
  context?: VerificationContext;
}

/** Classify one axis outcome; rules live in semantic.ts */
/** Kept as the public name for existing importers */
export function axisFailureClass(axis: string, result: VerifyState, note: string): FailureClass | null {
  return classifyFailure(axis, result, note);
}

function axis(axis: string, result: VerifyState, note: string): AxisResult {
  return { axis, result, note, failureClass: axisFailureClass(axis, result, note) };
}

// Absent and [] are conformant; null is malformed, reported unverifiable
// Verbatim in all three engines; drift fails cross-language anchor tests
/** Structure-axis note for the malformed null spelling of "no anchors" */
const ANCHORS_NULL_NOTE =
  "anchors is null: malformed; absent or [] is the conformant spelling of no anchors";

// anchors sits beside the signed payload, so no adapter schema reads it;
// it is caught here, on the document as parsed
/** The structure axis: the adapter schema plus one envelope-member rule */
function structureAxis(ad: FormatAdapter, doc: Record<string, unknown>): AxisResult {
  const [res, note] = ad.schema(doc);
  if (res === PASS && "anchors" in doc && doc.anchors === null) {
    return axis("structure", FAIL, ANCHORS_NULL_NOTE);
  }
  return axis("structure", res, note);
}

// Pinned so a reorder fails a gate instead of quietly renaming "first"
/** Fixed leading axis order every adapter walks, before its format extras */
export const AXIS_ORDER_PREFIX = ["structure", "signature", "chain", "seq"] as const;

/** Earliest axis in report order that drove the verdict away */
/** Exclusions mirror the semantic table: expiry never folds, chain tolerated */
export function firstFailingEdge(axes: AxisResult[]): string | null {
  for (const a of axes) {
    if (a.result === FAIL && a.axis !== "expiry") return a.axis;
    if (a.result === SKIPPED && a.axis !== "chain") return a.axis;
  }
  return null;
}

/** Fold per-axis outcomes; rules live in semantic.ts */
/** Kept as the public name for existing importers */
export function foldVerdict(
  axes: AxisResult[],
  keyed: boolean,
): readonly [Verdict, FailureClass | null] {
  return foldIndependent(axes, keyed);
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
  if (result !== PASS) {
    const pre = ad.preCutoverSigningInput(doc);
    if (pre !== null && verifySignature(sm.alg, pk, pre, sm.sig).result === PASS) {
      return axis(
        "signature",
        FAIL,
        "legacy dialect: the signature verifies only under code-point member order, " +
          "which is not RFC 8785; reported unverified, never verified",
      );
    }
  }
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
  // A chain link must stay within one format; cross-format is not a link
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

// A gap is omission evidence. Never SKIPPED: the semantic table blocks on a
// non-chain SKIPPED, so a receipt with no counter regresses to unverified
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
  // A counter only means anything within one format's own series
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

// A counter is a positive whole number; booleans and 1.5 are not counters
function isCounter(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v) && v >= 1;
}

/** Max nesting the recursive JCS encoder tolerates, as the Python core cap */
export const MAX_NESTING_DEPTH = 200;

const TOO_DEEP_NOTE = `receipt nesting exceeds the supported depth (> ${MAX_NESTING_DEPTH} levels)`;

/** One-axis unverified result for a pre-axis refusal (depth, profile range) */
function earlyUnverified(fmt: string, note: string): VerifyResult {
  const axes = [axis("structure", FAIL, note)];
  const [verdict, failureClass] = foldVerdict(axes, false);
  return { fmt, axes, verdict, failureClass, signer: null, firstFailingEdge: firstFailingEdge(axes), notChecked: notCheckedDeclaration(), coverage: coverageDeclaration(axes), trustBundle: null, registryEdition: REGISTRY_EDITION, registryVersion: REGISTRY_VERSION };
}

/** Unverified/unverifiable for input refused before hashing (runner form) */
export function refuseInput(note: string, context: VerificationContext = {}, fmt = "unknown"): VerifyResult {
  return { ...earlyUnverified(fmt, note), context };
}

/** Verify one parsed receipt and return a structured `VerifyResult`. */
export function verify(
  doc: Record<string, unknown>,
  adapters: FormatAdapter[],
  keyProvider: KeyProvider = null,
  predecessor: Record<string, unknown> | null = null,
  context: VerificationContext = {},
): VerifyResult {
  const ad = detect(doc, adapters);
  const resolved: VerificationContext = context ?? {};
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
      notChecked: notCheckedDeclaration(),
      coverage: coverageDeclaration(axes),
      trustBundle: null,
      registryEdition: REGISTRY_EDITION,
      registryVersion: REGISTRY_VERSION,
      context: resolved,
    };
  }
  // An over-nested receipt would crash the recursive JCS encoder; cap it here
  // and report unverified/unverifiable, never verified, like the Python gate
  let earlyNote: string | null = null;
  if (
    exceedsDepth(doc, MAX_NESTING_DEPTH) ||
    (predecessor !== null && exceedsDepth(predecessor, MAX_NESTING_DEPTH))
  ) {
    earlyNote = TOO_DEEP_NOTE;
  } else {
    const predAd = predecessor === null ? null : detect(predecessor, adapters);
    earlyNote = ad.profilePrecheck(
      doc,
      predecessor,
      predAd === null ? null : predAd.name,
      resolved.originatingEnvelope ?? null,
    );
  }
  if (earlyNote !== null) {
    const early = earlyUnverified(ad.name, earlyNote);
    return {
      fmt: early.fmt,
      axes: early.axes,
      verdict: early.verdict,
      failureClass: early.failureClass,
      signer: early.signer,
      firstFailingEdge: early.firstFailingEdge,
      notChecked: early.notChecked,
      coverage: early.coverage,
      trustBundle: null,
      registryEdition: REGISTRY_EDITION,
      registryVersion: REGISTRY_VERSION,
      context: resolved,
    };
  }

  const axes: AxisResult[] = [
    structureAxis(ad, doc),
    signatureAxis(ad, doc, keyProvider),
    chainAxis(ad, doc, adapters, predecessor),
    seqAxis(ad, doc, adapters, predecessor),
  ];
  for (const [name, res, note] of ad.extraAxesWithContext(doc, keyProvider, context)) {
    axes.push(axis(name, res, note));
  }

  // Expiry reports on its own axis and never folds the verdict;
  // a keyed digest reports verified_keyed, never plain verified
  const [verdict, failureClass] = foldVerdict(axes, ad.keyedDigest(doc));
  const signerVal = ad.attestation(doc).signer;
  const signer = typeof signerVal === "string" ? signerVal : null;
  return { fmt: ad.name, axes, verdict, failureClass, signer, firstFailingEdge: firstFailingEdge(axes), notChecked: notCheckedDeclaration(), coverage: coverageDeclaration(axes), trustBundle: null, registryEdition: REGISTRY_EDITION, registryVersion: REGISTRY_VERSION, context: resolved };
}
