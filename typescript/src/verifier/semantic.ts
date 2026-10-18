/** Independent verdict semantics (oracle-independence slice) */
/** Rules: invalid-vs-unverifiable, keyed, expiry, chain skip */
import { FAIL, PASS, SKIPPED, type VerifyState } from "./crypto.js";
import type { AxisResult, FailureClass, Verdict } from "./core.js";

/** Vocabulary literals; pinned equal to the core's surface */
const VERDICT_VERIFIED: Verdict = "verified";
const VERDICT_VERIFIED_KEYED: Verdict = "verified_keyed";
const VERDICT_UNVERIFIED: Verdict = "unverified";
const FAILURE_INVALID: FailureClass = "invalid";
const FAILURE_UNVERIFIABLE: FailureClass = "unverifiable";

/** Axes whose FAIL proves a binding failed (418 invalid limb). */
const ALWAYS_INVALID_AXES: ReadonlySet<string> = new Set([
  "signature",
  "anchors",
  "issuer_bind",
  "key_status",
  "key_binding",
  "counterparty",
  "payload_digest",
  "outer_copies",
  "nonce",
  "parent_signature",
  "pdp_signature",
  "seq",
]);

/** Axes that report but never fold the verdict (expiry-never-folds rule). */
const NON_FOLDING_AXES: ReadonlySet<string> = new Set(["expiry"]);
/** A skipped check blocks, except the chain axis. */
const TOLERATED_SKIP_AXES: ReadonlySet<string> = new Set(["chain"]);

/** Classify one axis outcome; table axes prove a binding failure */
/** Note branches split proven breaks from stops; unlisted never proves */
export function classifyFailure(axis: string, result: VerifyState, note: string): FailureClass | null {
  if (result === PASS) return null;
  if (result === SKIPPED) return FAILURE_UNVERIFIABLE;
  if (ALWAYS_INVALID_AXES.has(axis)) return FAILURE_INVALID;
  if (axis === "chain") {
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
    if (note.startsWith("unregistered type ") || note.startsWith("malformed type ")) {
      return FAILURE_INVALID;
    }
    return FAILURE_UNVERIFIABLE;
  }
  if (axis === "expiry") {
    if (note.startsWith("unreadable expires_at")) return FAILURE_UNVERIFIABLE;
    return FAILURE_INVALID;
  }
  return FAILURE_UNVERIFIABLE;
}

type FoldKey = `${boolean} ${boolean} ${boolean} ${boolean}`;

/** Verdict lookup keyed by four booleans; keyed never rescues */
const TABLE: Record<FoldKey, readonly [Verdict, FailureClass | null]> = {
  "false false false false": [VERDICT_VERIFIED, null],
  "false false false true": [VERDICT_VERIFIED_KEYED, null],
  "false false true false": [VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE],
  "false false true true": [VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE],
  "false true false false": [VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE],
  "false true false true": [VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE],
  "false true true false": [VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE],
  "false true true true": [VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE],
  "true false false false": [VERDICT_UNVERIFIED, FAILURE_INVALID],
  "true false false true": [VERDICT_UNVERIFIED, FAILURE_INVALID],
  "true false true false": [VERDICT_UNVERIFIED, FAILURE_INVALID],
  "true false true true": [VERDICT_UNVERIFIED, FAILURE_INVALID],
  "true true false false": [VERDICT_UNVERIFIED, FAILURE_INVALID],
  "true true false true": [VERDICT_UNVERIFIED, FAILURE_INVALID],
  "true true true false": [VERDICT_UNVERIFIED, FAILURE_INVALID],
  "true true true true": [VERDICT_UNVERIFIED, FAILURE_INVALID],
};

/** Fold per-axis outcomes by table lookup; invalid dominates */
export function foldIndependent(axes: AxisResult[], keyed: boolean): readonly [Verdict, FailureClass | null] {
  let invalid = false;
  let other = false;
  for (const a of axes) {
    if (NON_FOLDING_AXES.has(a.axis)) continue;
    if (a.result === FAIL) {
      if (a.failureClass === FAILURE_INVALID) invalid = true;
      else other = true;
    }
  }
  const blocked = axes.some((a) => a.result === SKIPPED && !TOLERATED_SKIP_AXES.has(a.axis));
  return TABLE[`${invalid} ${other} ${blocked} ${keyed}`];
}
