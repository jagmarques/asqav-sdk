/** Retain live oracle assessments, TS half (488 C16). */

import { AssessmentStore, type AssessmentEntry } from "./assessmentStore.js";

export interface AxisOutcome {
  readonly axis: string;
  readonly result: string;
}

export interface VerifyResultLike {
  readonly axes: AxisOutcome[];
  readonly verdict: string;
  readonly failureClass: string | null;
  readonly context?: unknown;
}

const CONTEXT_FIELDS = [
  "evaluator",
  "trustRoots",
  "revocationSnapshot",
  "policyRevision",
  "evidenceKnowledgeCutoff",
  "evaluationTime",
] as const;

export function contextDict(result: VerifyResultLike): Record<string, unknown> {
  const ctx = (result.context ?? {}) as Record<string, unknown>;
  return Object.fromEntries(CONTEXT_FIELDS.map((name) => [name, ctx[name] ?? null]));
}

export function axesDict(result: VerifyResultLike): Record<string, string> {
  return Object.fromEntries(result.axes.map((a) => [a.axis, a.result]));
}

export function recordVerifyResult(
  store: AssessmentStore,
  docSha256: string,
  result: VerifyResultLike,
  recordedAt: string | null = null,
): AssessmentEntry {
  return store.record(
    docSha256,
    contextDict(result),
    axesDict(result),
    result.verdict,
    result.failureClass,
    recordedAt,
  );
}
