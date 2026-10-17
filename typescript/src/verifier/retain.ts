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
  "eventTime",
  "effectiveTime",
] as const;

function freeze(value: unknown): unknown {
  if (value instanceof Date) return value.toISOString();
  if (Array.isArray(value)) return value.map(freeze);
  if (value !== null && typeof value === "object") {
    const frozen: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      frozen[k] = freeze(v);
    }
    return frozen;
  }
  return value;
}

export function contextDict(result: VerifyResultLike): Record<string, unknown> {
  const ctx = (result.context ?? {}) as Record<string, unknown>;
  return Object.fromEntries(CONTEXT_FIELDS.map((name) => [name, freeze(ctx[name]) ?? null]));
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

export function reassessVerifyResult(
  store: AssessmentStore,
  docSha256: string,
  result: VerifyResultLike,
  recordedAt: string | null = null,
  reason: string | null = null,
): AssessmentEntry {
  return store.reassess(
    docSha256,
    contextDict(result),
    axesDict(result),
    result.verdict,
    result.failureClass,
    recordedAt,
    reason,
  );
}
