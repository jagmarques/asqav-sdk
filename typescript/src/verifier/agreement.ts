/** Agreement reference and comparanda comparison (488 C12). */

import { createHash } from "node:crypto";

export interface AgreementRef {
  readonly agreementId: string | null;
  readonly predicateVersion: string | null;
  readonly termsHash: string | null;
}

export function agreementOfTerms(
  agreementId: string,
  predicateVersion: string,
  terms: unknown,
): AgreementRef {
  const termsHash = createHash("sha256").update(canonical(terms), "utf-8").digest("hex");
  return { agreementId, predicateVersion, termsHash };
}

export const COMPARANDA = [
  "evidenceBytes",
  "agreementVersion",
  "contentHashes",
  "evaluator",
  "trustRoots",
  "revocationSnapshot",
  "verificationTime",
] as const;

export type Comparandum = (typeof COMPARANDA)[number];

export interface AssessmentView {
  readonly docSha256: string;
  readonly axes: Record<string, string>;
  readonly context: Record<string, unknown>;
  readonly agreement?: AgreementRef | Record<string, string | null> | null;
}

function canonical(value: unknown): string {
  return JSON.stringify(sortDeep(value));
}

function sortDeep(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortDeep);
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(value).sort()) {
      out[k] = sortDeep((value as Record<string, unknown>)[k]);
    }
    return out;
  }
  return value;
}

type RawAgreement = {
  agreementId?: string | null;
  agreement_id?: string | null;
  predicateVersion?: string | null;
  predicate_version?: string | null;
  termsHash?: string | null;
  terms_hash?: string | null;
};

function normAgreement(view: AssessmentView): AgreementRef {
  const raw = (view.agreement ?? {}) as RawAgreement;
  return {
    agreementId: raw.agreementId ?? raw.agreement_id ?? null,
    predicateVersion: raw.predicateVersion ?? raw.predicate_version ?? null,
    termsHash: raw.termsHash ?? raw.terms_hash ?? null,
  };
}

function hashJson(value: unknown): string {
  return createHash("sha256").update(canonical(value), "utf-8").digest("hex");
}

function ctxGet(ctx: Record<string, unknown>, camel: string, snake: string): unknown {
  return ctx[camel] ?? ctx[snake];
}

export function compareAssessments(
  first: AssessmentView,
  second: AssessmentView,
): Record<Comparandum, boolean> {
  const fRoots = [...(((ctxGet(first.context, "trustRoots", "trust_roots") as string[] | undefined) ?? []))].sort();
  const sRoots = [...(((ctxGet(second.context, "trustRoots", "trust_roots") as string[] | undefined) ?? []))].sort();
  const fAgr = normAgreement(first);
  const sAgr = normAgreement(second);
  return {
    evidenceBytes: first.docSha256 === second.docSha256,
    agreementVersion:
      fAgr.agreementId === sAgr.agreementId &&
      fAgr.predicateVersion === sAgr.predicateVersion &&
      fAgr.termsHash === sAgr.termsHash,
    contentHashes: hashJson(first.axes ?? {}) === hashJson(second.axes ?? {}),
    evaluator: ctxGet(first.context, "evaluator", "evaluator") === ctxGet(second.context, "evaluator", "evaluator"),
    trustRoots: JSON.stringify(fRoots) === JSON.stringify(sRoots),
    revocationSnapshot:
      ctxGet(first.context, "revocationSnapshot", "revocation_snapshot") ===
      ctxGet(second.context, "revocationSnapshot", "revocation_snapshot"),
    verificationTime:
      ctxGet(first.context, "evaluationTime", "evaluation_time") ===
      ctxGet(second.context, "evaluationTime", "evaluation_time"),
  };
}
