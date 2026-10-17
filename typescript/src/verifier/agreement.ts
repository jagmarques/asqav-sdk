/** Agreement comparanda (488 C12, C29: ref carries bytes-derived envelope). */

import { createHash } from "node:crypto";

import type { FormatAdapter } from "./adapter.js";
import { detect } from "./core.js";
import { ADAPTERS } from "./index.js";

export interface AgreementRef {
  readonly agreementId: string | null;
  readonly predicateVersion: string | null;
  readonly termsHash: string | null;
  readonly kid: string | null;
  readonly signature: string | null;
}

export function agreementOfTerms(
  agreementId: string,
  predicateVersion: string,
  terms: unknown,
): AgreementRef {
  const termsHash = createHash("sha256").update(canonical(terms), "utf-8").digest("hex");
  return { agreementId, predicateVersion, termsHash, kid: null, signature: null };
}

/** Derive the verified envelope from a parsed receipt doc (488 C29). */
export function envelopeOf(
  doc: Record<string, unknown>,
  adapters: FormatAdapter[] = ADAPTERS,
): AgreementRef {
  const adapter = detect(doc, adapters);
  if (adapter === null) {
    throw new Error("no adapter detects this receipt; envelope unbound");
  }
  let material;
  try {
    material = adapter.extractSignature(doc);
  } catch (exc) {
    throw new Error(`envelope extraction failed: ${exc}`);
  }
  return {
    agreementId: null,
    predicateVersion: null,
    termsHash: null,
    kid: material.kid,
    signature: Buffer.from(material.sig).toString("hex"),
  };
}

/** Merge caller terms with the bytes-derived envelope (488 C29). */
export function bindEnvelope(
  agreement: AgreementRef | Record<string, string | null> | null,
  doc: Record<string, unknown>,
  adapters: FormatAdapter[] = ADAPTERS,
): AgreementRef {
  const derived = envelopeOf(doc, adapters);
  const stated = normAgreementValue(agreement);
  for (const field of ["kid", "signature"] as const) {
    if (stated[field] !== null && stated[field] !== derived[field]) {
      throw new Error(`stated ${field} contradicts the verified envelope`);
    }
  }
  return {
    agreementId: stated.agreementId,
    predicateVersion: stated.predicateVersion,
    termsHash: stated.termsHash,
    kid: derived.kid,
    signature: derived.signature,
  };
}

export const COMPARANDA = [
  "evidenceBytes",
  "agreementVersion",
  "contentHashes",
  "evaluator",
  "trustRoots",
  "revocationSnapshot",
  "verificationTime",
  // Interpretation inputs (488 C9): same facts under another policy
  // revision or knowledge cutoff attribute; they never read unexplained
  "policyRevision",
  "evidenceKnowledgeCutoff",
  // Event-side timing (488 C9): same bytes under another declared event
  // or effective time attribute; they never read unexplained
  "eventTime",
  "effectiveTime",
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
  kid?: string | null;
  signature?: string | null;
};

export function normAgreementValue(
  raw: AgreementRef | Record<string, string | null> | null | undefined,
): AgreementRef {
  // One lookup shape for refs and dicts: refs carry camelCase keys, so the
  // snake_case fallback never fires for them; dicts may carry either
  const r = (raw ?? {}) as RawAgreement;
  return {
    agreementId: r.agreementId ?? r.agreement_id ?? null,
    predicateVersion: r.predicateVersion ?? r.predicate_version ?? null,
    termsHash: r.termsHash ?? r.terms_hash ?? null,
    kid: r.kid ?? null,
    signature: r.signature ?? null,
  };
}

function normAgreement(view: AssessmentView): AgreementRef {
  return normAgreementValue(view.agreement);
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
      fAgr.termsHash === sAgr.termsHash &&
      fAgr.kid === sAgr.kid &&
      fAgr.signature === sAgr.signature,
    contentHashes: hashJson(first.axes ?? {}) === hashJson(second.axes ?? {}),
    evaluator: ctxGet(first.context, "evaluator", "evaluator") === ctxGet(second.context, "evaluator", "evaluator"),
    trustRoots: JSON.stringify(fRoots) === JSON.stringify(sRoots),
    revocationSnapshot:
      ctxGet(first.context, "revocationSnapshot", "revocation_snapshot") ===
      ctxGet(second.context, "revocationSnapshot", "revocation_snapshot"),
    verificationTime:
      ctxGet(first.context, "evaluationTime", "evaluation_time") ===
      ctxGet(second.context, "evaluationTime", "evaluation_time"),
    policyRevision:
      ctxGet(first.context, "policyRevision", "policy_revision") ===
      ctxGet(second.context, "policyRevision", "policy_revision"),
    evidenceKnowledgeCutoff:
      ctxGet(first.context, "evidenceKnowledgeCutoff", "evidence_knowledge_cutoff") ===
      ctxGet(second.context, "evidenceKnowledgeCutoff", "evidence_knowledge_cutoff"),
    eventTime:
      ctxGet(first.context, "eventTime", "event_time") ===
      ctxGet(second.context, "eventTime", "event_time"),
    effectiveTime:
      ctxGet(first.context, "effectiveTime", "effective_time") ===
      ctxGet(second.context, "effectiveTime", "effective_time"),
  };
}
