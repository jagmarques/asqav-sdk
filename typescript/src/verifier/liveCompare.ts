/** Compare two live VerifyResults, then classify (488 C19, C5 core). */

import { createHash } from "node:crypto";

import type { VerificationContext } from "./adapter.js";
import { type AgreementRef, type AssessmentView, bindEnvelope, normAgreementValue } from "./agreement.js";
import { AssessmentStore, type AssessmentEntry } from "./assessmentStore.js";
import { verify } from "./core.js";
import { classify, specCase, type Disagreement, type SpecCase } from "./disagreement.js";
import { ADAPTERS } from "./index.js";
import { axesDict, contextDict, type VerifyResultLike } from "./retain.js";

type LiveView = {
  docSha256: string;
  axes: Record<string, string>;
  context: Record<string, unknown>;
  agreement: AgreementRef | Record<string, string | null> | null;
};

function view(
  digest: string,
  result: VerifyResultLike,
  agreement: AgreementRef | Record<string, string | null> | null = null,
): LiveView {
  return { docSha256: digest, axes: axesDict(result), context: contextDict(result), agreement };
}

function hasStatedEnvelope(
  agreement: AgreementRef | Record<string, string | null> | null,
): boolean {
  if (agreement === null) return false;
  const stated = normAgreementValue(agreement);
  return stated.kid !== null || stated.signature !== null;
}

function boundAgreement(
  agreement: AgreementRef | Record<string, string | null> | null,
  raw: Uint8Array | null,
): AgreementRef | Record<string, string | null> | null {
  if (raw === null || agreement === null) return agreement;
  const doc = JSON.parse(new TextDecoder().decode(raw)) as Record<string, unknown>;
  return bindEnvelope(agreement, doc);
}

export function compareLive(
  firstDigest: string,
  first: VerifyResultLike,
  secondDigest: string,
  second: VerifyResultLike,
  firstAgreement: AgreementRef | Record<string, string | null> | null = null,
  secondAgreement: AgreementRef | Record<string, string | null> | null = null,
  firstBytes: Uint8Array | null = null,
  secondBytes: Uint8Array | null = null,
): Disagreement {
  const sides = [["first", firstDigest, firstBytes, firstAgreement], ["second", secondDigest, secondBytes, secondAgreement]] as const;
  for (const [tag, digest, raw, agr] of sides) {
    if (raw !== null && createHash("sha256").update(raw).digest("hex") !== digest) {
      throw new Error(`${tag} digest is not the sha256 of the stated bytes`);
    }
    if (raw === null && hasStatedEnvelope(agr)) {
      throw new Error(`${tag} envelope without bytes; pass the receipt bytes`);
    }
  }  const firstView = view(firstDigest, first, boundAgreement(firstAgreement, firstBytes));
  const secondView = view(secondDigest, second, boundAgreement(secondAgreement, secondBytes));
  return classify(firstView, secondView, first.verdict, second.verdict);
}

export function compareLiveStrict(
  firstBytes: Uint8Array,
  secondBytes: Uint8Array,
  firstAgreement: AgreementRef | Record<string, string | null> | null = null,
  secondAgreement: AgreementRef | Record<string, string | null> | null = null,
  context: VerificationContext = {},
): Disagreement {
  const text = new TextDecoder();
  const firstDoc = JSON.parse(text.decode(firstBytes)) as Record<string, unknown>;
  const secondDoc = JSON.parse(text.decode(secondBytes)) as Record<string, unknown>;
  const first = verify(firstDoc, ADAPTERS, null, null, context);
  const second = verify(secondDoc, ADAPTERS, null, null, context);
  const firstDigest = createHash("sha256").update(firstBytes).digest("hex");
  const secondDigest = createHash("sha256").update(secondBytes).digest("hex");
  const firstView = view(firstDigest, first, bindEnvelope(firstAgreement, firstDoc));
  const secondView = view(secondDigest, second, bindEnvelope(secondAgreement, secondDoc));
  return classify(firstView, secondView, first.verdict, second.verdict);
}

export function attributeSplit(
  firstView: AssessmentView,
  secondView: AssessmentView,
  verdictA: string,
  verdictB: string,
  rerunAView: AssessmentView | null = null,
  rerunBView: AssessmentView | null = null,
): { disagreement: Disagreement; specCase: SpecCase } {
  return {
    disagreement: classify(firstView, secondView, verdictA, verdictB),
    specCase: specCase(firstView, secondView, verdictA, verdictB, rerunAView, rerunBView),
  };
}

export function attributeLiveSplit(
  firstDigest: string,
  first: VerifyResultLike,
  secondDigest: string,
  second: VerifyResultLike,
  rerunA: VerifyResultLike | null = null,
  rerunB: VerifyResultLike | null = null,
  firstAgreement: AgreementRef | Record<string, string | null> | null = null,
  secondAgreement: AgreementRef | Record<string, string | null> | null = null,
  firstBytes: Uint8Array | null = null,
  secondBytes: Uint8Array | null = null,
  rerunABytes: Uint8Array | null = null,
  rerunBBytes: Uint8Array | null = null,
): { disagreement: Disagreement; specCase: SpecCase } {
  const sides = [["first", firstDigest, firstBytes, firstAgreement, true], ["second", secondDigest, secondBytes, secondAgreement, true], ["rerunA", firstDigest, rerunABytes, firstAgreement, rerunA !== null], ["rerunB", secondDigest, rerunBBytes, secondAgreement, rerunB !== null]] as const;
  for (const [tag, digest, raw, agr, present] of sides) {
    if (raw !== null && createHash("sha256").update(raw).digest("hex") !== digest) {
      throw new Error(`${tag} digest is not the sha256 of the stated bytes`);
    }
    if (present && raw === null && hasStatedEnvelope(agr)) {
      throw new Error(`${tag} envelope without bytes; pass the receipt bytes`);
    }
  }  const firstView = view(firstDigest, first, boundAgreement(firstAgreement, firstBytes));
  const secondView = view(secondDigest, second, boundAgreement(secondAgreement, secondBytes));
  const rerunAView = rerunA === null ? null : view(firstDigest, rerunA, boundAgreement(firstAgreement, rerunABytes));
  const rerunBView = rerunB === null ? null : view(secondDigest, rerunB, boundAgreement(secondAgreement, rerunBBytes));
  return attributeSplit(firstView, secondView, first.verdict, second.verdict, rerunAView, rerunBView);
}

function retainedView(entry: AssessmentEntry): AssessmentView {
  return {
    docSha256: entry.docSha256,
    axes: { ...entry.axes },
    context: { ...entry.context },
    agreement: null,
  };
}

export function classifyAndAttribute(
  store: AssessmentStore,
  docSha256: string,
  result: VerifyResultLike,
  rerunA: VerifyResultLike | null = null,
  rerunB: VerifyResultLike | null = null,
): { disagreement: Disagreement; specCase: SpecCase } {
  const original = store.replay(docSha256);
  if (original === null) {
    throw new Error("no retained original for this digest; record first");
  }
  const firstView = retainedView(original);
  const secondView = view(docSha256, result, null);
  const rerunAView = rerunA === null ? null : view(docSha256, rerunA, null);
  const rerunBView = rerunB === null ? null : view(docSha256, rerunB, null);
  return attributeSplit(firstView, secondView, original.verdict, result.verdict, rerunAView, rerunBView);
}
