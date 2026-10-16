/** Compare two live VerifyResults, then classify (488 C19, C5 core). */

import { createHash } from "node:crypto";

import { type AgreementRef, type AssessmentView } from "./agreement.js";
import { classify, specCase, type Disagreement, type SpecCase } from "./disagreement.js";
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
  for (const [tag, digest, raw] of [["first", firstDigest, firstBytes], ["second", secondDigest, secondBytes]] as const) {
    if (raw !== null && createHash("sha256").update(raw).digest("hex") !== digest) {
      throw new Error(`${tag} digest is not the sha256 of the stated bytes`);
    }
  }  const firstView = view(firstDigest, first, firstAgreement);
  const secondView = view(secondDigest, second, secondAgreement);
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
  for (const [tag, digest, raw] of [["first", firstDigest, firstBytes], ["second", secondDigest, secondBytes], ["rerunA", firstDigest, rerunABytes], ["rerunB", secondDigest, rerunBBytes]] as const) {
    if (raw !== null && createHash("sha256").update(raw).digest("hex") !== digest) {
      throw new Error(`${tag} digest is not the sha256 of the stated bytes`);
    }
  }  const firstView = view(firstDigest, first, firstAgreement);
  const secondView = view(secondDigest, second, secondAgreement);
  const rerunAView = rerunA === null ? null : view(firstDigest, rerunA, firstAgreement);
  const rerunBView = rerunB === null ? null : view(secondDigest, rerunB, secondAgreement);
  return attributeSplit(firstView, secondView, first.verdict, second.verdict, rerunAView, rerunBView);
}
