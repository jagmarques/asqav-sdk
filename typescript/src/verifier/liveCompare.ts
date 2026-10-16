/** Compare two live VerifyResults, then classify (488 C19, C5 core). */

import { type AgreementRef } from "./agreement.js";
import { classify, type Disagreement } from "./disagreement.js";
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
): Disagreement {
  const firstView = view(firstDigest, first, firstAgreement);
  const secondView = view(secondDigest, second, secondAgreement);
  return classify(firstView, secondView, first.verdict, second.verdict);
}
