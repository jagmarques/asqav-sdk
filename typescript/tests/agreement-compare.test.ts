/** Agreement comparanda, TS half (C12; C29 binds envelope from the bytes). */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { ADAPTERS } from "../src/verifier/index.js";
import {
  COMPARANDA,
  agreementOfTerms,
  bindEnvelope,
  compareAssessments,
  envelopeOf,
  type AssessmentView,
} from "../src/verifier/agreement.js";

const CTX = {
  evaluator: "c12-probe",
  trustRoots: ["root-a", "root-b"],
  revocationSnapshot: "snap-1",
  evaluationTime: "2029-06-01T00:00:00Z",
};
const AXES = { expiry: "PASS", skew: "PASS" };
const AGR = agreementOfTerms("agr-1", "2029-01", { clause: "pay on delivery" });

function view(over: Partial<AssessmentView> = {}): AssessmentView {
  return {
    docSha256: "dd".repeat(32),
    axes: { ...AXES },
    context: { ...CTX },
    agreement: AGR,
    ...over,
  };
}

describe("agreement comparanda", () => {
  it("matches on all seven for identical views", () => {
    expect(compareAssessments(view(), view())).toEqual(
      Object.fromEntries(COMPARANDA.map((k) => [k, true])),
    );
  });

  it("flips only agreementVersion on changed terms", () => {
    const other = view({ agreement: agreementOfTerms("agr-1", "2029-01", { clause: "pay upfront" }) });
    const result = compareAssessments(view(), other);
    expect(result.agreementVersion).toBe(false);
    expect(Object.entries(result).filter(([k]) => k !== "agreementVersion").every(([, v]) => v)).toBe(true);
  });

  it("flips only verificationTime on a later evaluation", () => {
    const other = view({ context: { ...CTX, evaluationTime: "2031-01-01T00:00:00Z" } });
    const result = compareAssessments(view(), other);
    expect(result.verificationTime).toBe(false);
    expect(Object.entries(result).filter(([k]) => k !== "verificationTime").every(([, v]) => v)).toBe(true);
  });

  it("treats a missing agreement as mismatch, not a crash", () => {
    const result = compareAssessments(view({ agreement: null }), view());
    expect(result.agreementVersion).toBe(false);
    expect(result.evidenceBytes).toBe(true);
  });

  it("reads snake_case contexts from the other language", () => {
    const snake = view({
      context: {
        evaluator: "c12-probe",
        trust_roots: ["root-b", "root-a"],
        revocation_snapshot: "snap-1",
        evaluation_time: "2029-06-01T00:00:00Z",
      },
    });
    expect(compareAssessments(view(), snake)).toEqual(
      Object.fromEntries(COMPARANDA.map((k) => [k, true])),
    );
  });

  it("hashes non-ASCII terms identically to the other language", () => {
    const ref = agreementOfTerms("agr-1", "2029-01", { clause: "páy on delivery ✓" });
    expect(ref.termsHash).toBe("3a357c459fc615cbc788440180f136108db0051615defa7f819bf38b8b72f3f7");
  });

  it("compares cross-casing agreement dicts by value", () => {
    const snake = { agreement_id: "agr-1", predicate_version: "2029-01", terms_hash: AGR.termsHash };
    expect(compareAssessments(view(), view({ agreement: snake })).agreementVersion).toBe(true);
    const different = { ...snake, terms_hash: "DIFFERENT" };
    expect(
      compareAssessments(view(), view({ agreement: different })).agreementVersion,
    ).toBe(false);
  });

  it("matches two absent agreements", () => {
    expect(compareAssessments(view({ agreement: null }), view({ agreement: null })).agreementVersion).toBe(
      true,
    );
  });
});

const FIX = join(__dirname, "..", "..", "verifier", "signed-conflict-fixtures");
const CORPUS = join(__dirname, "..", "..", "verifier", "conformance-vectors");

const ALLOW_KID =
  "did:key:z6MktqPnbGGsq5TR4h9dYRWtkoSQtiZt9CXfUFNcfxA7Fi5x#z6MktqPnbGGsq5TR4h9dYRWtkoSQtiZt9CXfUFNcfxA7Fi5x";
const ALLOW_SIG =
  "e7ef246a853f05b7731bec610f91481a147589e751ef642d15b4f3c5c32062eddf1277413023aa9b677b293d751e81722e356df2eeb211d1eaaa3d0f8421c904";
const DENY_SIG =
  "831d9f99e3fdd33580257453aaea01ac1a5b677cd96955886aaa6da812b47408d904d2b3a51c96ba7de4611c7b113ad7b702aa8142652f8f80b8770ff010c006";
const GENESIS_KID = "asqav-oracle-vec-key";
const GENESIS_SIG =
  "332a20933137e4daa5e4e23830e95631cc7facbb04a1d8bb5013bc358a189e531df6cdfbac191fe814c206f28b7261d71205cfae904d21bcf265f2c130f8d903";

function doc(name: string): Record<string, unknown> {
  return JSON.parse(readFileSync(join(FIX, `${name}.json`), "utf-8")) as Record<string, unknown>;
}

function genesisDoc(): Record<string, unknown> {
  return JSON.parse(
    readFileSync(join(CORPUS, "asqav-01-genesis-permit", "receipt.json"), "utf-8"),
  ) as Record<string, unknown>;
}

function envView(docSha: string, agreement: AssessmentView["agreement"]): AssessmentView {
  return { docSha256: docSha, axes: { ...AXES }, context: { ...CTX }, agreement };
}

describe("envelope binding", () => {
  it("reads kid and signature from the bytes", () => {
    const allow = envelopeOf(doc("conflict-allow"), ADAPTERS);
    expect([allow.kid, allow.signature]).toEqual([ALLOW_KID, ALLOW_SIG]);
    expect([allow.agreementId, allow.predicateVersion, allow.termsHash]).toEqual([null, null, null]);
    const genesis = envelopeOf(genesisDoc(), ADAPTERS);
    expect([genesis.kid, genesis.signature]).toEqual([GENESIS_KID, GENESIS_SIG]);
  });

  it("throws on an undetectable format", () => {
    expect(() => envelopeOf({ hello: "world" }, ADAPTERS)).toThrow();
  });

  it("flips only agreementVersion on a kid change", () => {
    const result = compareAssessments(
      envView("ab".repeat(32), envelopeOf(doc("conflict-allow"), ADAPTERS)),
      envView("ab".repeat(32), envelopeOf(genesisDoc(), ADAPTERS)),
    );
    expect(result.agreementVersion).toBe(false);
    expect(Object.entries(result).filter(([k]) => k !== "agreementVersion").every(([, v]) => v)).toBe(true);
  });

  it("flips agreementVersion on a kid change alone", () => {
    const base = envelopeOf(doc("conflict-allow"), ADAPTERS);
    const other = { ...base, kid: "another-key" };
    const result = compareAssessments(
      envView("ab".repeat(32), base),
      envView("ab".repeat(32), other),
    );
    expect(result.agreementVersion).toBe(false);
    expect(Object.entries(result).filter(([k]) => k !== "agreementVersion").every(([, v]) => v)).toBe(true);
  });

  it("flips only agreementVersion on a signature change", () => {
    const allow = envelopeOf(doc("conflict-allow"), ADAPTERS);
    const deny = envelopeOf(doc("conflict-deny"), ADAPTERS);
    expect(allow.kid).toBe(deny.kid);
    expect(allow.signature).not.toBe(deny.signature);
    expect(deny.signature).toBe(DENY_SIG);
    const result = compareAssessments(
      envView("ab".repeat(32), allow),
      envView("ab".repeat(32), deny),
    );
    expect(result.agreementVersion).toBe(false);
    expect(Object.entries(result).filter(([k]) => k !== "agreementVersion").every(([, v]) => v)).toBe(true);
  });

  it("merges caller terms with the bytes envelope", () => {
    const terms = agreementOfTerms("agr-1", "2029-01", { clause: "pay on delivery" });
    const bound = bindEnvelope(terms, doc("conflict-allow"), ADAPTERS);
    expect([bound.agreementId, bound.predicateVersion, bound.termsHash]).toEqual([
      terms.agreementId, terms.predicateVersion, terms.termsHash,
    ]);
    expect([bound.kid, bound.signature]).toEqual([ALLOW_KID, ALLOW_SIG]);
    const bare = bindEnvelope(null, doc("conflict-allow"), ADAPTERS);
    expect([bare.agreementId, bare.termsHash, bare.kid]).toEqual([null, null, ALLOW_KID]);
  });

  it("throws on a forged stated envelope", () => {
    expect(() =>
      bindEnvelope({ kid: "attacker-key", signature: "00".repeat(64) }, doc("conflict-allow"), ADAPTERS),
    ).toThrow();
    expect(() =>
      bindEnvelope({ kid: ALLOW_KID, signature: "00".repeat(64) }, doc("conflict-allow"), ADAPTERS),
    ).toThrow();
    const honest = bindEnvelope({ kid: ALLOW_KID, signature: ALLOW_SIG }, doc("conflict-allow"), ADAPTERS);
    expect(honest.kid).toBe(ALLOW_KID);
  });
});
