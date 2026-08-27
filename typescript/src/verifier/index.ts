/**
 * Universal neutral verifier (TypeScript) - verify agent receipts across formats.
 * A port of the Python oracle at `verifier/oracle/`, to parity with the proven
 * Python verifier.
 *
 * It proves only what a verifier can prove from the bytes: a valid signature over
 * the canonical bytes, a reproducible hash-chain link, and structural presence at
 * time T. It never attests behaviour or correctness of the recorded action.
 *
 * Public surface:
 *   - `FormatAdapter` : the 6-method per-format seam.
 *   - `ADAPTERS`      : ordered registry the dispatcher walks for detection.
 *   - `verify`        : verify one parsed receipt; returns a `VerifyResult`.
 *   - `checkAnchors` / `checkSkew` : the two axes the oracle leaves out.
 *   - `checkExpiry`   : the signed expires_at window, mirroring the hosted verdict.
 *   - canonicalisers, crypto, and the conformance runner.
 */

import { FormatAdapter } from "./adapter.js";
import { ActaAdapter } from "./adapters/acta.js";
import { AerfAdapter } from "./adapters/aerf.js";
import { AgentReceiptsAdapter } from "./adapters/agentreceipts.js";
import { AsqavNativeAdapter } from "./adapters/asqavNative.js";
import { AuthproofAdapter } from "./adapters/authproof.js";
import { PipelockEvidenceAdapter } from "./adapters/pipelock.js";
import { W3cVcAdapter } from "./adapters/w3cVc.js";

/** Detection fingerprints are mutually exclusive, so registration order is not load-bearing. */
export const ADAPTERS: FormatAdapter[] = [
  new AsqavNativeAdapter(),
  new AerfAdapter(),
  new ActaAdapter(),
  new AgentReceiptsAdapter(),
  new W3cVcAdapter(),
  new AuthproofAdapter(),
  new PipelockEvidenceAdapter(),
];

export { FormatAdapter } from "./adapter.js";
export type {
  AxisCheck,
  ChainStep,
  ExtraAxis,
  KeyProvider,
  SignatureMaterial,
} from "./adapter.js";
export { ActaAdapter } from "./adapters/acta.js";
export { AerfAdapter } from "./adapters/aerf.js";
export { AgentReceiptsAdapter } from "./adapters/agentreceipts.js";
export { AsqavNativeAdapter } from "./adapters/asqavNative.js";
export { AuthproofAdapter } from "./adapters/authproof.js";
export { PipelockEvidenceAdapter } from "./adapters/pipelock.js";
export { W3cVcAdapter } from "./adapters/w3cVc.js";
export {
  detect,
  FAILURE_INVALID,
  FAILURE_UNVERIFIABLE,
  MAX_NESTING_DEPTH,
  VERDICT_UNVERIFIED,
  VERDICT_VERIFIED,
  VERDICT_VERIFIED_KEYED,
  axisFailureClass,
  foldVerdict,
  verify,
} from "./core.js";
export type { AxisResult, FailureClass, Verdict, VerifyResult } from "./core.js";
// The axes the oracle leaves out by design, so a TypeScript caller runs what a
// Python caller runs. normaliseEnvelope ships too: skip it and you digest other bytes.
export {
  checkAnchors,
  checkExpiry,
  checkSkew,
  envelopeMinusAnchorsJcs,
  normaliseEnvelope,
  SKEW_BOUND_SECONDS,
} from "./vrShim.js";
export {
  asqavJcs,
  DuplicateMemberError,
  jcs,
  jcsRfc8785,
  parseJsonPreservingFloats,
  parseJsonStrict,
  RawFloat,
} from "./canonical.js";
export {
  FAIL,
  PASS,
  SKIPPED,
  sha256Hex,
  verifySignature,
  type VerifyOutcome,
  type VerifyState,
} from "./crypto.js";
export { resolveEd25519Key } from "./did.js";
export {
  buildPae,
  extractSubjectDigest,
  IN_TOTO_PAYLOAD_TYPE,
  IN_TOTO_STATEMENT_TYPE,
  verifyAttestation,
  type AttestationAxis,
  type AttestationAxes,
  type AttestationVerdict,
  type VerifyAttestationOptions,
} from "./dsse.js";
export {
  runCorpus,
  runOne,
  type VectorOutcome,
} from "./runner.js";
