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
 *   - canonicalisers, crypto, and the conformance runner.
 */

import { FormatAdapter } from "./adapter.js";
import { ActaAdapter } from "./adapters/acta.js";
import { AerfAdapter } from "./adapters/aerf.js";
import { AgentReceiptsAdapter } from "./adapters/agentreceipts.js";
import { AsqavNativeAdapter } from "./adapters/asqavNative.js";
import { AuthproofAdapter } from "./adapters/authproof.js";

/** Detection fingerprints are mutually exclusive, so registration order is not load-bearing. */
export const ADAPTERS: FormatAdapter[] = [
  new AsqavNativeAdapter(),
  new AerfAdapter(),
  new ActaAdapter(),
  new AgentReceiptsAdapter(),
  new AuthproofAdapter(),
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
export { detect, verify } from "./core.js";
export type { AxisResult, Verdict, VerifyResult } from "./core.js";
export { asqavJcs, jcs, jcsRfc8785, parseJsonPreservingFloats, RawFloat } from "./canonical.js";
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
  runCorpus,
  runOne,
  type VectorOutcome,
} from "./runner.js";
