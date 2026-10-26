/**
 * Barrel for Asqav framework adapters. Each is also published as a discrete subpath export so peers stay
 * tree-shakable; this exists for callers that already depend on several.
 */

export {
  AsqavAdapter,
  AsqavRefusedError,
  assertSignedVerdict,
  raiseMissingPeer,
  signOrThrow,
  type AsqavAdapterOptions,
  type AsqavErrorReporter,
  type SignActionInput,
} from "./_base.js";
export { AsqavCallbackHandler, enableLangchainGovernance } from "./langchain.js";
export type { AsqavCallbackHandlerOptions } from "./langchain.js";
export { AsqavMastraHook } from "./mastra.js";
export type { AsqavMastraHookOptions, MastraToolHooks } from "./mastra.js";
export { AsqavOpenAIAgentsAdapter } from "./openai-agents.js";
export type {
  AsqavOpenAIAgentsAdapterOptions,
  OpenAIAgentsToolLike,
} from "./openai-agents.js";
export {
  AsqavBlockedError,
  asqavGuard,
  createAsqavExporter,
  mapSpanNameToActionType,
  wrapTools,
} from "./vercel-ai.js";
export type { AiToolLike, AsqavGuardOptions } from "./vercel-ai.js";
// Reference detector extras (criterion 331).
export { PresidioDetector } from "./presidio.js";
export { OpaDetector } from "./opa.js";
