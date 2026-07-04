/**
 * Barrel for Asqav framework adapters.
 *
 * Each adapter is also published as a discrete subpath export
 * (``@asqav/sdk/extras/<framework>``) so peers stay tree-shakable. This
 * barrel exists so callers who already depend on multiple peers can
 * import everything from a single specifier.
 */

export { AsqavAdapter, raiseMissingPeer, type AsqavAdapterOptions } from "./_base.js";
export { AsqavCallbackHandler } from "./langchain.js";
export type { AsqavCallbackHandlerOptions } from "./langchain.js";
export { AsqavMcpAdapter, enableMcpGovernance } from "./mcp.js";
export type { AsqavMcpAdapterOptions, McpServerLike } from "./mcp.js";
export { AsqavMastraHook } from "./mastra.js";
export type { AsqavMastraHookOptions } from "./mastra.js";
export { AsqavOpenAIAgentsAdapter } from "./openai-agents.js";
export type {
  AsqavOpenAIAgentsAdapterOptions,
  OpenAIAgentsToolLike,
} from "./openai-agents.js";
export { createAsqavExporter, mapSpanNameToActionType } from "./vercel-ai.js";
// Reference detector extras (criterion 331).
export { PresidioDetector } from "./presidio.js";
export { OpaDetector } from "./opa.js";
