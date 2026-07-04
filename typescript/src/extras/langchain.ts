// LangChain.js integration for Asqav. Install: npm install @langchain/core.
// Default-on: await enableLangchainGovernance({ agentName: "my-chain" }) then chain.invoke.
// Manual: const h = await AsqavCallbackHandler.create({...}); chain.invoke(i, {callbacks:[h]}).

import { AsqavAdapter, raiseMissingPeer, type AsqavAdapterOptions } from "./_base.js";

// Local LangChain.js types so this module compiles even when the peer
// package is not installed. The actual base class is loaded lazily.

interface LCSerialized {
  id?: string[];
  name?: string;
  [key: string]: unknown;
}

interface LCGeneration {
  text?: string;
  [key: string]: unknown;
}

interface LCLLMResult {
  generations: LCGeneration[][];
  llmOutput?: Record<string, unknown> | null;
}

interface BaseCallbackHandlerLike {
  // The LangChain.js base class declares many optional handlers. We only
  // need ours to type-check; consumers receive a real subclass instance.
  name?: string;
}

type BaseCallbackHandlerCtor = new (...args: unknown[]) => BaseCallbackHandlerLike;

const MAX_PREVIEW = 200;

function truncate(s: string, max = MAX_PREVIEW): string {
  return s.length > max ? s.slice(0, max) : s;
}

// Lazily import BaseCallbackHandler from @langchain/core.
// Re-throws a clear error when the optional peer is missing.
async function loadBase(): Promise<BaseCallbackHandlerCtor> {
  try {
    // Dynamic specifier: the peer is optional (package.json peerDependenciesMeta).
    // @ts-ignore optional peer dependency
    const mod = (await import("@langchain/core/callbacks/base")) as unknown;
    const ctor = (mod as { BaseCallbackHandler?: BaseCallbackHandlerCtor })
      .BaseCallbackHandler;
    if (!ctor) {
      throw new Error("BaseCallbackHandler not exported from @langchain/core/callbacks/base");
    }
    return ctor;
  } catch (err) {
    raiseMissingPeer(
      "LangChain.js",
      "@langchain/core",
      "npm install @langchain/core",
      err,
    );
  }
}

export interface AsqavCallbackHandlerOptions extends AsqavAdapterOptions {
  /** Optional callback handler name; defaults to "asqav-callback-handler". */
  name?: string;
}

// Signs chain, tool, and LLM events via the Asqav governance pipeline.
// Use AsqavCallbackHandler.create() since the peer is loaded asynchronously.
export class AsqavCallbackHandler extends AsqavAdapter {
  /** Friendly name surfaced to LangChain's tracing UI. */
  public readonly name: string = "asqav-callback-handler";

  // Async factory: loads BaseCallbackHandler from the optional peer at runtime.
  static async create(
    opts: AsqavCallbackHandlerOptions,
  ): Promise<AsqavCallbackHandler & BaseCallbackHandlerLike> {
    const Base = await loadBase();

    class _AsqavCallbackHandlerImpl extends (Base as unknown as new () => object) {
      public readonly name: string;
      private readonly _asqav: AsqavCallbackHandler;

      constructor(asqav: AsqavCallbackHandler, name: string) {
        super();
        this._asqav = asqav;
        this.name = name;
      }

      handleChainStart(
        serialized: LCSerialized | null | undefined,
        inputs: Record<string, unknown>,
      ): void {
        // Guard: LCEL/RunnableLambda passes non-object inputs (str/arr/num).
        // Object.keys on a string yields index keys ["0","1",...], so guard first.
        const input_keys =
          inputs !== null &&
          typeof inputs === "object" &&
          !Array.isArray(inputs)
            ? Object.keys(inputs)
            : [];
        // serialized may be null on the LCEL/RunnableLambda path; fall back
        // to a stable "chain" label so the receipt is always signed.
        if (!serialized) {
          this._asqav._emit("chain:start", {
            chain: "chain",
            input_keys,
          });
          return;
        }
        const fallbackName = (serialized.id ?? ["unknown"]).slice(-1)[0];
        this._asqav._emit("chain:start", {
          chain: String(serialized.name ?? fallbackName),
          input_keys,
        });
      }

      handleChainEnd(outputs: Record<string, unknown>): void {
        this._asqav._emit("chain:end", {
          output_keys: Object.keys(outputs ?? {}),
        });
      }

      handleChainError(err: Error): void {
        this._asqav._emit("chain:error", {
          error_type: err?.name ?? "Error",
          error: truncate(String(err?.message ?? err)),
        });
      }

      handleToolStart(serialized: LCSerialized, input: string): void {
        const fallbackName = (serialized.id ?? ["unknown"]).slice(-1)[0];
        this._asqav._emit("tool:start", {
          tool: String(serialized.name ?? fallbackName),
          input: truncate(String(input ?? "")),
        });
      }

      handleToolEnd(output: unknown): void {
        const outStr = typeof output === "string" ? output : JSON.stringify(output);
        this._asqav._emit("tool:end", {
          output_type: typeof output,
          output_length: outStr?.length ?? 0,
        });
      }

      handleToolError(err: Error): void {
        this._asqav._emit("tool:error", {
          error_type: err?.name ?? "Error",
          error: truncate(String(err?.message ?? err)),
        });
      }

      handleLLMStart(serialized: LCSerialized, prompts: string[]): void {
        const fallbackName = (serialized.id ?? ["unknown"]).slice(-1)[0];
        this._asqav._emit("llm:start", {
          model: String(serialized.name ?? fallbackName),
          prompt_count: Array.isArray(prompts) ? prompts.length : 0,
        });
      }

      handleLLMEnd(result: LCLLMResult): void {
        const generationCount = (result?.generations ?? []).reduce(
          (acc, gens) => acc + (gens?.length ?? 0),
          0,
        );
        const ctx: Record<string, unknown> = { generation_count: generationCount };
        const usage = result?.llmOutput?.tokenUsage ?? result?.llmOutput?.token_usage;
        if (usage && typeof usage === "object") {
          ctx.token_usage = Object.fromEntries(
            Object.entries(usage as Record<string, unknown>).filter(
              ([, v]) => typeof v === "number",
            ),
          );
        }
        this._asqav._emit("llm:end", ctx);
      }

      handleLLMError(err: Error): void {
        this._asqav._emit("llm:error", {
          error_type: err?.name ?? "Error",
          error: truncate(String(err?.message ?? err)),
        });
      }
    }

    const wrapper = new AsqavCallbackHandler(opts);
    const impl = new _AsqavCallbackHandlerImpl(
      wrapper,
      opts.name ?? "asqav-callback-handler",
    );
    // Make the underlying signAction reachable through the wrapper.
    return impl as unknown as AsqavCallbackHandler & BaseCallbackHandlerLike;
  }

  /** Internal hook used by the inner LangChain subclass. */
  public _emit(actionType: string, context: Record<string, unknown>): void {
    this.signAction({ actionType, context });
  }
}

// -- Default-on entrypoint ---------------------------------------------------

interface LangchainContextModule {
  registerConfigureHook: (config: { contextVar?: string; inheritable?: boolean }) => void;
  setContextVariable: (name: string, value: unknown) => void;
}

// Lazily import registerConfigureHook/setContextVariable from @langchain/core/context.
async function loadContext(): Promise<LangchainContextModule> {
  try {
    // @ts-ignore optional peer dependency
    const mod = (await import("@langchain/core/context")) as unknown;
    const m = mod as Partial<LangchainContextModule>;
    if (!m.registerConfigureHook || !m.setContextVariable) {
      throw new Error("registerConfigureHook/setContextVariable not exported from @langchain/core/context");
    }
    return m as LangchainContextModule;
  } catch (err) {
    raiseMissingPeer(
      "LangChain.js",
      "@langchain/core",
      "npm install @langchain/core",
      err,
    );
  }
}

const ASQAV_LC_CONTEXT_VAR = "asqav_langchain_handler";
let hookRegistered = false;

// Turn on default-on Asqav governance for LangChain.js. Registers a configure
// hook once and binds the handler to a context variable so every run signs
// automatically. Signing stays fail-open. Returns the handler.
export async function enableLangchainGovernance(
  opts: AsqavCallbackHandlerOptions,
): Promise<AsqavCallbackHandler & BaseCallbackHandlerLike> {
  const handler = await AsqavCallbackHandler.create(opts);
  const { registerConfigureHook, setContextVariable } = await loadContext();
  // Register the hook once so a second call cannot add a duplicate entry to
  // langchain-core's hook list and double-sign every action.
  if (!hookRegistered) {
    registerConfigureHook({ contextVar: ASQAV_LC_CONTEXT_VAR, inheritable: true });
    hookRegistered = true;
  }
  setContextVariable(ASQAV_LC_CONTEXT_VAR, handler);
  return handler;
}
