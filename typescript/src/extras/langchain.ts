/**
 * LangChain.js integration for Asqav.
 *
 * Install (peer dependency):
 *   npm install @langchain/core
 *
 * Usage:
 *   import { init } from "@asqav/sdk";
 *   import { AsqavCallbackHandler } from "@asqav/sdk/extras/langchain";
 *   init({ apiKey: process.env.ASQAV_API_KEY! });
 *   const handler = await AsqavCallbackHandler.create({ agentName: "my-chain" });
 *   await chain.invoke(input, { callbacks: [handler] });
 *
 * Mirrors the Python ``asqav.extras.langchain.AsqavCallbackHandler``
 * (chain/tool/LLM lifecycle signing) using the LangChain.js
 * ``BaseCallbackHandler`` from ``@langchain/core/callbacks/base``.
 */

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

/**
 * Lazily import ``BaseCallbackHandler`` from ``@langchain/core``.
 *
 * The Python ImportError contract is preserved by re-throwing a clear
 * error when the peer is missing. Callers normally invoke
 * ``AsqavCallbackHandler.create`` which forwards through this.
 */
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

/**
 * LangChain.js callback handler that signs chain, tool, and LLM events
 * via the Asqav governance pipeline. Use
 * ``AsqavCallbackHandler.create()`` since the parent class is loaded
 * asynchronously.
 */
export class AsqavCallbackHandler extends AsqavAdapter {
  /** Friendly name surfaced to LangChain's tracing UI. */
  public readonly name: string = "asqav-callback-handler";

  /**
   * Build the handler. The factory is async because LangChain.js's
   * ``BaseCallbackHandler`` is the constructor we extend at runtime via
   * a returned subclass (so this class itself stays usable without the
   * peer installed).
   */
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
        serialized: LCSerialized,
        inputs: Record<string, unknown>,
      ): void {
        const fallbackName = (serialized.id ?? ["unknown"]).slice(-1)[0];
        this._asqav._emit("chain:start", {
          chain: String(serialized.name ?? fallbackName),
          input_keys: Object.keys(inputs ?? {}),
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
