// Mastra integration. Gates tool calls through current hooks or wrapped tools
// Every call is signed before it runs; a refusal stops it. See the README

import {
  AsqavAdapter,
  raiseMissingPeer,
  safeStringify,
  type AsqavAdapterOptions,
} from "./_base.js";

const MAX_PREVIEW = 200;

function truncate(s: string, max = MAX_PREVIEW): string {
  return s.length > max ? s.slice(0, max) : s;
}

/** The current `@mastra/core` tool shape: a name plus an awaited `execute`. */
interface MastraToolLike {
  id?: string;
  execute?: (...args: any[]) => unknown;
}

// The current @mastra/core agent surface: listTools and __setTools on Agent
// @mastra/core is imported lazily, so the SDK installs without it
interface MastraAgentLike {
  readonly name?: string;
  listTools?: () => Promise<Record<string, MastraToolLike>>;
  __setTools?: (tools: any) => void;
}

interface MastraToolHookContext {
  toolName?: string;
  input?: unknown;
  context?: unknown;
  metadata?: { agentName?: string };
  output?: unknown;
  error?: unknown;
}

/** The hooks object accepted by new Agent({ hooks }) and generate */
export interface MastraToolHooks {
  beforeToolCall?: (context: MastraToolHookContext) => Promise<void>;
  afterToolCall?: (context: MastraToolHookContext) => Promise<void>;
}

interface MastraStepEvent {
  /** The Mastra step identifier (tool name, LLM call, sub-workflow). */
  stepId?: string;
  stepType?: string;
  input?: unknown;
  output?: unknown;
  error?: unknown;
  agentName?: string;
}

export interface AsqavMastraHookOptions extends AsqavAdapterOptions {
  /** Optional agent display name; defaults to "asqav-mastra-hook". */
  name?: string;
}

// Mastra hook signing tool lifecycle events through Asqav. The pre-execution
// before awaits the signature and throws on refusal; after records outcomes
export class AsqavMastraHook extends AsqavAdapter {
  public readonly name: string;

  constructor(options: AsqavMastraHookOptions) {
    super(options);
    this.name = options.name ?? "asqav-mastra-hook";
  }

  /**
   * Verify the Mastra peer is installed and return its module. Throws a
   * clear error matching the Python ImportError contract when missing.
   */
  static async loadPeer(): Promise<unknown> {
    try {
      // Optional peer; not declared in normal dependencies.
      // @ts-ignore optional peer dependency
      return (await import("@mastra/core")) as unknown;
    } catch (err) {
      raiseMissingPeer(
        "Mastra",
        "@mastra/core",
        "npm install @mastra/core",
        err,
      );
    }
  }

  // Hooks for new Agent({ hooks }) or agent.generate(..., { hooks }). Use these
  // when the agent is constructed with hooks, or attach to wrap its tools
  get toolHooks(): MastraToolHooks {
    return {
      beforeToolCall: async (ctx) => {
        await this.before({
          stepId: ctx.toolName,
          stepType: "tool",
          input: ctx.input,
          agentName: ctx.metadata?.agentName,
        });
      },
      afterToolCall: async (ctx) => {
        const evt: MastraStepEvent = {
          stepId: ctx.toolName,
          stepType: "tool",
          output: ctx.output,
          error: ctx.error,
          agentName: ctx.metadata?.agentName,
        };
        if (ctx.error !== undefined) {
          this.onError(evt);
        } else {
          this.after(evt);
        }
      },
    };
  }

  // Wrap every tool in place through the agent's listTools and __setTools
  // Each execute signs first; a refusal throws before the original tool runs
  async attach(agent: MastraAgentLike): Promise<void> {
    await AsqavMastraHook.loadPeer();

    if (typeof agent.listTools !== "function" || typeof agent.__setTools !== "function") {
      throw new Error(
        "AsqavMastraHook.attach: this @mastra/core Agent has no listTools/__setTools; "
        + "pass toolHooks to new Agent({ hooks: hook.toolHooks }) instead.",
      );
    }

    const tools = await agent.listTools();
    agent.__setTools(this.wrapTools(tools));
  }

  /** Wrap a tool set so each call is signed before it runs */
  wrapTools<T extends Record<string, MastraToolLike>>(tools: T): T {
    const out: Record<string, MastraToolLike> = {};
    for (const [name, tool] of Object.entries(tools)) {
      if (typeof tool?.execute !== "function") {
        out[name] = tool;
        continue;
      }
      const original = tool.execute.bind(tool);
      out[name] = {
        ...tool,
        execute: async (input: unknown, context?: unknown) => {
          await this.before({ stepId: name, stepType: "tool", input });
          if (this.options.observe) {
            return original(input, context);
          }
          try {
            const output = await original(input, context);
            this.after({ stepId: name, stepType: "tool", output });
            return output;
          } catch (err) {
            this.onError({ stepId: name, stepType: "tool", error: err });
            throw err;
          }
        },
      };
    }
    return out as T;
  }

  /** Sign a Mastra step start; throws on refusal, so the step does not run */
  public async before(evt: MastraStepEvent): Promise<void> {
    const ctx: Record<string, unknown> = {
      step_id: String(evt.stepId ?? "unknown"),
      step_type: String(evt.stepType ?? "step"),
      agent_name: evt.agentName ?? this.name,
    };
    if (evt.input !== undefined) {
      const inputStr = safeStringify(evt.input);
      ctx.input_preview = truncate(inputStr);
    }
    await this.signActionOrThrow({ actionType: "step:before", context: ctx });
  }

  /** Sign the successful end of a Mastra step. */
  public after(evt: MastraStepEvent): void {
    const ctx: Record<string, unknown> = {
      step_id: String(evt.stepId ?? "unknown"),
      step_type: String(evt.stepType ?? "step"),
      agent_name: evt.agentName ?? this.name,
    };
    if (evt.output !== undefined) {
      const outputStr = safeStringify(evt.output);
      ctx.output_preview = truncate(outputStr);
      ctx.output_length = outputStr?.length ?? 0;
    }
    this.signAction({ actionType: "step:after", context: ctx, result: evt.output });
  }

  /** Sign an error event from a Mastra step. */
  public onError(evt: MastraStepEvent): void {
    const err = evt.error;
    const ctx: Record<string, unknown> = {
      step_id: String(evt.stepId ?? "unknown"),
      step_type: String(evt.stepType ?? "step"),
      agent_name: evt.agentName ?? this.name,
      error_type: err instanceof Error ? err.name : typeof err,
      error: truncate(err instanceof Error ? err.message : String(err)),
    };
    this.signAction({
      actionType: "step:error",
      context: ctx,
      result: err instanceof Error ? `${err.name}: ${err.message}` : String(err),
    });
  }
}
