/**
 * OpenAI Agents JS integration for Asqav.
 *
 * Install (peer dependency):
 *   npm install @openai/agents
 *
 * Usage:
 *   import { init } from "@asqav/sdk";
 *   import { AsqavOpenAIAgentsAdapter } from "@asqav/sdk/extras/openai-agents";
 *   init({ apiKey: process.env.ASQAV_API_KEY! });
 *   const adapter = new AsqavOpenAIAgentsAdapter({ agentName: "my-agent" });
 *   adapter.wrapTool(myTool);
 *
 * Mirrors the Python ``asqav.extras.openai_agents.AsqavGuardrail`` /
 * ``AsqavTracingProcessor`` adapter: signs ``agent:input`` /
 * ``agent:output`` / ``tool:start`` / ``tool:end`` events as the agent
 * runs. Operates fail-open so governance never breaks the agent loop.
 */

import { AsqavAdapter, raiseMissingPeer, type AsqavAdapterOptions } from "./_base.js";

const MAX_PREVIEW = 200;

function truncate(s: string, max = MAX_PREVIEW): string {
  return s.length > max ? s.slice(0, max) : s;
}

/**
 * Minimal surface of an OpenAI Agents JS ``tool``. We accept any object
 * with ``name`` and ``execute`` so we never bind to a specific SDK
 * version.
 */
export interface OpenAIAgentsToolLike<I = unknown, O = unknown> {
  name?: string;
  description?: string;
  execute?: (input: I, ...rest: unknown[]) => Promise<O> | O;
}

export interface AsqavOpenAIAgentsAdapterOptions extends AsqavAdapterOptions {
  /** Optional adapter display name. */
  name?: string;
}

/**
 * Wraps OpenAI Agents JS tools and run-loop callbacks so every
 * tool-call lifecycle event flows through Asqav. Mirrors the Python
 * ``AsqavGuardrail`` and ``AsqavTracingProcessor``.
 */
export class AsqavOpenAIAgentsAdapter extends AsqavAdapter {
  public readonly name: string;

  constructor(options: AsqavOpenAIAgentsAdapterOptions) {
    super(options);
    this.name = options.name ?? "asqav-openai-agents-adapter";
  }

  /** Verify ``@openai/agents`` is installed; throw a clear error if not. */
  static async loadPeer(): Promise<unknown> {
    try {
      // Optional peer; not declared in normal dependencies.
      // @ts-ignore optional peer dependency
      return (await import("@openai/agents")) as unknown;
    } catch (err) {
      raiseMissingPeer(
        "OpenAI Agents JS",
        "@openai/agents",
        "npm install @openai/agents",
        err,
      );
    }
  }

  /**
   * Wrap an OpenAI Agents tool so each invocation signs
   * ``tool:start`` and ``tool:end`` (or ``tool:error``). Returns a new
   * tool object with the same ``name``/``description`` and a wrapped
   * ``execute``.
   */
  wrapTool<I, O>(tool: OpenAIAgentsToolLike<I, O>): OpenAIAgentsToolLike<I, O> {
    if (typeof tool.execute !== "function") {
      throw new Error("wrapTool: tool.execute must be a function");
    }
    const original = tool.execute.bind(tool);
    const self = this;
    const toolName = tool.name ?? "unknown";

    const execute = async (input: I, ...rest: unknown[]): Promise<O> => {
      const inputStr = typeof input === "string" ? input : JSON.stringify(input);
      self.signAction({
        actionType: "tool:start",
        context: {
          tool: String(toolName),
          input_preview: truncate(inputStr ?? ""),
        },
      });
      try {
        const out = await original(input, ...rest);
        const outStr = typeof out === "string" ? out : JSON.stringify(out);
        self.signAction({
          actionType: "tool:end",
          context: {
            tool: String(toolName),
            output_length: outStr?.length ?? 0,
            output_preview: truncate(outStr ?? ""),
          },
        });
        return out;
      } catch (err) {
        self.signAction({
          actionType: "tool:error",
          context: {
            tool: String(toolName),
            error_type: err instanceof Error ? err.name : typeof err,
            error: truncate(err instanceof Error ? err.message : String(err)),
          },
        });
        throw err;
      }
    };

    return { ...tool, execute };
  }

  /** Sign an ``agent:input`` event. Use as an input guardrail. */
  public onAgentInput(input: unknown, agentName?: string): void {
    const repr = typeof input === "string" ? input : JSON.stringify(input);
    this.signAction({
      actionType: "agent:input",
      context: {
        agent_name: agentName ?? this.name,
        input_type: typeof input,
        input_length: repr?.length ?? 0,
        input_preview: truncate(repr ?? ""),
      },
    });
  }

  /** Sign an ``agent:output`` event. Use as an output guardrail. */
  public onAgentOutput(output: unknown, agentName?: string): void {
    const repr = typeof output === "string" ? output : JSON.stringify(output);
    this.signAction({
      actionType: "agent:output",
      context: {
        agent_name: agentName ?? this.name,
        output_type: typeof output,
        output_length: repr?.length ?? 0,
        output_preview: truncate(repr ?? ""),
      },
    });
  }
}
