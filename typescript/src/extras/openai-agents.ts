// OpenAI Agents JS integration. Wraps the current FunctionTool invoke (or an
// execute-only tool) so a sign precedes the body unless observe; refusal throws

import { AsqavAdapter, raiseMissingPeer, safeStringify, type AsqavAdapterOptions } from "./_base.js";

const MAX_PREVIEW = 200;

function truncate(s: string, max = MAX_PREVIEW): string {
  return s.length > max ? s.slice(0, max) : s;
}

// Minimal OpenAI Agents JS tool surface: the current FunctionTool has invoke,
// older shapes exposed execute; nothing is bound to an SDK version
export interface OpenAIAgentsToolLike<I = unknown, O = unknown> {
  name?: string;
  description?: string;
  invoke?: (runContext: any, input: string, details?: any) => Promise<O> | O;
  execute?: (input: I, ...rest: unknown[]) => Promise<O> | O;
  [key: string]: unknown;
}

export interface AsqavOpenAIAgentsAdapterOptions extends AsqavAdapterOptions {
  /** Optional adapter display name. */
  name?: string;
}

// Wraps OpenAI Agents JS tools and callbacks so tool events flow through
// Asqav, mirroring the Python AsqavGuardrail and AsqavTracingProcessor
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

  // Sign tool:start, run the call, then sign tool:end or tool:error
  // A refused start throws, so the original tool never runs
  private async runGated<O>(
    toolName: string,
    inputPreview: string,
    call: () => Promise<O> | O,
  ): Promise<O> {
    await this.signActionOrThrow({
      actionType: "tool:start",
      context: {
        tool: String(toolName),
        input_preview: truncate(inputPreview),
      },
    });
    try {
      const out = await call();
      const outStr = safeStringify(out);
      this.signAction({
        actionType: "tool:end",
        context: {
          tool: String(toolName),
          output_length: outStr?.length ?? 0,
          output_preview: truncate(outStr ?? ""),
        },
        result: out,
      });
      return out;
    } catch (err) {
      this.signAction({
        actionType: "tool:error",
        context: {
          tool: String(toolName),
          error_type: err instanceof Error ? err.name : typeof err,
          error: truncate(err instanceof Error ? err.message : String(err)),
        },
        result: err instanceof Error ? `${err.name}: ${err.message}` : String(err),
      });
      throw err;
    }
  }

  // Wrap a tool so each call signs tool:start first, then tool:end or error
  // A refused start throws, so the original tool never runs
  wrapTool<I, O>(tool: OpenAIAgentsToolLike<I, O>): OpenAIAgentsToolLike<I, O> {
    const toolName = tool.name ?? "unknown";

    if (typeof tool.invoke === "function") {
      const original = tool.invoke.bind(tool);
      const invoke = (
        runContext: unknown,
        input: string,
        details?: unknown,
      ): Promise<O> => this.runGated(toolName, input, () => original(runContext, input, details));
      return { ...tool, invoke };
    }

    if (typeof tool.execute !== "function") {
      throw new Error("wrapTool: tool.invoke and tool.execute are both missing");
    }
    const original = tool.execute.bind(tool);
    const execute = (input: I, ...rest: unknown[]): Promise<O> =>
      this.runGated(toolName, safeStringify(input) ?? "", () => original(input, ...rest));
    return { ...tool, execute };
  }

  /** Sign an ``agent:input`` event. Use as an input guardrail. */
  public onAgentInput(input: unknown, agentName?: string): void {
    const repr = safeStringify(input);
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
    const repr = safeStringify(output);
    this.signAction({
      actionType: "agent:output",
      context: {
        agent_name: agentName ?? this.name,
        output_type: typeof output,
        output_length: repr?.length ?? 0,
        output_preview: truncate(repr ?? ""),
      },
      result: output,
    });
  }
}
