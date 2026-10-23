// Mastra integration: signs agent step ``before`` / ``after`` events so each
// tool call, LLM call and workflow step is a receipt; needs `@mastra/core`

import { AsqavAdapter, raiseMissingPeer, safeStringify, type AsqavAdapterOptions } from "./_base.js";

const MAX_PREVIEW = 200;

function truncate(s: string, max = MAX_PREVIEW): string {
  return s.length > max ? s.slice(0, max) : s;
}

// ``@mastra/core`` is never imported at the top level, so the SDK installs
// cleanly without it
/** Minimum surface we touch on a Mastra agent */
interface MastraAgentLike {
  readonly name?: string;
  on?: (event: string, listener: (...args: unknown[]) => void) => unknown;
  use?: (middleware: unknown) => unknown;
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

// ``before`` and ``after`` are public so callers can wire them into custom
// middleware or call them from tests
/** Mastra hook signing agent step lifecycle events through Asqav */
export class AsqavMastraHook extends AsqavAdapter {
  public readonly name: string;

  constructor(options: AsqavMastraHookOptions) {
    super(options);
    this.name = options.name ?? "asqav-mastra-hook";
  }

  // Throws a clear error matching the Python ImportError contract when missing
  /** Verify the Mastra peer is installed and return its module */
  static async loadPeer(): Promise<unknown> {
    try {
      // Optional peer; not declared in normal dependencies
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

  // Uses the agent's event emitter when it exposes one, otherwise registers a
  // middleware that wraps each step
  /** Attach to a Mastra agent: subscribe to ``step:before`` / ``step:after`` */
  async attach(agent: MastraAgentLike): Promise<void> {
    // Validate the peer is available so attach() fails fast with a helpful
    // message rather than at first event time
    await AsqavMastraHook.loadPeer();

    if (typeof agent.on === "function") {
      agent.on("step:before", (evt: unknown) => {
        this.before((evt ?? {}) as MastraStepEvent);
      });
      agent.on("step:after", (evt: unknown) => {
        this.after((evt ?? {}) as MastraStepEvent);
      });
      agent.on("step:error", (evt: unknown) => {
        this.onError((evt ?? {}) as MastraStepEvent);
      });
      return;
    }

    if (typeof agent.use === "function") {
      const self = this;
      agent.use({
        async wrap(
          ctx: MastraStepEvent,
          next: () => Promise<unknown>,
        ): Promise<unknown> {
          self.before(ctx);
          try {
            const out = await next();
            self.after({ ...ctx, output: out });
            return out;
          } catch (err) {
            self.onError({ ...ctx, error: err });
            throw err;
          }
        },
      });
      return;
    }

    throw new Error(
      "AsqavMastraHook.attach: agent exposes neither .on() nor .use(); "
      + "wire .before/.after manually instead.",
    );
  }

  /** Sign the start of a Mastra step. Safe to call directly from tests. */
  public before(evt: MastraStepEvent): void {
    const ctx: Record<string, unknown> = {
      step_id: String(evt.stepId ?? "unknown"),
      step_type: String(evt.stepType ?? "step"),
      agent_name: evt.agentName ?? this.name,
    };
    if (evt.input !== undefined) {
      const inputStr = safeStringify(evt.input);
      ctx.input_preview = truncate(inputStr);
    }
    this.signAction({ actionType: "step:before", context: ctx });
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
