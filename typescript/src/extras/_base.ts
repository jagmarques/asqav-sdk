// Base adapter for Asqav framework integrations, as Python asqav.extras._base:
// resolves an Agent, fail-open ``signAction``; adapters compose, not inherit

import { Agent } from "../index.js";

export { safeStringify, toJsonSafeResult } from "../jcs.js";

// Provide exactly one of ``agent`` (used directly), ``agentId`` (lazy
// ``Agent.get``) or ``agentName`` (lazy ``Agent.create``)
/** Options accepted by every Asqav adapter constructor */
export interface AsqavAdapterOptions {
  /** Pre-built Asqav Agent. Takes precedence over agentId/agentName. */
  agent?: Agent;
  /** ID of an existing Asqav agent. Resolved lazily via Agent.get. */
  agentId?: string;
  /** Name for a new Asqav agent. Resolved lazily via Agent.create. */
  agentName?: string;
  // The adapter never throws into the host AI pipeline: governance fails open
  /** Optional error sink, defaults to ``console.warn`` */
  onError?: (err: unknown, context: { actionType: string }) => void;
  // Useful for local development and snapshot testing
  /** When true, log what would be signed and skip the network call */
  observe?: boolean;
}

interface SignActionInput {
  actionType: string;
  context?: Record<string, unknown>;
  /** Completion value; its digest is default-sent (787). */
  result?: unknown;
}

// Subclasses call ``signAction`` on every framework callback and MUST NOT call
// the SDK directly, so the fail-open contract lives in one place
/** Shared base for the framework adapters here */
export class AsqavAdapter {
  protected readonly options: AsqavAdapterOptions;
  protected agent: Agent | null;
  private agentPromise: Promise<Agent> | null = null;

  constructor(options: AsqavAdapterOptions) {
    if (!options.agent && !options.agentId && !options.agentName) {
      throw new Error(
        "AsqavAdapter requires one of { agent, agentId, agentName }. "
        + "Call asqav.init() first and supply an Agent or identifier.",
      );
    }
    this.options = options;
    this.agent = options.agent ?? null;
  }

  /** Resolve the Asqav agent, creating or fetching lazily if needed. */
  protected async resolveAgent(): Promise<Agent> {
    if (this.agent) return this.agent;
    if (this.agentPromise) return this.agentPromise;

    const promise = (async (): Promise<Agent> => {
      if (this.options.agentId) {
        return Agent.get(this.options.agentId);
      }
      if (this.options.agentName) {
        return Agent.create({ name: this.options.agentName });
      }
      throw new Error("AsqavAdapter: no agent identifier available");
    })();

    this.agentPromise = promise;
    const resolved = await promise;
    this.agent = resolved;
    return resolved;
  }

  // Errors route through ``options.onError``
  /** Sign a governance action, fire-and-forget; MUST NOT throw */
  protected signAction(input: SignActionInput): void {
    const onError = this.options.onError ?? defaultOnError;

    if (this.options.observe) {
      // observe mode mirrors Python ``AsqavAdapter._observe``
      // eslint-disable-next-line no-console
      console.info(
        `[asqav] OBSERVE: would sign ${input.actionType}`,
        input.context ?? {},
      );
      return;
    }

    // Fire-and-forget; never await so callbacks stay synchronous for host
    // frameworks that expect synchronous return values
    void (async () => {
      try {
        const agent = await this.resolveAgent();
        await agent.sign({
          actionType: input.actionType,
          context: input.context,
          ...(input.result !== undefined ? { result: input.result as never } : {}),
        });
      } catch (err) {
        try {
          onError(err, { actionType: input.actionType });
        } catch {
          // Swallow secondary errors from the user-supplied handler
        }
      }
    })();
  }
}

function defaultOnError(err: unknown, ctx: { actionType: string }): void {
  // eslint-disable-next-line no-console
  console.warn(`[asqav] sign failed for ${ctx.actionType}:`, err);
}

// ``cause`` keeps the underlying module-resolution error visible to the user
/** Throw the canonical missing-peer error (Python ``ImportError`` contract) */
export function raiseMissingPeer(
  framework: string,
  peer: string,
  install: string,
  cause?: unknown,
): never {
  const suffix = cause
    ? ` (import error: ${cause instanceof Error ? cause.message : String(cause)})`
    : "";
  throw new Error(
    `${framework} integration requires ${peer}. Install with: ${install}.${suffix}`,
  );
}
