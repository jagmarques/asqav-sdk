/**
 * Base adapter for Asqav framework integrations.
 *
 * Mirrors the Python ``asqav.extras._base.AsqavAdapter`` surface so the
 * documentation can cross-link the two SDKs. Framework-specific adapters
 * (LangChain.js, Mastra, OpenAI Agents JS, etc.) compose this class via
 * an internal field rather than multiple inheritance because TypeScript
 * does not support multi-class inheritance.
 *
 * Responsibilities:
 *   - Resolve an Asqav ``Agent`` from either a pre-built instance,
 *     ``agentId`` (calls ``Agent.get``), or ``agentName`` (calls
 *     ``Agent.create``).
 *   - Provide a shared ``signAction`` helper that is fail-open: signing
 *     errors are routed to the caller-supplied ``onError`` handler and
 *     never raise into the host pipeline.
 *   - Provide a stable place to attach future cross-cutting features
 *     (observe mode, session grouping) without touching every adapter.
 */

import { Agent } from "../index.js";

/**
 * Options accepted by every Asqav adapter constructor.
 *
 * Exactly one of ``agent``, ``agentId``, or ``agentName`` should be
 * provided. When ``agent`` is set the adapter signs against it directly.
 * When ``agentId`` is set the adapter calls ``Agent.get`` lazily on the
 * first event. When ``agentName`` is set the adapter calls
 * ``Agent.create`` lazily on the first event.
 */
export interface AsqavAdapterOptions {
  /** Pre-built Asqav Agent. Takes precedence over agentId/agentName. */
  agent?: Agent;
  /** ID of an existing Asqav agent. Resolved lazily via Agent.get. */
  agentId?: string;
  /** Name for a new Asqav agent. Resolved lazily via Agent.create. */
  agentName?: string;
  /**
   * Optional error sink. Defaults to ``console.warn``. The adapter never
   * throws into the host AI pipeline; governance must fail-open.
   */
  onError?: (err: unknown, context: { actionType: string }) => void;
  /**
   * When true, the adapter logs what it would sign but skips the network
   * call. Useful for local development and snapshot testing.
   */
  observe?: boolean;
}

export interface SignActionInput {
  actionType: string;
  context?: Record<string, unknown>;
}

/**
 * Shared base for the framework adapters in this folder.
 *
 * Subclasses call ``protected signAction`` on every framework callback.
 * They MUST NOT call the Asqav SDK directly so the fail-open contract is
 * preserved at one place.
 */
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

  /**
   * Sign a governance action. Fire-and-forget by default; errors route
   * through ``options.onError``. This method MUST NOT throw.
   */
  protected signAction(input: SignActionInput): void {
    const onError = this.options.onError ?? defaultOnError;

    if (this.options.observe) {
      // observe mode mirrors Python ``AsqavAdapter._observe``.
      // eslint-disable-next-line no-console
      console.info(
        `[asqav] OBSERVE: would sign ${input.actionType}`,
        input.context ?? {},
      );
      return;
    }

    // Fire-and-forget; never await so callbacks stay synchronous for
    // host frameworks that expect synchronous return values.
    void (async () => {
      try {
        const agent = await this.resolveAgent();
        await agent.sign({
          actionType: input.actionType,
          context: input.context,
        });
      } catch (err) {
        try {
          onError(err, { actionType: input.actionType });
        } catch {
          // Swallow secondary errors from the user-supplied handler.
        }
      }
    })();
  }
}

function defaultOnError(err: unknown, ctx: { actionType: string }): void {
  // eslint-disable-next-line no-console
  console.warn(`[asqav] sign failed for ${ctx.actionType}:`, err);
}

/**
 * Throw the canonical missing-peer error for an Asqav framework adapter.
 * Mirrors the Python ``ImportError`` contract. Optional ``cause`` keeps
 * the underlying module-resolution error (ERR_MODULE_NOT_FOUND, ESM/CJS
 * interop, version mismatch, native binding crash) visible to the user.
 */
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
