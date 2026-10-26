// Base adapter for Asqav framework integrations, mirroring Python extras
// Provides post-execution signAction and fail-closed signActionOrThrow

import { Agent, mapPolicyDecisionToDecision } from "../index.js";

export { safeStringify, toJsonSafeResult } from "../jcs.js";

// Options for every adapter. Supply one of agent, agentId or agentName
export interface AsqavAdapterOptions {
  /** Pre-built Asqav Agent. Takes precedence over agentId/agentName. */
  agent?: Agent;
  /** ID of an existing Asqav agent. Resolved lazily via Agent.get. */
  agentId?: string;
  /** Name for a new Asqav agent. Resolved lazily via Agent.create. */
  agentName?: string;
  /** Optional error sink; pre-execution gates still throw after reporting. */
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
  /** Completion value; its digest is default-sent (787). */
  result?: unknown;
}

export type AsqavErrorReporter = (err: unknown, context: { actionType: string }) => void;

/** A pre-execution gate refused the action; the wrapped tool must not run. */
export class AsqavRefusedError extends Error {
  readonly actionType: string;

  constructor(actionType: string, reason: string, cause?: unknown) {
    super(`[asqav] refused ${actionType}: ${reason}`, cause === undefined ? undefined : { cause });
    this.name = "AsqavRefusedError";
    this.actionType = actionType;
  }
}

interface SignVerdictLike {
  signature?: unknown;
  signatureId?: unknown;
  policyDecision?: string;
  decision?: string;
}

// Throw unless verdict is a signed receipt that permits the action. A thrown or
// rejected sign, a missing signature, or a deny verdict all count as refused
export function assertSignedVerdict(verdict: unknown, actionType: string): void {
  if (!verdict || typeof verdict !== "object") {
    throw new AsqavRefusedError(actionType, "sign returned no verdict");
  }
  const v = verdict as SignVerdictLike;
  const sig = v.signature;
  const hasSignature = typeof sig === "string"
    ? sig.length > 0
    : Boolean(sig && typeof sig === "object" && Object.keys(sig as object).length > 0);
  if (!hasSignature || typeof v.signatureId !== "string" || v.signatureId.length === 0) {
    throw new AsqavRefusedError(actionType, "sign did not return a signed receipt");
  }
  const rawDecision = v.decision ?? v.policyDecision;
  if (rawDecision !== undefined) {
    const decision = mapPolicyDecisionToDecision(rawDecision);
    if (decision !== "allow") {
      throw new AsqavRefusedError(actionType, `policy ${decision}`);
    }
  }
}

// Run one sign and throw unless the verdict permits; report the failure first
// A handler that throws cannot hide the refusal it was told about
export async function signOrThrow(
  input: SignActionInput,
  sign: () => Promise<unknown>,
  onError: AsqavErrorReporter = defaultOnError,
): Promise<void> {
  try {
    assertSignedVerdict(await sign(), input.actionType);
  } catch (err) {
    try {
      onError(err, { actionType: input.actionType });
    } catch (handlerErr) {
      // A failing error sink must not replace the refusal it reported
      console.warn("[asqav] onError handler failed:", handlerErr);
    }
    throw err instanceof AsqavRefusedError
      ? err
      : new AsqavRefusedError(input.actionType, err instanceof Error ? err.message : String(err), err);
  }
}

// Shared base for the framework adapters. Subclasses use signAction for
// telemetry; signActionOrThrow gates a tool before it runs
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
          result: input.result as never,
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

  // Sign before a tool runs. Unless observe is set, resolves only for a
  // signed, permitting verdict; a failure is reported and then thrown
  protected async signActionOrThrow(input: SignActionInput): Promise<void> {
    if (this.options.observe) {
      console.info(`[asqav] OBSERVE: would sign ${input.actionType}`, input.context ?? {});
      return;
    }

    await signOrThrow(
      input,
      async () => {
        const agent = await this.resolveAgent();
        return agent.sign({
          actionType: input.actionType,
          context: input.context,
          result: input.result as never,
        });
      },
      this.options.onError ?? defaultOnError,
    );
  }
}

function defaultOnError(err: unknown, ctx: { actionType: string }): void {
  // eslint-disable-next-line no-console
  console.warn(`[asqav] sign failed for ${ctx.actionType}:`, err);
}

/**
 * Throw the canonical missing-peer error, mirroring the Python ``ImportError`` contract. ``cause`` keeps
 * the underlying module-resolution error visible to the user.
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
