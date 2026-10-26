import { vi } from "vitest";
import type { Agent } from "../../src/index.js";

export interface SignCall {
  actionType: string;
  context: Record<string, unknown>;
  result?: unknown;
}

/** Fake Asqav `Agent` whose `sign` records every call; no real client, no
 * network, no init(), so the tests stay hermetic */
export function makeFakeAgent(opts: { failWith?: Error } = {}) {
  const calls: SignCall[] = [];
  const sign = vi.fn(async (signOpts: { actionType: string; context?: Record<string, unknown>; result?: unknown }) => {
    calls.push({ actionType: signOpts.actionType, context: signOpts.context ?? {}, result: signOpts.result });
    if (opts.failWith) throw opts.failWith;
    return {
      signature: "sig-bytes",
      signatureId: "sig_1",
      actionId: "act_1",
      timestamp: "2026-01-01T00:00:00Z",
      verificationUrl: "https://asqav.com/verify/sig_1",
    };
  });
  const agent = { agentId: "agt_fake", sign } as unknown as Agent;
  return { agent, calls, sign };
}

export const tick = () => new Promise((r) => setTimeout(r, 0));
