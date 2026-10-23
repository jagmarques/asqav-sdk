// Client-side spend budget tracking mirroring the Python BudgetTracker; each
// spend is signed so the trail is tamper-evident

import type { Agent, SignatureResponse } from "./index.js";

export interface BudgetCheckResult {
  allowed: boolean;
  currentSpend: number;
  limit: number;
  remaining: number;
  reason?: "invalid_cost" | "budget_exhausted";
}

export interface BudgetTrackerOptions {
  agent: Agent;
  limit: number;
  currency?: string;
}

export class BudgetTracker {
  readonly agent: Agent;
  readonly limit: number;
  readonly currency: string;
  private spend = 0;
  private records: string[] = [];

  constructor(options: BudgetTrackerOptions) {
    if (!isFinite(options.limit) || options.limit < 0) {
      throw new Error("budget limit must be a finite non-negative number");
    }
    this.agent = options.agent;
    this.limit = options.limit;
    this.currency = options.currency ?? "USD";
  }

  // Fail-closed on negative, NaN, or infinite costs; enforcement is the
  // caller's: this is arithmetic only and the SDK blocks nothing
  /** Decide whether a pending action's estimated cost is within budget */
  check(estimatedCost: number): BudgetCheckResult {
    const remaining = this.limit - this.spend;
    const cost = Number(estimatedCost);
    if (!Number.isFinite(cost) || cost < 0) {
      return {
        allowed: false,
        currentSpend: this.spend,
        limit: this.limit,
        remaining,
        reason: "invalid_cost",
      };
    }
    if (cost > remaining) {
      return {
        allowed: false,
        currentSpend: this.spend,
        limit: this.limit,
        remaining,
        reason: "budget_exhausted",
      };
    }
    return {
      allowed: true,
      currentSpend: this.spend,
      limit: this.limit,
      remaining: remaining - cost,
    };
  }

  // Updates cumulative spend after the signature is accepted by the server
  /** Sign a spend record on the underlying agent */
  async record(
    actionType: string,
    actualCost: number,
    context: Record<string, unknown> = {},
  ): Promise<SignatureResponse> {
    const cost = Number(actualCost);
    if (!Number.isFinite(cost) || cost < 0) {
      throw new Error("actualCost must be a finite non-negative number");
    }
    const next = this.spend + cost;
    const outcome: Record<string, unknown> = {
      ...context,
      cost,
      currency: this.currency,
      cumulative_spend: next,
      limit: this.limit,
    };
    const sig = await this.agent.sign({
      actionType: `budget:${actionType}`,
      context: outcome,
      result: outcome,
    });
    this.spend = next;
    this.records.push(sig.signatureId);
    return sig;
  }

  status(): {
    limit: number;
    currency: string;
    spend: number;
    remaining: number;
    records: number;
    signatureIds: string[];
  } {
    return {
      limit: this.limit,
      currency: this.currency,
      spend: this.spend,
      remaining: this.limit - this.spend,
      records: this.records.length,
      signatureIds: [...this.records],
    };
  }
}
