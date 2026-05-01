/**
 * Client-side spend budget tracking for AI agent actions.
 *
 * Mirrors the Python BudgetTracker. Each recorded spend is signed via
 * the agent so the budget trail itself is tamper-evident. Enforcement
 * is the caller's responsibility: `check()` is a fail-closed arithmetic
 * check; the SDK does not block actions on its own.
 */

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

  /**
   * Decide whether a pending action with the given estimated cost is
   * within budget. Fail-closed on negative, NaN, or infinite costs.
   */
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

  /**
   * Sign a spend record on the underlying agent. Updates cumulative
   * spend after the signature is accepted by the server.
   */
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
    const sig = await this.agent.sign({
      actionType: `budget:${actionType}`,
      context: {
        ...context,
        cost,
        currency: this.currency,
        cumulative_spend: next,
        limit: this.limit,
      },
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
