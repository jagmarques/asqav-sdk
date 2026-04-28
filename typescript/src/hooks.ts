/**
 * Hooks / middleware system for the Asqav TypeScript SDK.
 *
 * Mirrors the Python sibling's public surface:
 *   registerBefore(actionType, fn)  - mutate context before sign
 *   registerAfter(actionType, fn)   - observe response after sign
 *   clearHooks()                    - empty the registry
 *
 * Pattern matching:
 *   - "*"            matches every action type
 *   - "strands:*"    glob-matches any action type with the prefix "strands:"
 *   - "exact:type"   matches only that exact action type
 */

import type { SignatureResponse } from "./index.js";

export type BeforeHook = (
  actionType: string,
  context: Record<string, unknown>,
) => Record<string, unknown> | void;

export type AfterHook = (response: SignatureResponse) => void;

interface BeforeEntry {
  pattern: string;
  fn: BeforeHook;
}

interface AfterEntry {
  pattern: string;
  fn: AfterHook;
}

const beforeHooks: BeforeEntry[] = [];
const afterHooks: AfterEntry[] = [];

function patternMatches(pattern: string, actionType: string): boolean {
  if (pattern === "*") return true;
  if (pattern.endsWith(":*")) {
    const prefix = pattern.slice(0, -1); // keep the colon
    return actionType.startsWith(prefix);
  }
  if (pattern.endsWith("*")) {
    return actionType.startsWith(pattern.slice(0, -1));
  }
  return pattern === actionType;
}

export function registerBefore(actionType: string, fn: BeforeHook): void {
  beforeHooks.push({ pattern: actionType, fn });
}

export function registerAfter(actionType: string, fn: AfterHook): void {
  afterHooks.push({ pattern: actionType, fn });
}

export function clearHooks(): void {
  beforeHooks.length = 0;
  afterHooks.length = 0;
}

/** @internal */
export function _dispatchBefore(
  actionType: string,
  context: Record<string, unknown>,
): Record<string, unknown> {
  let current = context;
  for (const entry of beforeHooks) {
    if (!patternMatches(entry.pattern, actionType)) continue;
    try {
      const result = entry.fn(actionType, current);
      if (result && typeof result === "object") {
        current = result;
      }
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(
        `[asqav] before-hook for "${entry.pattern}" threw:`,
        err instanceof Error ? err.message : err,
      );
    }
  }
  return current;
}

/** @internal */
export function _dispatchAfter(
  actionType: string,
  response: SignatureResponse,
): void {
  for (const entry of afterHooks) {
    if (!patternMatches(entry.pattern, actionType)) continue;
    try {
      entry.fn(response);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(
        `[asqav] after-hook for "${entry.pattern}" threw:`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}
