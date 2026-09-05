/**
 * Shared User-Agent for outbound SDK calls: the api.asqav.com edge 403s anonymous default agents. Only
 * added under Node, since browsers forbid setting User-Agent on fetch.
 */

export const SDK_VERSION = "0.10.9";

export const USER_AGENT = `asqav-js/${SDK_VERSION} (+https://www.asqav.com)`;

/** Headers to merge into fetch calls: `{ "User-Agent": ... }` in Node, `{}` in browsers. */
export function userAgentHeaders(): Record<string, string> {
  try {
    if (typeof process !== "undefined" && process.versions?.node) {
      return { "User-Agent": USER_AGENT };
    }
  } catch {
    // Some sandboxes throw on `process` access; treat as browser.
  }
  return {};
}
