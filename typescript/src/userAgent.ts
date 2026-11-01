/** Shared User-Agent for outbound SDK calls: the api.asqav.com edge 403s
 * anonymous default agents. Node only: browsers forbid setting it on fetch */

export const SDK_VERSION = "0.10.14";

export const USER_AGENT = `asqav-js/${SDK_VERSION} (+https://www.asqav.com)`;

/** Fetch headers to merge: `{ "User-Agent": ... }` in Node, `{}` in browsers */
export function userAgentHeaders(): Record<string, string> {
  try {
    if (typeof process !== "undefined" && process.versions?.node) {
      return { "User-Agent": USER_AGENT };
    }
  } catch {
    // Some sandboxes throw on `process` access; treat as browser
  }
  return {};
}
