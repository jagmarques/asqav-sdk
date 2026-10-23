// Shared User-Agent for outbound SDK calls: api.asqav.com edge 403s anonymous
// default agents; added only under Node, browsers forbid setting it on fetch

export const SDK_VERSION = "0.10.14";

export const USER_AGENT = `asqav-js/${SDK_VERSION} (+https://www.asqav.com)`;

/** Headers to merge into fetch: `{ "User-Agent": ... }` in Node, `{}` else */
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
