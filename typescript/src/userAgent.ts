/**
 * Shared User-Agent for every outbound HTTP call the SDK makes from Node.
 *
 * The api.asqav.com edge runs a Cloudflare browser-integrity check that
 * 403s anonymous default agents, so SDK requests must identify themselves.
 * Browsers forbid setting User-Agent on fetch (it is a forbidden header
 * name), so the helper only adds it when running under Node.
 */

export const SDK_VERSION = "0.5.15";

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
