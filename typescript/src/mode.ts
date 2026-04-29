/**
 * Mode resolver for hash-only vs full-payload signing.
 *
 * Mirrors ``asqav._mode`` in the Python SDK exactly: same precedence
 * (explicit > env > URL auto-detection) and same "is this an asqav cloud
 * URL?" rule.
 */

export type Mode = "hash-only" | "full-payload";

const VALID_EXPLICIT = new Set<string>(["auto", "hash-only", "full-payload"]);
const VALID_ENV = new Set<string>(["hash-only", "full-payload"]);

/** True if ``hostname`` is the asqav cloud API. Subdomain-attack-safe. */
export function isAsqavCloudHost(hostname: string | null | undefined): boolean {
  if (!hostname) return false;
  const h = hostname.toLowerCase().trim().replace(/\.+$/, "");
  if (h === "asqav.com") return false; // marketing site
  if (h === "api.asqav.com") return true;
  return h.endsWith(".asqav.com");
}

/**
 * Resolve the wire mode for a client.
 *
 * @param apiBaseUrl  The configured API URL.
 * @param env         Value of ASQAV_MODE env var (or null).
 * @param explicit    Caller-passed mode. ``"auto"`` defers to env then URL.
 */
export function resolveMode(
  apiBaseUrl: string | null | undefined,
  env: string | null | undefined,
  explicit: string | null | undefined = "auto",
): Mode {
  const exp = (explicit ?? "auto").trim();
  if (!VALID_EXPLICIT.has(exp)) {
    throw new Error(
      `mode must be one of auto / hash-only / full-payload, got ${JSON.stringify(exp)}`,
    );
  }
  if (exp !== "auto") return exp as Mode;

  if (env) {
    const e = env.trim().toLowerCase();
    if (VALID_ENV.has(e)) return e as Mode;
    // garbage env value -> ignore, fall through
  }

  let hostname: string | null = null;
  if (apiBaseUrl) {
    try {
      const withScheme = /:\/\//.test(apiBaseUrl) ? apiBaseUrl : `https://${apiBaseUrl}`;
      hostname = new URL(withScheme).hostname;
    } catch {
      hostname = null;
    }
  }
  return isAsqavCloudHost(hostname) ? "hash-only" : "full-payload";
}
