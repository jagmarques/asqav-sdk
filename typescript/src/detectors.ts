/**
 * Pluggable detector framework (criterion 331): detectors run inside Agent.sign()
 * after normalization; a deny throws DetectorBlockedError. See docs/detector-plugins.
 */

/** Verdict returned by a DetectorPlugin's inspect() call. */
export interface DetectorResult {
  /** True = sign proceeds; False = sign is blocked. */
  allow: boolean;
  /** Confidence in [0, 1]. Informational. */
  confidence: number;
  /** Semantic tags that fired (entity types, policy IDs, etc). */
  labels: string[];
  /** Name of the detector that produced this result. */
  detector: string;
  /** Human-readable explanation for the decision. */
  reason: string;
}

/** Contract every detector must implement. */
export interface DetectorPlugin {
  /** Stable identifier; appears in the signed receipt _detectors record. */
  name: string;
  /** Inspect a pending sign action; return a DetectorResult (sync or Promise). */
  inspect(
    actionType: string,
    context: Record<string, unknown>,
  ): DetectorResult | Promise<DetectorResult>;
}

/** Thrown when a detector returns allow=false or raises with fail_open=false. */
export class DetectorBlockedError extends Error {
  /** Name of the detector that blocked the sign call. */
  detectorName: string;
  /** Full result from the blocking detector. */
  result: DetectorResult;
  /** Docs page for the detector-plugin feature. */
  docsUrl: string;

  constructor(message: string, result: DetectorResult) {
    super(message);
    this.name = "DetectorBlockedError";
    this.detectorName = result.detector;
    this.result = result;
    this.docsUrl = "https://asqav.com/docs/detector-plugins";
  }
}

type RegistryEntry = { detector: DetectorPlugin; failOpen: boolean };

let _registry: RegistryEntry[] = [];

/** Register a detector to run on every Agent.sign(). failOpen=true ignores
 * inspect() errors; default is fail-closed (an inspect error blocks sign). */
export function registerDetector(
  detector: DetectorPlugin,
  { failOpen = false }: { failOpen?: boolean } = {},
): void {
  _registry.push({ detector, failOpen });
}

/** Remove all registered detectors. Useful in tests. */
export function clearDetectors(): void {
  _registry = [];
}

/** Run all registered detectors; throw DetectorBlockedError on first deny.
 * Returns [] when none registered (callers keep context additive). */
export async function runDetectors(
  actionType: string,
  context: Record<string, unknown> | null | undefined,
): Promise<DetectorResult[]> {
  if (_registry.length === 0) return [];

  const ctx = context ?? {};
  const records: DetectorResult[] = [];

  for (const { detector, failOpen } of _registry) {
    let result: DetectorResult;
    try {
      result = await detector.inspect(actionType, ctx);
    } catch (err) {
      if (failOpen) {
        const msg = err instanceof Error ? err.message : String(err);
        console.warn(`[asqav] detector '${detector.name}' raised (fail-open): ${msg}`);
        records.push({
          allow: true,
          confidence: 0,
          labels: ["error"],
          detector: detector.name,
          reason: `inspector raised (fail-open): ${msg}`,
        });
        continue;
      }
      // Fail-closed: block sign().
      const msg = err instanceof Error ? err.message : String(err);
      const blockResult: DetectorResult = {
        allow: false,
        confidence: 0,
        labels: ["error"],
        detector: detector.name,
        reason: `inspector raised (fail-closed): ${msg}`,
      };
      throw new DetectorBlockedError(
        `detector_error_blocked: detector '${detector.name}' raised during inspect() ` +
          `and failOpen=false; sign() blocked. Error: ${msg}`,
        blockResult,
      );
    }

    // Back-fill detector name if the result omitted it.
    if (!result.detector) result = { ...result, detector: detector.name };
    records.push(result);

    if (!result.allow) {
      throw new DetectorBlockedError(
        `detector_blocked: detector '${result.detector}' denied the sign() call. ` +
          `reason='${result.reason}' labels=${JSON.stringify(result.labels)}`,
        result,
      );
    }
  }

  return records;
}
