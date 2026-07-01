/**
 * OpaDetector: POST {opaUrl}/v1/data/{policyPath} with {input:{action_type,context}};
 * {"result": true|false} maps to allow, missing result is deny. See docs/detector-plugins.
 */

import type { DetectorPlugin, DetectorResult } from "../detectors.js";

interface OpaDetectorOptions {
  /** Base URL of the OPA server. Default: http://localhost:8181. */
  opaUrl?: string;
  /** OPA data path. Default: asqav/action/allow. */
  policyPath?: string;
  /** Timeout for the OPA request in milliseconds. Default: 5000. */
  timeoutMs?: number;
}

export class OpaDetector implements DetectorPlugin {
  name = "opa";

  private readonly _opaUrl: string;
  private readonly _policyPath: string;
  private readonly _timeoutMs: number;

  constructor(options: OpaDetectorOptions = {}) {
    this._opaUrl = (options.opaUrl ?? "http://localhost:8181").replace(/\/$/, "");
    this._policyPath = (options.policyPath ?? "asqav/action/allow").replace(/^\//, "");
    this._timeoutMs = options.timeoutMs ?? 5000;
  }

  private get _endpoint(): string {
    return `${this._opaUrl}/v1/data/${this._policyPath}`;
  }

  async inspect(
    actionType: string,
    context: Record<string, unknown>,
  ): Promise<DetectorResult> {
    const payload = { input: { action_type: actionType, context } };

    let data: { result?: unknown };
    try {
      const controller = new AbortController();
      const tid = setTimeout(() => controller.abort(), this._timeoutMs);
      try {
        const resp = await fetch(this._endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        if (!resp.ok) {
          throw new Error(
            `opa_request_failed: OPA returned HTTP ${resp.status}`,
          );
        }
        data = await resp.json() as { result?: unknown };
      } finally {
        clearTimeout(tid);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`opa_request_failed: could not reach OPA at ${this._endpoint}: ${msg}`);
    }

    // Missing result key = fail-closed deny.
    const allow = data.result !== undefined ? Boolean(data.result) : false;

    if (allow) {
      return { allow: true, confidence: 1, labels: [], detector: this.name, reason: "" };
    }

    return {
      allow: false,
      confidence: 1,
      labels: ["opa_deny"],
      detector: this.name,
      reason: `OPA policy '${this._policyPath}' returned deny (result=${JSON.stringify(data.result)})`,
    };
  }
}
