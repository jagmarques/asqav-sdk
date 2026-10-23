/**
 * PresidioDetector: wraps the Presidio REST analyzer; posts stringified context and
 * maps detected entities to a DetectorResult. Docs: microsoft.github.io/presidio.
 */

import type { DetectorPlugin, DetectorResult } from "../detectors.js";

interface PresidioAnalyzerResult {
  entity_type: string;
  score: number;
  start: number;
  end: number;
}

interface PresidioDetectorOptions {
  /** Presidio Analyzer REST endpoint. Default: http://localhost:5002. */
  url?: string;
  /** Language code. Default: "en". */
  language?: string;
  /** Block when any entity meets or exceeds this score. Default: 0.5. */
  threshold?: number;
  /** Entity types to scan. Omit to scan all. */
  entities?: string[];
}

export class PresidioDetector implements DetectorPlugin {
  name = "presidio";

  private readonly _url: string;
  private readonly _language: string;
  private readonly _threshold: number;
  private readonly _entities: string[] | undefined;

  constructor(options: PresidioDetectorOptions = {}) {
    this._url = (options.url ?? "http://localhost:5002").replace(/\/$/, "");
    this._language = options.language ?? "en";
    this._threshold = options.threshold ?? 0.5;
    this._entities = options.entities;
  }

  async inspect(
    _actionType: string,
    context: Record<string, unknown>,
  ): Promise<DetectorResult> {
    const text = JSON.stringify(context);
    const payload: Record<string, unknown> = {
      text,
      language: this._language,
    };
    if (this._entities) payload.entities = this._entities;

    const resp = await fetch(`${this._url}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      throw new Error(
        `presidio_request_failed: analyzer returned HTTP ${resp.status}`,
      );
    }

    const results = (await resp.json()) as PresidioAnalyzerResult[];
    const hits = results.filter((r) => r.score >= this._threshold);

    if (hits.length === 0) {
      return { allow: true, confidence: 1, labels: [], detector: this.name, reason: "" };
    }

    const labels = [...new Set(hits.map((r) => r.entity_type))].sort();
    const maxScore = Math.max(...hits.map((r) => r.score));

    return {
      allow: false,
      confidence: maxScore,
      labels,
      detector: this.name,
      reason: `PII detected: ${labels.join(", ")} (max score ${maxScore.toFixed(2)}, threshold ${this._threshold})`,
    };
  }
}
