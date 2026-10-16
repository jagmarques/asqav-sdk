/** Retained assessment inputs with reassessment labelling (488 C10). */

import { createHash } from "node:crypto";

export interface AssessmentEntry {
  readonly docSha256: string;
  readonly context: Record<string, unknown>;
  readonly axes: Record<string, string>;
  readonly verdict: string;
  readonly failureClass: string | null;
  readonly recordedAt: string | null;
  readonly kind: string;
  readonly supersedes: string | null;
  readonly prevHash: string;
  readonly entryHash: string;
}

function canonical(value: unknown): string {
  return JSON.stringify(sortDeep(value));
}

function sortDeep(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortDeep);
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(value).sort()) {
      out[k] = sortDeep((value as Record<string, unknown>)[k]);
    }
    return out;
  }
  return value;
}

function hashEntry(entry: Omit<AssessmentEntry, "entryHash">): string {
  return createHash("sha256").update(canonical(entry), "utf-8").digest("hex");
}

const GENESIS = "GENESIS";

export class AssessmentStore {
  private entries: AssessmentEntry[] = [];

  get size(): number {
    return this.entries.length;
  }

  record(
    docSha256: string,
    context: Record<string, unknown>,
    axes: Record<string, string>,
    verdict: string,
    failureClass: string | null,
    recordedAt: string | null = null,
  ): AssessmentEntry {
    const prev = this.entries.length > 0 ? this.entries[this.entries.length - 1].entryHash : GENESIS;
    const body = {
      docSha256,
      context: { ...context },
      axes: { ...axes },
      verdict,
      failureClass,
      recordedAt,
      kind: "original",
      supersedes: null,
      prevHash: prev,
    };
    const entry: AssessmentEntry = { ...body, entryHash: hashEntry(body) };
    this.entries.push(entry);
    return entry;
  }

  replay(docSha256: string): AssessmentEntry | null {
    return this.entries.find((e) => e.docSha256 === docSha256 && e.kind === "original") ?? null;
  }

  reassess(
    docSha256: string,
    context: Record<string, unknown>,
    axes: Record<string, string>,
    verdict: string,
    failureClass: string | null,
    recordedAt: string | null = null,
  ): AssessmentEntry {
    const original = this.replay(docSha256);
    const prev = this.entries.length > 0 ? this.entries[this.entries.length - 1].entryHash : GENESIS;
    const body = {
      docSha256,
      context: { ...context },
      axes: { ...axes },
      verdict,
      failureClass,
      recordedAt,
      kind: "reassessment",
      supersedes: original !== null ? original.entryHash : null,
      prevHash: prev,
    };
    const entry: AssessmentEntry = { ...body, entryHash: hashEntry(body) };
    this.entries.push(entry);
    return entry;
  }

  verifyChain(): boolean {
    let prev = GENESIS;
    for (const e of this.entries) {
      if (e.prevHash !== prev) return false;
      const { entryHash: _drop, ...body } = e;
      if (hashEntry(body as Omit<AssessmentEntry, "entryHash">) !== e.entryHash) return false;
      prev = e.entryHash;
    }
    return true;
  }

  toJsonl(): string {
    return this.entries.map((e) => JSON.stringify(sortDeep(e))).join("\n");
  }

  static fromJsonl(text: string): AssessmentStore {
    const store = new AssessmentStore();
    for (const line of text.split("\n")) {
      if (line.trim() !== "") store.entries.push(JSON.parse(line) as AssessmentEntry);
    }
    if (!store.verifyChain()) throw new Error("assessment log chain broken");
    return store;
  }
}
