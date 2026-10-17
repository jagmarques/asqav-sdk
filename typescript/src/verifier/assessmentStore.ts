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
  // Why a reassessment exists (488 C10); originals keep null
  readonly reason: string | null;
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

// The reason rides outside the chain like the Python twin's label:
// old logs hashed body without it, so chain hashes never move
function hashBody(entry: Omit<AssessmentEntry, "entryHash">): string {
  const { reason: _dropped, ...body } = entry;
  return hashEntry(body as Omit<AssessmentEntry, "entryHash">);
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
    if (this.replay(docSha256) !== null) {
      throw new Error(`duplicate original for ${docSha256.slice(0, 16)}; reassess instead`);
    }
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
      reason: null,
      prevHash: prev,
    };
    const entry: AssessmentEntry = { ...body, entryHash: hashBody(body) };
    this.entries.push(entry);
    return entry;
  }

  replay(docSha256: string): AssessmentEntry | null {
    return this.entries.find((e) => e.docSha256 === docSha256 && e.kind === "original") ?? null;
  }

  current(docSha256: string): AssessmentEntry | null {
    // Latest assessment wins; replay() stays the contemporaneous original
    let seen: AssessmentEntry | null = null;
    for (const e of this.entries) {
      if (e.docSha256 === docSha256 && (e.kind === "original" || e.kind === "reassessment")) seen = e;
    }
    return seen;
  }

  reassess(
    docSha256: string,
    context: Record<string, unknown>,
    axes: Record<string, string>,
    verdict: string,
    failureClass: string | null,
    recordedAt: string | null = null,
    reason: string | null = null,
  ): AssessmentEntry {
    const original = this.replay(docSha256);
    if (original === null) {
      throw new Error(`no retained original for ${docSha256.slice(0, 16)}; record first`);
    }
    const prev = this.entries.length > 0 ? this.entries[this.entries.length - 1].entryHash : GENESIS;
    const body = {
      docSha256,
      context: { ...context },
      axes: { ...axes },
      verdict,
      failureClass,
      recordedAt,
      kind: "reassessment",
      supersedes: original.entryHash,
      reason,
      prevHash: prev,
    };
    const entry: AssessmentEntry = { ...body, entryHash: hashBody(body) };
    this.entries.push(entry);
    return entry;
  }

  verifyChain(): boolean {
    let prev = GENESIS;
    for (const e of this.entries) {
      if (e.prevHash !== prev) return false;
      const { entryHash: _drop, ...body } = e;
      if (hashBody(body as Omit<AssessmentEntry, "entryHash">) !== e.entryHash) return false;
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
      if (line.trim() === "") continue;
      const parsed = JSON.parse(line) as AssessmentEntry;
      store.entries.push({ ...parsed, reason: parsed.reason ?? null });
    }
    if (!store.verifyChain()) throw new Error("assessment log chain broken");
    const originals = new Map<string, string>();
    for (const e of store.entries) {
      if (e.kind === "original") {
        if (e.supersedes !== null) throw new Error("original must not supersede");
        if (originals.has(e.docSha256)) throw new Error("duplicate original in restored log");
        originals.set(e.docSha256, e.entryHash);
      } else if (e.kind !== "reassessment") {
        throw new Error(`unknown entry kind ${JSON.stringify(e.kind)}`);
      }
    }
    for (const e of store.entries) {
      if (e.kind === "reassessment" && originals.get(e.docSha256) !== e.supersedes) {
        throw new Error("reassessment links a missing original");
      }
    }
    return store;
  }
}
