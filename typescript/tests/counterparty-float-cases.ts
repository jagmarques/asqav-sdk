// Loader for verifier/counterparty-float-cases.json, the shared float table
// Its expected hashes are Python's compute_envelope_hash output

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

export interface FloatCase {
  name: string;
  payload_text?: string;
  nest?: "object" | "array";
  wrappers?: number;
  leaf?: string;
  envelope_hash: string | null;
  error: string | null;
}

interface FloatTable {
  signature: { alg: string; kid: string; sig: string };
  cases: FloatCase[];
  float_reprs: Array<[string, string]>;
}

export const FLOAT_TABLE = JSON.parse(
  readFileSync(resolve(__dirname, "..", "..", "verifier", "counterparty-float-cases.json"), "utf-8"),
) as FloatTable;

/** The payload JSON text of one case, nested to its depth when it has one */
export function payloadText(c: FloatCase): string {
  if (c.payload_text !== undefined) return c.payload_text;
  const w = c.wrappers as number;
  const leaf = c.leaf as string;
  const inner =
    c.nest === "object"
      ? '{"n":'.repeat(w) + `{"leaf":${leaf}}` + "}".repeat(w)
      : "[".repeat(w) + leaf + "]".repeat(w);
  return `{"type":"protectmcp:decision","junk":${inner}}`;
}

/** The originating envelope as JSON text for a float-preserving parser */
export function envelopeText(c: FloatCase): string {
  return `{"payload":${payloadText(c)},"signature":${JSON.stringify(FLOAT_TABLE.signature)}}`;
}

/** An acknowledgment whose binding claims `envelopeHash` over the originator */
export function acknowledgment(envelopeHash: string): Record<string, unknown> {
  return {
    payload: {
      counterparty_binding: {
        scope: "envelope_minus_anchors",
        receipt_ref: "r1",
        envelope_hash: envelopeHash,
      },
    },
    signature: { alg: "ML-DSA-65", kid: "kid-B", sig: "BBBB" },
  };
}
