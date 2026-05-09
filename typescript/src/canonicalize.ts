/**
 * Canonical JSON + content hashing for the @asqav/sdk TypeScript client.
 *
 * Mirrors the Python ``asqav.canonicalize`` module byte-for-byte. The
 * conformance vectors at ``conformance/vectors.json`` are validated against
 * both implementations, so a hash computed in Node and a hash computed in
 * Python over the same dict are guaranteed to agree.
 *
 * RFC 8785 (JCS) subset:
 *   - object keys sorted lexicographically
 *   - no whitespace anywhere
 *   - non-ASCII strings emitted as raw UTF-8
 *   - integers as integers (no trailing ``.0``)
 *   - finite floats only (NaN / Infinity rejected)
 */

import { createHash, createHmac } from "node:crypto";

export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

/**
 * Canonicalize a JSON-serializable value into UTF-8 bytes.
 *
 * Throws on values that aren't representable in JSON (functions, undefined,
 * BigInt, Symbol) and on non-finite numbers (NaN / Infinity), matching
 * the behavior of Python's ``json.dumps(allow_nan=False)``.
 */
export function canonicalize(value: unknown): Uint8Array {
  return new TextEncoder().encode(canonicalString(value));
}

function canonicalString(value: unknown): string {
  if (value === null) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error("NaN / Infinity are not allowed in canonical JSON");
    }
    // Shortest round-trip representation per the canonical JSON spec via ECMAScript.
    return numberToCanonical(value);
  }
  if (typeof value === "string") {
    return jsonString(value);
  }
  if (Array.isArray(value)) {
    return "[" + value.map(canonicalString).join(",") + "]";
  }
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const keys = Object.keys(obj).sort();
    const parts: string[] = [];
    for (const k of keys) {
      parts.push(jsonString(k) + ":" + canonicalString(obj[k]));
    }
    return "{" + parts.join(",") + "}";
  }
  throw new TypeError(`Not JSON-serializable: ${typeof value}`);
}

function numberToCanonical(n: number): string {
  // For integers within safe range, plain toString matches Python's repr.
  if (Number.isInteger(n) && Number.isSafeInteger(n)) {
    return n.toString();
  }
  return n.toString();
}

/**
 * RFC 8259 / RFC 8785 string serialization.
 *
 * Escapes control chars below U+0020, plus the two characters that JSON
 * requires (`"` and `\`). Everything U+0020 and above (including all
 * non-ASCII letters) is emitted as raw UTF-8. This matches Python's
 * ``json.dumps(ensure_ascii=False)`` which is what the conformance
 * vectors are generated with.
 */
function jsonString(s: string): string {
  let out = '"';
  for (let i = 0; i < s.length; i++) {
    const code = s.charCodeAt(i);
    const ch = s[i];
    if (ch === '"') {
      out += '\\"';
    } else if (ch === "\\") {
      out += "\\\\";
    } else if (code === 0x08) {
      out += "\\b";
    } else if (code === 0x09) {
      out += "\\t";
    } else if (code === 0x0a) {
      out += "\\n";
    } else if (code === 0x0c) {
      out += "\\f";
    } else if (code === 0x0d) {
      out += "\\r";
    } else if (code < 0x20) {
      out += "\\u" + code.toString(16).padStart(4, "0");
    } else {
      out += ch;
    }
  }
  out += '"';
  return out;
}

/** Canonical bytes of `{action_type, context}` for hashing or signing. */
export function canonicalizeAction(
  actionType: string,
  context: Record<string, unknown> = {},
): Uint8Array {
  return canonicalize({ action_type: actionType, context });
}

/**
 * Self-describing hash for an action: `sha256:<hex>`.
 * With `salt` set, uses HMAC-SHA-256.
 */
export async function hashAction(
  actionType: string,
  context: Record<string, unknown> = {},
  salt?: Uint8Array,
): Promise<string> {
  const canonical = canonicalizeAction(actionType, context);
  const hex = salt
    ? createHmac("sha256", Buffer.from(salt)).update(canonical).digest("hex")
    : createHash("sha256").update(canonical).digest("hex");
  return `sha256:${hex}`;
}
