/**
 * JCS canonicalization helper for the IETF Compliance Receipts profile.
 *
 * Public surface: `canonicalJson(obj): Uint8Array`.
 *
 * The output bytes are the exact bytes the cloud (`core/canonical.py`)
 * signs and the bytes the chain hashes over.
 *
 * See https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/
 *
 * Subset implemented:
 *   - Lexicographic (UTF-16 code unit) ordering of object keys.
 *   - No insignificant whitespace.
 *   - UTF-8 byte output, no BOM, non-ASCII passes through verbatim.
 *   - Standard JSON string escapes (backslash, quote, b, f, n, r, t,
 *     and `\u00xx` for U+0000..U+001F).
 *   - Numbers: integers within IEEE-754 safe range serialize without
 *     trailing ".0"; finite floats use the shortest round-trip form
 *     (V8's Number.toString already produces this, matching ECMAScript).
 *   - NaN / Infinity rejected (not representable in JSON).
 *
 * Out of scope: pre-2024 erratum normalization for non-finite numbers
 * and explicit handling of negative zero (we map -0 to "0", as
 * Python's json.dumps does, since the Compliance Receipts profile signs
 * no floats).
 *
 * The companion module `canonicalize.ts` exposes the same function under
 * the alias `canonicalize()`. They are kept byte-identical; this module
 * is the named landing for the IETF profile callers.
 */

export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

/**
 * Canonicalize a JSON-serializable value to JCS bytes.
 *
 * Throws TypeError for values not representable in JSON (functions,
 * undefined, BigInt, Symbol) and for non-finite numbers (NaN, Infinity,
 * -Infinity).
 */
export function canonicalJson(value: unknown): Uint8Array {
  return new TextEncoder().encode(canonicalString(value));
}

function canonicalString(value: unknown): string {
  if (value === null) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new TypeError("NaN / Infinity are not allowed in canonical JSON");
    }
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
    // Sort by UTF-16 code unit; JS default Array.prototype.sort on strings already does this.
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
  // Map -0 to 0 so equal mathematical values produce identical bytes.
  if (n === 0) return "0";
  // For safe integers, plain toString matches the canonical bytes byte-for-byte.
  if (Number.isInteger(n) && Number.isSafeInteger(n)) {
    return n.toString();
  }
  // V8's Number.toString already produces the shortest round-trip form;
  // unused by Compliance Receipts (no floats signed), kept for safety.
  return n.toString();
}

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
