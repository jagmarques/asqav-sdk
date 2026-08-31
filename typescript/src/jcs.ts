/**
 * JCS canonicalization for the IETF Compliance Receipts profile; `canonicalJson(obj)` returns the exact
 * bytes the cloud signs and the chain hashes over. `canonicalize.ts` exposes the same bytes by alias.
 */

type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

/**
 * Canonicalize a JSON-serializable value to JCS bytes. Throws TypeError for values JSON cannot
 * represent (functions, undefined, BigInt, Symbol) and for non-finite numbers.
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
