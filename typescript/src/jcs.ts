/**
 * JCS canonicalization for the IETF Compliance Receipts profile; `canonicalJson(obj)` returns the exact
 * bytes the cloud signs and the chain hashes over. `canonicalize.ts` exposes the same bytes by alias.
 */

import { isRawFloatLeaf } from "./verifier/canonical.js";

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
  if (isRawFloatLeaf(value)) return floatLeafToCanonical(value.value);
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

/** Largest integer magnitude the Asqav profile admits (draft Section 4). */
export const MAX_PROFILE_INTEGER = 9007199254740991;

function numberToCanonical(n: number): string {
  // Map -0 to 0 so equal mathematical values produce identical bytes.
  if (n === 0) return "0";
  assertProfileRange(n);
  // For safe integers, plain toString matches the canonical bytes byte-for-byte.
  if (Number.isInteger(n) && Number.isSafeInteger(n)) {
    return n.toString();
  }
  // V8's Number.toString already produces the shortest round-trip form;
  // unused by Compliance Receipts (no floats signed), kept for safety.
  return n.toString();
}

function assertProfileRange(n: number): void {
  // The profile ends at 2**53 - 1, so 2**53 is refused though it round-trips
  // (a rounded 2**53+1 looks identical)
  if (Number.isInteger(n) && Math.abs(n) > MAX_PROFILE_INTEGER) {
    throw new RangeError(
      `integer outside the Asqav profile range +/-(2**53 - 1): ${n}; serialise ` +
        `it as a JSON string or an integer-rational pair`,
    );
  }
}

/**
 * A float literal kept by the float-preserving parser. json.dumps writes
 * repr(float) for it, so follow that, not the JavaScript number form.
 */
function floatLeafToCanonical(n: number): string {
  if (!Number.isFinite(n)) {
    throw new TypeError("NaN / Infinity are not allowed in canonical JSON");
  }
  assertProfileRange(n);
  if (n === 0) return Object.is(n, -0) ? "-0.0" : "0.0";
  return pythonFloatRepr(n);
}

/**
 * Python repr(float) of a finite nonzero double. Both languages pick the
 * shortest round-trip digits, so only the layout differs.
 */
function pythonFloatRepr(n: number): string {
  const [mantissa, exponent] = Math.abs(n).toExponential().split("e");
  const digits = mantissa.replace(".", "");
  const exp10 = Number(exponent);
  const sign = n < 0 ? "-" : "";
  // repr uses exponent form from 1e16 up and below 1e-4
  if (exp10 >= 16 || exp10 < -4) return sign + exponentLayout(digits, exp10);
  return sign + positionalLayout(digits, exp10 + 1);
}

function exponentLayout(digits: string, exp10: number): string {
  const head = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
  const magnitude = String(Math.abs(exp10)).padStart(2, "0");
  return `${head}e${exp10 < 0 ? "-" : "+"}${magnitude}`;
}

// point counts the digits left of the decimal point, 0 or less when none
function positionalLayout(digits: string, point: number): string {
  if (point <= 0) return `0.${"0".repeat(-point)}${digits}`;
  if (point >= digits.length) return `${digits}${"0".repeat(point - digits.length)}.0`;
  return `${digits.slice(0, point)}.${digits.slice(point)}`;
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
