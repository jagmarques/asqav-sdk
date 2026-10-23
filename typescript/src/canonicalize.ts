// Canonical JSON + content hashing, as Python asqav.canonicalize byte-for-byte:
// sorted keys, no whitespace, raw UTF-8, ints as ints, finite floats only

import { createHash, createHmac } from "node:crypto";

type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

// Throws on values JSON cannot represent and on non-finite numbers, as Python's
// ``json.dumps(allow_nan=False)``
/** Canonicalize a JSON-serializable value into UTF-8 bytes */
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
    // Integers serialize without a trailing .0; floats use V8's shortest
    // round-trip
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
  // For integers within safe range, plain toString matches Python's repr
  if (Number.isInteger(n) && Number.isSafeInteger(n)) {
    return n.toString();
  }
  return n.toString();
}

// Escapes control chars below U+0020 plus `"` and `\`, everything else raw
// UTF-8; matches the ``ensure_ascii=False`` form the conformance vectors use
/** Standard JSON string serialization of control chars, quote and backslash */
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

function escapeJsonPointerToken(token: string): string {
  return token.replace(/~/g, "~0").replace(/\//g, "~1");
}

function findFloatPointer(value: unknown, pointer: string): string | null {
  if (typeof value === "number" && !Number.isInteger(value)) {
    return pointer;
  }
  if (typeof value === "number" && !Number.isFinite(value)) {
    return pointer;
  }
  if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) {
      const hit = findFloatPointer(value[i], `${pointer}/${i}`);
      if (hit !== null) return hit;
    }
    return null;
  }
  if (value !== null && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    for (const key of Object.keys(obj)) {
      const hit = findFloatPointer(obj[key], `${pointer}/${escapeJsonPointerToken(key)}`);
      if (hit !== null) return hit;
    }
    return null;
  }
  return null;
}

// Floats are not byte-stable across runtimes, so a non-integer leaf throws with
// a JSON pointer to it; serialize numerics as strings
/** Pre-validate ``tool_args`` and return JCS-canonical bytes */
export function canonicalizeToolArgs(args: unknown): Uint8Array {
  const pointer = findFloatPointer(args, "");
  if (pointer !== null) {
    const where = pointer === "" ? "/" : pointer;
    throw new Error(
      `tool_args float at ${where}: floats are not byte-stable across runtimes `
      + "per RFC 8785 section 3.2.2; serialize numerics as strings or rationals "
      + "before passing to canonicalizeToolArgs",
    );
  }
  return canonicalize(args);
}

// The prefix reads `sha256` either way, so a verifier cannot tell a salted
// digest from a plain one
/** Self-describing hash for an action: `sha256:<hex>`, or HMAC with `salt` */
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
