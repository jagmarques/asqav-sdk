/**
 * Canonicalization shared across format adapters, a byte-for-byte port of the Python oracle's
 * `canonical.py`: `jcs`, `jcsRfc8785` and `asqavJcs` differ on key order, NFC and numbers.
 */

type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

/**
 * A JSON number written as a FLOAT literal, kept so `500.0` re-emits as `500.0` rather than the
 * `500` JSON.parse collapses it to; the jcs dialects sign numbers as the producer emitted them.
 */
export class RawFloat {
  constructor(public readonly value: number) {}

  // JSON.stringify re-emits the number itself, never the wrapper shape.
  toJSON(): number {
    return this.value;
  }
}

function isRawFloat(v: unknown): v is RawFloat {
  return v instanceof RawFloat;
}

/**
 * An integer beyond IEEE-754 safe range, kept as exact source digits. Collapsing it to the nearest
 * double would let two distinct integers share one signature and a tampered receipt verify.
 */
export class RawBigInt {
  constructor(public readonly source: string) {}

  // JSON.stringify collapses exactly like JSON.parse would (nearest double).
  toJSON(): number {
    return Number(this.source);
  }
}

function isRawBigInt(v: unknown): v is RawBigInt {
  return v instanceof RawBigInt;
}

/**
 * A JSON object repeated a member name (criterion 419). Last-wins would hash the bytes an attacker
 * kept, so the strict parser throws before any hashing, canonicalisation or signature check.
 */
export class DuplicateMemberError extends SyntaxError {
  constructor(message: string) {
    super(message);
    this.name = "DuplicateMemberError";
  }
}

/** ECMAScript-shortest decimal of a float, forced to carry a `.0` when whole. */
function floatToString(n: number): string {
  if (!Number.isFinite(n)) {
    throw new Error(`non-finite number not allowed in JCS: ${n}`);
  }
  let s = n.toString();
  // A whole-valued float must keep its decimal point (Python repr(500.0)=="500.0").
  if (!/[.eE]/.test(s)) s += ".0";
  return s;
}

/**
 * Parse JSON preserving float literals as `RawFloat` and out-of-safe-range integers as `RawBigInt`,
 * mirroring Python's `json.load`. Strict: a duplicated member at any depth throws (criterion 419).
 */
export function parseJsonPreservingFloats(text: string): unknown {
  let i = 0;
  const ws = () => {
    while (i < text.length && (text[i] === " " || text[i] === "\t" || text[i] === "\n" || text[i] === "\r")) i++;
  };
  const err = (m: string): never => {
    throw new SyntaxError(`${m} at position ${i}`);
  };
  let depth = 0;
  const value = (): unknown => {
    ws();
    const c = text[i];
    if (c === "{" || c === "[") {
      if (++depth > 1000) err("maximum nesting depth exceeded");
      const v = c === "{" ? object() : array();
      depth--;
      return v;
    }
    if (c === '"') return str();
    if (c === "-" || (c >= "0" && c <= "9")) return num();
    if (text.startsWith("true", i)) return (i += 4), true;
    if (text.startsWith("false", i)) return (i += 5), false;
    if (text.startsWith("null", i)) return (i += 4), null;
    return err(`unexpected token ${c ?? "EOF"}`);
  };
  const object = (): Record<string, unknown> => {
    const out: Record<string, unknown> = {};
    // Strict ingest (419): a repeated member name at any depth is terminal.
    const seen = new Set<string>();
    i++; // {
    ws();
    if (text[i] === "}") return i++, out;
    for (;;) {
      ws();
      if (text[i] !== '"') err("expected object key");
      const k = str();
      if (seen.has(k)) {
        throw new DuplicateMemberError(
          `duplicate JSON member name: ${JSON.stringify(k)} at position ${i}`,
        );
      }
      seen.add(k);
      ws();
      if (text[i++] !== ":") err("expected ':'");
      out[k] = value();
      ws();
      const ch = text[i++];
      if (ch === "}") return out;
      if (ch !== ",") err("expected ',' or '}'");
    }
  };
  const array = (): unknown[] => {
    const out: unknown[] = [];
    i++; // [
    ws();
    if (text[i] === "]") return i++, out;
    for (;;) {
      out.push(value());
      ws();
      const ch = text[i++];
      if (ch === "]") return out;
      if (ch !== ",") err("expected ',' or ']'");
    }
  };
  const str = (): string => {
    const start = i;
    i++; // opening quote
    while (i < text.length) {
      const c = text[i];
      if (c === '"') {
        i++;
        return JSON.parse(text.slice(start, i)) as string; // reuse the engine for escape handling
      }
      if (c === "\\") i += 2;
      else i++;
    }
    return err("unterminated string");
  };
  const digit = (): boolean => text[i] >= "0" && text[i] <= "9";
  const digits = (): void => {
    while (i < text.length && digit()) i++;
  };
  const num = (): RawFloat | RawBigInt | number => {
    // Strict RFC 8259 number grammar, matching Python json.loads (no leading zero,
    // a digit required after '.' and after the exponent marker, no bare '-').
    const start = i;
    if (text[i] === "-") i++;
    if (text[i] === "0") i++;
    else if (text[i] >= "1" && text[i] <= "9") {
      i++;
      digits();
    } else err("invalid number");
    let isFloat = false;
    if (text[i] === ".") {
      isFloat = true;
      i++;
      if (!digit()) err("digit expected after decimal point");
      digits();
    }
    if (text[i] === "e" || text[i] === "E") {
      isFloat = true;
      i++;
      if (text[i] === "+" || text[i] === "-") i++;
      if (!digit()) err("digit expected in exponent");
      digits();
    }
    const lexeme = text.slice(start, i);
    const parsed = Number(lexeme);
    if (isFloat) return new RawFloat(parsed);
    if (!Number.isSafeInteger(parsed)) return new RawBigInt(lexeme);
    return parsed;
  };
  const result = value();
  ws();
  if (i !== text.length) err("trailing content after JSON value");
  return result;
}

/**
 * Strip the `RawFloat` / `RawBigInt` wrappers back to the exact shapes `JSON.parse` produces.
 * Canonicalising callers keep the wrappers; callers needing the plain wire shape unwrap them.
 */
export function unwrapPreservedFloats(value: unknown): unknown {
  if (isRawFloat(value)) return value.value;
  if (isRawBigInt(value)) return Number(value.source);
  if (Array.isArray(value)) return value.map(unwrapPreservedFloats);
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = unwrapPreservedFloats(v);
    }
    return out;
  }
  return value;
}

/**
 * Strict ingest with `JSON.parse`-compatible values (criterion 419): a duplicated member name at any
 * depth throws. Replaces `JSON.parse` wherever the float-preserving dialects are not needed.
 */
export function parseJsonStrict(text: string): unknown {
  return unwrapPreservedFloats(parseJsonPreservingFloats(text));
}

/** Recursively NFC-normalise every string key and value (mirrors `_nfc`). */
function nfc(obj: unknown): unknown {
  if (typeof obj === "string") return obj.normalize("NFC");
  if (isRawFloat(obj) || isRawBigInt(obj)) return obj;
  if (Array.isArray(obj)) return obj.map(nfc);
  if (obj !== null && typeof obj === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
      out[(k as string).normalize("NFC")] = nfc(v);
    }
    return out;
  }
  return obj;
}

/**
 * Compare two strings by Unicode CODE POINT (Python `sort_keys` ordering). JS string `<` compares by
 * UTF-16 code unit, which differs from code point on supplementary-plane characters.
 */
function compareCodePoints(a: string, b: string): number {
  const ai = a[Symbol.iterator]();
  const bi = b[Symbol.iterator]();
  for (;;) {
    const an = ai.next();
    const bn = bi.next();
    if (an.done && bn.done) return 0;
    if (an.done) return -1;
    if (bn.done) return 1;
    const ac = an.value.codePointAt(0)!;
    const bc = bn.value.codePointAt(0)!;
    if (ac !== bc) return ac < bc ? -1 : 1;
  }
}

/**
 * JSON string serialization matching Python `json.dumps(ensure_ascii=False)`: short-escapes the five
 * C0 shorthands, other control chars as \u00xx, everything else verbatim. `/` is not escaped.
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

/** ECMAScript Number.prototype.toString form (RFC 8785 number rule). */
function numberToString(n: number): string {
  if (!Number.isFinite(n)) {
    throw new Error(`non-finite number not allowed in JCS: ${n}`);
  }
  // Map -0 to "0", as both Python json.dumps and ECMAScript do for the corpus.
  if (n === 0) return "0";
  // V8's Number.toString is already the ECMAScript shortest round-trip form,
  // which is exactly the RFC 8785 number serialisation (verified byte-equal).
  return n.toString();
}

/**
 * Serialise with a chosen key-comparison function. `honorFloat` picks the number dialect: true emits
 * the float form `500.0`, false collapses a whole-valued float to `500` (strict RFC 8785).
 */
function serialize(
  value: unknown,
  keyCompare: (a: string, b: string) => number,
  honorFloat: boolean,
): string {
  if (value === null) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (isRawFloat(value)) return honorFloat ? floatToString(value.value) : numberToString(value.value);
  // A >2^53 integer: emit exact source digits in every dialect (Python str(int)
  // parity) so distinct ints stay distinct.
  if (isRawBigInt(value)) return value.source;
  if (typeof value === "number") return numberToString(value);
  if (typeof value === "string") return jsonString(value);
  if (Array.isArray(value)) {
    return "[" + value.map((v) => serialize(v, keyCompare, honorFloat)).join(",") + "]";
  }
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const keys = Object.keys(obj).sort(keyCompare);
    const parts: string[] = [];
    for (const k of keys) {
      parts.push(jsonString(k) + ":" + serialize(obj[k], keyCompare, honorFloat));
    }
    return "{" + parts.join(",") + "}";
  }
  throw new TypeError(`unserialisable type in JCS input: ${typeof value}`);
}

const utf16CompareKeys = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/** AERF/ACTA-dialect JCS: NFC, code-point key sort, numbers as the producer emits them. */
export function jcs(obj: unknown): Uint8Array {
  return new TextEncoder().encode(serialize(nfc(obj), compareCodePoints, true));
}

/**
 * Asqav cloud JCS bytes (no NFC, producer numbers), byte-identical to the cloud signer: member names in
 * UTF-16 code-unit order per RFC 8785 section 3.2.3, which JS string comparison already gives.
 */
export function asqavJcs(obj: unknown): Uint8Array {
  return new TextEncoder().encode(serialize(obj, utf16CompareKeys, true));
}

/**
 * Instant from which the issuing platform emits RFC 8785 member order on the wire. Pinned to the
 * production deploy of the emitter change; receipts issued later never get the pre-cutover retry.
 */
export const JCS_UTF16_CUTOVER = "2026-09-02T12:00:00+00:00";

/**
 * The code-point member order the cloud emitted before JCS_UTF16_CUTOVER. Diagnostic only: a signature
 * that verifies solely under these bytes is reported as the pre-cutover dialect, never as verified.
 */
export function asqavJcsPreCutover(obj: unknown): Uint8Array {
  return new TextEncoder().encode(serialize(obj, compareCodePoints, true));
}

/** True when any object member name, at any depth, carries a character above U+FFFF. */
export function hasSupplementaryMemberName(obj: unknown): boolean {
  const stack: unknown[] = [obj];
  while (stack.length > 0) {
    const node = stack.pop();
    if (Array.isArray(node)) {
      for (const v of node) stack.push(v);
    } else if (node !== null && typeof node === "object") {
      for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
        if (/[\uD800-\uDBFF]/.test(k)) return true;
        stack.push(v);
      }
    }
  }
  return false;
}

/** Strict RFC 8785 JCS bytes (UTF-16 code-unit key sort, ECMAScript numbers) with NFC. */
export function jcsRfc8785(obj: unknown): Uint8Array {
  return new TextEncoder().encode(serialize(nfc(obj), utf16CompareKeys, false));
}

export type { JsonValue };
