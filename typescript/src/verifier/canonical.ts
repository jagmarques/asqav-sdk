/**
 * Canonicalization shared across format adapters - a byte-for-byte port of the
 * Python oracle's `verifier/oracle/canonical.py`.
 *
 * Three callable shapes, because three signers disagree on what "JCS" means:
 *
 *   - `jcs(obj)`          : NFC-normalised, code-POINT key ordering (Python
 *                           `sort_keys`), numbers preserved as Python emits them.
 *                           Matches the AERF / ACTA reference producers.
 *   - `jcsRfc8785(obj)`   : strict RFC 8785 - UTF-16 code-UNIT key ordering and
 *                           ECMAScript Number.prototype.toString number output,
 *                           plus NFC. Byte-matches the agent-receipts upstream
 *                           canonicalization vectors.
 *   - `asqavJcs(obj)`     : the Asqav cloud's historical JCS WITHOUT NFC, kept
 *                           byte-identical to `verify_receipt.canonical_json`.
 *
 * Implementation parity notes vs the Python source:
 *   - Python `sort_keys` sorts by Unicode CODE POINT; JavaScript's default
 *     `Array.prototype.sort` on strings sorts by UTF-16 CODE UNIT. These differ
 *     only on supplementary-plane keys. `jcs`/`asqavJcs` therefore sort by code
 *     point (mirroring Python); `jcsRfc8785` sorts by code unit (RFC 8785).
 *   - V8's `Number.prototype.toString` already produces the ECMAScript shortest
 *     round-trip form RFC 8785 mandates; verified byte-equal against the upstream
 *     number vectors (1e-7, 1e+21, 0.30000000000000004, ...).
 *   - Python `json.dumps(ensure_ascii=False)` short-escapes \b \t \n \f \r,
 *     emits other control chars < 0x20 as \u00xx, and passes everything >= 0x20
 *     (including 0x7f and all non-ASCII) verbatim. `jsonString` reproduces this.
 *   - Python `json.dumps(allow_nan=False)` rejects NaN/Infinity; we throw too.
 *   - `jcs`/`asqavJcs` keep numbers as Python emits them. For the integer field
 *     sets these receipts carry, Python `repr` and V8 `toString` agree; floats
 *     are out of the Asqav/AERF/ACTA signed scope, so this is exact for the corpus.
 */

type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

/**
 * A JSON number that was written as a FLOAT literal (had a `.`, `e`, or `E`).
 *
 * JavaScript's `JSON.parse` collapses `500.0` to the integer `500` because
 * IEEE-754 has no int/float distinction, so the trailing `.0` is lost. Python's
 * `json.load` keeps it as a `float` and `json.dumps` re-emits `500.0`. The
 * AERF / ACTA / Asqav `jcs` dialects sign numbers "as the producer emitted them",
 * so a receipt carrying `500.0` is signed over the bytes `500.0`, not `500`.
 *
 * To reproduce those exact bytes, parse receipts with `parseJsonPreservingFloats`
 * (below), which wraps float literals in this class; the canonicalisers then
 * re-emit the float form. Integer literals are left as plain `number`.
 *
 * Parity scope: the corpus floats are whole-magnitude (`500.0`, `1250.0`), where
 * Python repr, the AERF Go producer, and this emitter all agree on `<int>.0`.
 * Exotic floats (e.g. `1e-7` vs Python's `1e-07`) do not occur in any signed
 * receipt; were they to, the AERF Go producer - not Python repr - is the target.
 */
export class RawFloat {
  constructor(public readonly value: number) {}
}

function isRawFloat(v: unknown): v is RawFloat {
  return v instanceof RawFloat;
}

/**
 * An integer literal beyond IEEE-754 safe range (|n| > 2^53). JS `JSON.parse`
 * collapses it to the nearest double, so two distinct integers can read as one
 * and a tampered receipt could verify. Keeping the exact source digits lets the
 * Python-dialect canonicalisers emit them verbatim (matching Python's
 * arbitrary-precision int), closing that false-PASS path.
 */
export class RawBigInt {
  constructor(public readonly source: string) {}
}

function isRawBigInt(v: unknown): v is RawBigInt {
  return v instanceof RawBigInt;
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
 * Parse JSON while preserving float literals as `RawFloat` and out-of-safe-range
 * integers as `RawBigInt`, so the canonicalisers re-emit `500.0` rather than the
 * collapsed `500` and never let two distinct big integers share one signature.
 * Mirrors what Python's `json.load` preserves implicitly.
 *
 * Hand-rolled recursive descent rather than a `JSON.parse` reviver: the reviver's
 * raw-literal `context.source` is only available on Node 21.1+, and this package
 * supports Node 18+. The grammar is RFC 8259 JSON; it is not a lenient parser.
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
    i++; // {
    ws();
    if (text[i] === "}") return i++, out;
    for (;;) {
      ws();
      if (text[i] !== '"') err("expected object key");
      const k = str();
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
 * Compare two strings by Unicode CODE POINT (Python `sort_keys` ordering).
 *
 * JavaScript string `<` compares by UTF-16 code unit, which differs from code
 * point on supplementary-plane characters. Iterating with `for...of` yields code
 * points, so we compare those directly.
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
 * Standard JSON string serialization matching Python `json.dumps(ensure_ascii=False)`.
 *
 * Short-escapes \b \t \n \f \r, emits other control chars < 0x20 as \u00xx, and
 * passes everything else (including 0x7f and all non-ASCII) verbatim. Note `/` is
 * NOT escaped, matching Python.
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
 * Serialise with a chosen key-comparison function.
 *
 * `honorFloat` selects the number dialect for `RawFloat` tokens:
 *   - true  (jcs / asqavJcs, the Python-`json.dumps` dialect): emit the float
 *     form, e.g. `500.0` (Python `repr(500.0)`).
 *   - false (jcsRfc8785, strict ECMAScript): collapse a whole-valued float to its
 *     integer form `500`, matching Python `_number_8785`'s `str(int(value))`.
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

/** Asqav cloud JCS bytes (no NFC); byte-identical to the cloud signer. */
export function asqavJcs(obj: unknown): Uint8Array {
  return new TextEncoder().encode(serialize(obj, compareCodePoints, true));
}

/** Strict RFC 8785 JCS bytes (UTF-16 code-unit key sort, ECMAScript numbers) with NFC. */
export function jcsRfc8785(obj: unknown): Uint8Array {
  return new TextEncoder().encode(serialize(nfc(obj), utf16CompareKeys, false));
}

export type { JsonValue };
