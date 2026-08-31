/**
 * Shared DID resolver mapping a verificationMethod DID URL to a raw Ed25519 key, a port of the Python
 * oracle's `did.py`. did:key resolves inline; everything else comes from an injected map, never a fetch.
 */

/** Bitcoin base58 alphabet - the base58btc encoding multibase 'z' uses. */
const B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
const B58_INDEX: Record<string, number> = {};
for (let i = 0; i < B58_ALPHABET.length; i++) B58_INDEX[B58_ALPHABET[i]] = i;

/** Multicodec prefix for an Ed25519 public key (unsigned-varint 0xed01). */
const ED25519_MULTICODEC = Uint8Array.from([0xed, 0x01]);

/** Decode a base58btc string to bytes, preserving leading-zero bytes as '1's. */
export function b58btcDecode(text: string): Uint8Array {
  let num = 0n;
  for (const ch of text) {
    const idx = B58_INDEX[ch];
    if (idx === undefined) throw new Error(`invalid base58btc character: '${ch}'`);
    num = num * 58n + BigInt(idx);
  }
  const bodyBytes: number[] = [];
  while (num > 0n) {
    bodyBytes.unshift(Number(num & 0xffn));
    num >>= 8n;
  }
  let pad = 0;
  for (const ch of text) {
    if (ch === "1") pad++;
    else break;
  }
  return Uint8Array.from([...new Array(pad).fill(0), ...bodyBytes]);
}

function startsWith(buf: Uint8Array, prefix: Uint8Array): boolean {
  if (buf.length < prefix.length) return false;
  for (let i = 0; i < prefix.length; i++) {
    if (buf[i] !== prefix[i]) return false;
  }
  return true;
}

/** Return the raw 32-byte Ed25519 key from a `did:key` 'z...' identifier, or null. */
function decodeDidKey(identifier: string): Uint8Array | null {
  if (!identifier.startsWith("z")) return null;
  let decoded: Uint8Array;
  try {
    decoded = b58btcDecode(identifier.slice(1));
  } catch {
    return null;
  }
  if (!startsWith(decoded, ED25519_MULTICODEC)) return null;
  const key = decoded.slice(ED25519_MULTICODEC.length);
  return key.length === 32 ? key : null;
}

function hexToBytes(hex: string): Uint8Array | null {
  if (hex.length % 2 !== 0) return null;
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    const byte = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
    if (Number.isNaN(byte)) return null;
    out[i] = byte;
  }
  return out;
}

/** Coerce injected key material (raw bytes or hex string) to raw 32-byte form. */
function coerceRaw(material: unknown): Uint8Array | null {
  let raw: Uint8Array | null;
  if (material instanceof Uint8Array) {
    raw = material;
  } else if (typeof material === "string") {
    raw = hexToBytes(material);
  } else {
    return null;
  }
  return raw !== null && raw.length === 32 ? raw : null;
}

/** Decode a base64url string to bytes (JWK coordinate form). */
function b64urlDecode(value: string): Uint8Array {
  const body = value.replace(/-/g, "+").replace(/_/g, "/");
  const padded = body + "=".repeat((-body.length % 4 + 4) % 4);
  return new Uint8Array(Buffer.from(padded, "base64"));
}

/** OKP/Ed25519 JWK -> raw 32-byte key; null for any other curve or bad encoding. */
function rawFromJwk(jwk: unknown): Uint8Array | null {
  if (jwk === null || typeof jwk !== "object" || Array.isArray(jwk)) return null;
  const rec = jwk as Record<string, unknown>;
  if (rec.kty !== "OKP" || rec.crv !== "Ed25519" || typeof rec.x !== "string") return null;
  let raw: Uint8Array;
  try {
    raw = b64urlDecode(rec.x);
  } catch {
    return null;
  }
  return raw.length === 32 ? raw : null;
}

/** Extract the raw Ed25519 key one DID-document verificationMethod publishes. */
function rawFromVerificationMethod(vm: unknown): Uint8Array | null {
  if (vm === null || typeof vm !== "object" || Array.isArray(vm)) return null;
  const rec = vm as Record<string, unknown>;
  if (typeof rec.publicKeyMultibase === "string") {
    // A Multikey multibase value is shaped exactly like a did:key identifier
    const key = decodeDidKey(rec.publicKeyMultibase);
    if (key !== null) return key;
  }
  const jwk = rawFromJwk(rec.publicKeyJwk);
  if (jwk !== null) return jwk;
  if (typeof rec.publicKeyBase58 === "string") {
    let raw: Uint8Array;
    try {
      raw = b58btcDecode(rec.publicKeyBase58);
    } catch {
      return null;
    }
    return raw.length === 32 ? raw : null;
  }
  return null;
}

/**
 * Walk an injected DID document like the fetched one: exact fragment match first,
 * else assertionMethod-authorized Ed25519 methods, then any remaining method.
 */
function keyFromDidDocument(
  didDoc: Record<string, unknown>,
  didUrl: string,
): readonly [Uint8Array | null, string] {
  const methods = Array.isArray(didDoc.verificationMethod)
    ? didDoc.verificationMethod.filter(
        (vm): vm is Record<string, unknown> => vm !== null && typeof vm === "object" && !Array.isArray(vm),
      )
    : [];
  const assertion = Array.isArray(didDoc.assertionMethod) ? didDoc.assertionMethod : [];
  let pool: Array<Record<string, unknown>> = [
    ...methods,
    ...assertion.filter(
      (vm): vm is Record<string, unknown> => vm !== null && typeof vm === "object" && !Array.isArray(vm),
    ),
  ];
  if (didUrl.includes("#")) {
    pool = pool.filter((vm) => vm.id === didUrl);
    if (pool.length === 0) {
      return [null, `no verificationMethod '${didUrl}' in injected DID document`];
    }
  } else {
    const refs = new Set(assertion.filter((vm): vm is string => typeof vm === "string"));
    pool.sort((a, b) => (refs.has(a.id as string) ? 0 : 1) - (refs.has(b.id as string) ? 0 : 1));
  }
  for (const vm of pool) {
    const key = rawFromVerificationMethod(vm);
    if (key !== null) {
      return [key, `resolved ${typeof vm.id === "string" ? vm.id : didUrl} from injected DID document`];
    }
  }
  return [null, `no Ed25519 verificationMethod for '${didUrl}' in injected DID document`];
}

/**
 * Resolve a verificationMethod DID URL to `[rawKeyOrNull, note]`. did:key resolves inline; others come
 * from `injected`, keyed by full DID URL then bare DID. No network.
 */
export function resolveEd25519Key(
  didUrl: string,
  injected: Record<string, unknown> | null = null,
): readonly [Uint8Array | null, string] {
  if (typeof didUrl !== "string" || !didUrl.startsWith("did:")) {
    return [null, `not a DID URL: '${didUrl}'`];
  }
  const bare = didUrl.split("#", 1)[0];
  if (bare.startsWith("did:key:")) {
    const key = decodeDidKey(bare.slice("did:key:".length));
    if (key === null) {
      return [null, "did:key identifier is not a base58btc Ed25519 multikey"];
    }
    return [key, "resolved did:key inline (multicodec ed25519)"];
  }
  const keys = injected || {};
  for (const candidate of [didUrl, bare]) {
    if (Object.prototype.hasOwnProperty.call(keys, candidate)) {
      const material = keys[candidate];
      if (material !== null && typeof material === "object" && !Array.isArray(material)) {
        return keyFromDidDocument(material as Record<string, unknown>, didUrl);
      }
      const raw = coerceRaw(material);
      if (raw === null) {
        return [null, `injected key for '${candidate}' is not a 32-byte Ed25519 key`];
      }
      return [raw, `resolved ${candidate} from injected map`];
    }
  }
  return [null, `no injected key for '${didUrl}' (oracle never fetches a DID document)`];
}
