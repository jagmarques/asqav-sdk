/**
 * Shared DID resolver - map a verificationMethod DID URL to a raw Ed25519 key.
 * A port of the Python oracle's `verifier/oracle/did.py`.
 *
 * `resolveEd25519Key(didUrl, injected)` returns the raw 32-byte Ed25519 public
 * key for a signer, or null when it cannot resolve without a network the oracle
 * is forbidden from touching.
 *
 *   - `did:key`  : self-contained. The multibase `z` (base58btc) identifier
 *                  decodes to a multicodec frame: the two-byte 0xed 0x01 Ed25519
 *                  prefix followed by the 32 raw key bytes. No network.
 *   - other      : resolved from an injected `{didOrKid: hexOrRaw}` map. The
 *                  oracle NEVER fetches a DID document over the network.
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

/**
 * Resolve a verificationMethod DID URL to `[rawKeyOrNull, note]`.
 *
 * did:key resolves inline; every other method resolves from `injected` keyed by
 * the full DID URL first, then by the bare DID (fragment stripped). No network.
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
      const raw = coerceRaw(keys[candidate]);
      if (raw === null) {
        return [null, `injected key for '${candidate}' is not a 32-byte Ed25519 key`];
      }
      return [raw, `resolved ${candidate} from injected map`];
    }
  }
  return [null, `no injected key for '${didUrl}' (oracle never fetches a DID document)`];
}
