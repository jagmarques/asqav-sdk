"""Shared DID resolver - map a verificationMethod DID URL to a raw Ed25519 key.

``resolve_ed25519_key(did_url, injected)`` returns the raw 32-byte Ed25519 public
key for a signer, or None when it cannot resolve without a network the oracle is
forbidden from touching. Two methods are supported:

  - ``did:key`` - self-contained. The multibase ``z`` (base58btc) identifier
    decodes to a multicodec frame: the two-byte ``0xed 0x01`` Ed25519 prefix
    followed by the 32 raw key bytes. No network, no injected map needed.
  - ``did:agent`` / ``did:web`` (and any other method) - resolved from an injected
    ``{did_or_kid: hex_or_raw}`` map the caller supplies. The oracle NEVER fetches
    a DID document over the network; an unmapped DID returns None.

This resolver is adapter-agnostic so a later ERC-8004 / DID hook reuses it. base58
is an encoding, not crypto, so the base58btc decoder is implemented here and pinned
by a unit test against the upstream did:key reference vectors; the Ed25519 verify
itself stays in ``crypto.py`` behind the ``cryptography`` library.
"""
from __future__ import annotations

#: Bitcoin base58 alphabet - the base58btc encoding multibase 'z' uses.
_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {ch: i for i, ch in enumerate(_B58_ALPHABET)}

#: Multicodec prefix for an Ed25519 public key (unsigned-varint 0xed01).
_ED25519_MULTICODEC = b"\xed\x01"


def b58btc_decode(text: str) -> bytes:
    """Decode a base58btc string to bytes, preserving leading-zero bytes as '1's."""
    num = 0
    for ch in text:
        if ch not in _B58_INDEX:
            raise ValueError(f"invalid base58btc character: {ch!r}")
        num = num * 58 + _B58_INDEX[ch]
    body = num.to_bytes((num.bit_length() + 7) // 8, "big") if num else b""
    pad = len(text) - len(text.lstrip("1"))
    return b"\x00" * pad + body


def _decode_did_key(identifier: str) -> bytes | None:
    """Return the raw 32-byte Ed25519 key from a ``did:key`` 'z...' identifier."""
    if not identifier.startswith("z"):
        return None  # multibase base58btc is the only did:key form this resolver decodes
    try:
        decoded = b58btc_decode(identifier[1:])
    except ValueError:
        return None
    if not decoded.startswith(_ED25519_MULTICODEC):
        return None  # not an Ed25519 did:key (e.g. did:key for a different curve)
    key = decoded[len(_ED25519_MULTICODEC) :]
    return key if len(key) == 32 else None


def _coerce_raw(material: object) -> bytes | None:
    """Coerce injected key material (raw bytes or hex string) to raw 32-byte form."""
    if isinstance(material, bytes):
        raw = material
    elif isinstance(material, str):
        try:
            raw = bytes.fromhex(material)
        except ValueError:
            return None
    else:
        return None
    return raw if len(raw) == 32 else None


def resolve_ed25519_key(
    did_url: str, injected: dict | None = None
) -> tuple[bytes | None, str]:
    """Resolve a verificationMethod DID URL to ``(raw_key_or_None, note)``.

    did:key resolves inline; every other method resolves from ``injected`` keyed by
    the full DID URL first, then by the bare DID (fragment stripped). No network.
    """
    if not isinstance(did_url, str) or not did_url.startswith("did:"):
        return None, f"not a DID URL: {did_url!r}"
    bare = did_url.split("#", 1)[0]
    if bare.startswith("did:key:"):
        key = _decode_did_key(bare[len("did:key:") :])
        if key is None:
            return None, "did:key identifier is not a base58btc Ed25519 multikey"
        return key, "resolved did:key inline (multicodec ed25519)"
    keys = injected or {}
    for candidate in (did_url, bare):
        if candidate in keys:
            raw = _coerce_raw(keys[candidate])
            if raw is None:
                return None, f"injected key for {candidate!r} is not a 32-byte Ed25519 key"
            return raw, f"resolved {candidate} from injected map"
    return None, f"no injected key for {did_url!r} (oracle never fetches a DID document)"
