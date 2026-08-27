"""Shared DID resolver - map a verificationMethod DID URL to a raw Ed25519 key.

``resolve_ed25519_key(did_url, injected)`` returns the raw 32-byte Ed25519 public
key for a signer, or None when it cannot resolve without a network the oracle is
forbidden from touching. Two resolution shapes are supported:

  - ``did:key`` - self-contained. The multibase ``z`` (base58btc) identifier
    decodes to a multicodec frame: the two-byte ``0xed 0x01`` Ed25519 prefix
    followed by the 32 raw key bytes. No network, no injected map needed.
  - ``did:agent`` / ``did:web`` (and any other method) - resolved from an injected
    map the caller supplies. A value is either a raw key (hex string or bytes,
    32 bytes) or the DID DOCUMENT the method's network fetch would have returned;
    the resolver then walks ``verificationMethod`` / ``assertionMethod`` and
    extracts an Ed25519 key (publicKeyMultibase Multikey, publicKeyJwk OKP, or
    legacy publicKeyBase58). The oracle NEVER fetches a DID document over the
    network; an unmapped DID returns None (fail closed).

This resolver is adapter-agnostic so a later ERC-8004 / DID hook reuses it. base58
is an encoding, not crypto, so the base58btc decoder is implemented here and pinned
by a unit test against the upstream did:key reference vectors; the Ed25519 verify
itself stays in ``crypto.py`` behind the ``cryptography`` library.
"""
from __future__ import annotations

import base64

#: Bitcoin base58 alphabet - the base58btc encoding multibase 'z' uses.
_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {ch: i for i, ch in enumerate(_B58_ALPHABET)}

#: Multicodec prefix for an Ed25519 public key (unsigned-varint 0xed01).
_ED25519_MULTICODEC = b"\xed\x01"


    # Decode a base58btc string to bytes, preserving leading-zero bytes as '1's.
def b58btc_decode(text: str) -> bytes:
    num = 0
    for ch in text:
        if ch not in _B58_INDEX:
            raise ValueError(f"invalid base58btc character: {ch!r}")
        num = num * 58 + _B58_INDEX[ch]
    body = num.to_bytes((num.bit_length() + 7) // 8, "big") if num else b""
    pad = len(text) - len(text.lstrip("1"))
    return b"\x00" * pad + body


    # Return the raw 32-byte Ed25519 key from a ``did:key`` 'z...' identifier.
def _decode_did_key(identifier: str) -> bytes | None:
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


    # Coerce injected key material (raw bytes or hex string) to raw 32-byte form.
def _coerce_raw(material: object) -> bytes | None:
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


    # OKP/Ed25519 JWK -> raw 32-byte key; None for any other curve or bad encoding.
def _raw_from_jwk(jwk: object) -> bytes | None:
    if not isinstance(jwk, dict) or jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519":
        return None
    x = jwk.get("x")
    if not isinstance(x, str):
        return None
    try:
        raw = base64.urlsafe_b64decode(x + "=" * (-len(x) % 4))
    except (ValueError, base64.binascii.Error):
        return None
    return raw if len(raw) == 32 else None


    # Extract the raw Ed25519 key one DID-document verificationMethod publishes.
def _raw_from_verification_method(vm: object) -> bytes | None:
    if not isinstance(vm, dict):
        return None
    multibase = vm.get("publicKeyMultibase")
    if isinstance(multibase, str):
        # A Multikey multibase value is shaped exactly like a did:key identifier.
        key = _decode_did_key(multibase)
        if key is not None:
            return key
    jwk = _raw_from_jwk(vm.get("publicKeyJwk"))
    if jwk is not None:
        return jwk
    b58 = vm.get("publicKeyBase58")
    if isinstance(b58, str):
        try:
            raw = b58btc_decode(b58)
        except ValueError:
            return None
        return raw if len(raw) == 32 else None
    return None


    # Walk an injected DID document like the fetched one: exact fragment match first,
    # else assertionMethod-authorized Ed25519 methods, then any remaining method.
def _key_from_did_document(did_doc: dict, did_url: str) -> tuple[bytes | None, str]:
    methods = did_doc.get("verificationMethod")
    methods = [vm for vm in methods if isinstance(vm, dict)] if isinstance(methods, list) else []
    assertion = did_doc.get("assertionMethod")
    assertion = assertion if isinstance(assertion, list) else []
    pool = methods + [vm for vm in assertion if isinstance(vm, dict)]
    if "#" in did_url:
        pool = [vm for vm in pool if vm.get("id") == did_url]
        if not pool:
            return None, f"no verificationMethod {did_url!r} in injected DID document"
    else:
        refs = {vm for vm in assertion if isinstance(vm, str)}
        pool.sort(key=lambda vm: 0 if vm.get("id") in refs else 1)
    for vm in pool:
        key = _raw_from_verification_method(vm)
        if key is not None:
            return key, f"resolved {vm.get('id', did_url)} from injected DID document"
    return None, f"no Ed25519 verificationMethod for {did_url!r} in injected DID document"


def resolve_ed25519_key(
    did_url: str, injected: dict | None = None
) -> tuple[bytes | None, str]:
    """Resolve a verificationMethod DID URL to ``(raw_key_or_None, note)``.

    did:key resolves inline; every other method resolves from ``injected`` keyed by
    the full DID URL first, then by the bare DID (fragment stripped). An injected
    value is a raw 32-byte key (bytes or hex) or the DID document the method's
    fetch would have returned. No network.
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
            material = keys[candidate]
            if isinstance(material, dict):
                return _key_from_did_document(material, did_url)
            raw = _coerce_raw(material)
            if raw is None:
                return None, f"injected key for {candidate!r} is not a 32-byte Ed25519 key"
            return raw, f"resolved {candidate} from injected map"
    return None, f"no injected key for {did_url!r} (oracle never fetches a DID document)"
