"""Authproof adapter - ES256 over insertion-order JSON, key embedded as a JWK.

Authproof delegation receipts are signed by the reference JS SDK
(``Commonguy25/authproof-sdk``, ``src/authproof.js``), which is authoritative for
the wire shape. A receipt is a flat object whose signature is ECDSA P-256 / SHA-256
over ``JSON.stringify(receipt-without-signature)`` in INSERTION order (not JCS, no
pre-hash). The signature is hex-encoded raw ``r || s`` (64 bytes), and the signer's
public key is embedded as a P-256 JWK at ``signerPublicKey``.

Divergences this adapter deliberately does NOT follow (the JS SDK is the target):
  - the bundled draft-nelson text specifies a base64url signature, a content-hash
    ``delegationId``, and canonical JSON; the shipping SDK emits hex, a random
    ``auth-<ms>-<rand>`` id, and insertion-order ``JSON.stringify``.
  - the repo's separate Python SDK signs DER over sorted-key JSON with snake_case
    fields; that is a distinct, non-interoperable variant.

Scope: verifies the issuer signature and structural presence. The base receipt
carries no in-band chain link (chaining is a separate SDK feature), so the chain
axis reports genesis. Interop is proven against a receipt minted by the real JS
SDK; there is no published portable vector corpus.
"""
from __future__ import annotations

import base64
import json
import re
from typing import Any

from ..adapter import ChainStep, FormatAdapter, SignatureMaterial

#: A shipping-SDK delegationId is ``auth-<epoch_ms>-<5 base36 chars>``.
_DELEGATION_ID = re.compile(r"^auth-\d+-[a-z0-9]{5}$")

#: Fields the SDK always writes, used for the structural check.
_REQUIRED = (
    "delegationId",
    "issuedAt",
    "scope",
    "boundaries",
    "timeWindow",
    "operatorInstructions",
    "instructionsHash",
    "signerPublicKey",
    "signature",
)


def _safe_hex(value: Any) -> bytes:
    """Decode a hex string; b'' on any malformed input so verify FAILs, never crashes."""
    if not isinstance(value, str):
        return b""
    try:
        return bytes.fromhex(value)
    except ValueError:
        return b""


def _b64url(value: Any) -> bytes:
    """Decode a base64url JWK coordinate; b'' on malformed input."""
    if not isinstance(value, str):
        return b""
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except Exception:
        return b""


def _is_p256_jwk(jwk: Any) -> bool:
    return isinstance(jwk, dict) and jwk.get("kty") == "EC" and jwk.get("crv") == "P-256"


class AuthproofAdapter(FormatAdapter):
    """Authproof delegation receipt - ES256 over insertion-order JSON.stringify."""

    name = "authproof"

    def detect(self, doc: dict) -> bool:
        did = doc.get("delegationId")
        return (
            isinstance(did, str)
            and _DELEGATION_ID.match(did) is not None
            and "operatorInstructions" in doc
            and "instructionsHash" in doc
            and _is_p256_jwk(doc.get("signerPublicKey"))
            and isinstance(doc.get("signature"), str)
        )

    def extract_signature(self, doc: dict) -> SignatureMaterial:
        return SignatureMaterial(sig=_safe_hex(doc.get("signature", "")), alg="ES256", kid="")

    def resolve_key(self, doc: dict, _key_provider: Any) -> tuple[bytes | None, str]:
        """Return the 65-byte uncompressed P-256 point from the embedded JWK (key is in-band)."""
        jwk = doc.get("signerPublicKey")
        if not _is_p256_jwk(jwk):
            return None, "signerPublicKey is not a P-256 JWK"
        x, y = _b64url(jwk.get("x")), _b64url(jwk.get("y"))
        if len(x) != 32 or len(y) != 32:
            return None, "P-256 JWK coordinates are not 32 bytes each"
        return b"\x04" + x + y, "resolved embedded P-256 JWK"

    def signing_input(self, doc: dict) -> bytes:
        """The SDK signs ``JSON.stringify`` of the receipt minus ``signature``, key order intact.

        ``json.dumps`` here reproduces ``JSON.stringify`` byte-for-byte for the SDK's
        string-valued fields (separators and string escaping agree, key order is
        preserved, non-ASCII is emitted verbatim). The one divergence is non-integer
        float serialisation, which Authproof's signed schema never carries; a numeric
        ``metadata`` value would be the only case to re-check.
        """
        body = {k: v for k, v in doc.items() if k != "signature"}
        return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def chain_step(self, doc: dict) -> ChainStep:
        # The base Authproof receipt carries no in-band chain link of its own.
        return ChainStep(prev_field=None, is_genesis=True, recompute=lambda _pred: "")

    def schema(self, doc: dict) -> tuple[str, str]:
        missing = [f for f in _REQUIRED if doc.get(f) is None]
        if missing:
            return "FAIL", f"authproof receipt missing fields: {','.join(missing)}"
        return "PASS", "authproof delegation receipt; required fields present"
