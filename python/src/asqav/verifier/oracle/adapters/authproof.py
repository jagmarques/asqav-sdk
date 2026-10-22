"""Authproof adapter: ES256 over insertion-order JSON, per the reference SDK.
The signature is hex raw ``r || s``, the signer's P-256 JWK embedded; no JCS."""
# Exercised with bytes from the real JavaScript library, not a portable vector
# set; only the issuer signature and the document's shape matter.
from __future__ import annotations

import base64
import json
import re
from typing import Any

from ..adapter import ChainStep, FormatAdapter, SignatureMaterial

#: A shipping-SDK delegationId is ``auth-<epoch_ms>-<5 base36 chars>``.
#: ``[0-9]`` and ``\A``/``\Z`` keep it byte-for-byte with the ECMAScript regex.
_DELEGATION_ID = re.compile(r"\Aauth-[0-9]+-[a-z0-9]{5}\Z")

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


    # Decode hex; b'' on malformed input so verify FAILs, never crashes.
def _safe_hex(value: Any) -> bytes:
    if not isinstance(value, str):
        return b""
    try:
        return bytes.fromhex(value)
    except ValueError:
        return b""


    # Decode a base64url JWK coordinate; b'' on malformed input.
def _b64url(value: Any) -> bytes:
    if not isinstance(value, str):
        return b""
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except Exception:
        return b""


def _is_p256_jwk(jwk: Any) -> bool:
    return isinstance(jwk, dict) and jwk.get("kty") == "EC" and jwk.get("crv") == "P-256"


    # Authproof delegation receipt - ES256 over insertion-order JSON.stringify.
class AuthproofAdapter(FormatAdapter):

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

        # Return the 65-byte uncompressed P-256 point from the in-band JWK.
    def resolve_key(self, doc: dict, _key_provider: Any) -> tuple[bytes | None, str]:
        jwk = doc.get("signerPublicKey")
        if not _is_p256_jwk(jwk):
            return None, "signerPublicKey is not a P-256 JWK"
        x, y = _b64url(jwk.get("x")), _b64url(jwk.get("y"))
        if len(x) != 32 or len(y) != 32:
            return None, "P-256 JWK coordinates are not 32 bytes each"
        return b"\x04" + x + y, "resolved embedded P-256 JWK"

    def signing_input(self, doc: dict) -> bytes:
        """The SDK signs ``JSON.stringify`` of the receipt minus ``signature``.
        ``json.dumps`` matches it bytewise; non-integer floats differ."""
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
