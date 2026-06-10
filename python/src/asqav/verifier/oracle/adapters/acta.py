"""ACTA adapter - Ed25519 over the JCS of the payload, signed directly.

Targets draft-farley-acta-signed-receipts-01 (the draft Asqav profiles). The
envelope is {payload, signature:{alg, kid, sig}}; the signature is Ed25519 over
the JCS canonical bytes of the `payload` object, signed DIRECTLY (no pre-hash),
with the sig as a lowercase hex string. The OPTIONAL Commitment Mode (signing
SHA-256(JCS(payload))) is NOT implemented: a commitment-mode receipt fails the
baseline signature check, which is the honest outcome - never a false pass, and
the adapter never tries both signing inputs.

Asqav-native is a PROFILE of ACTA, so the payloads look identical; the formats are
kept disjoint by the signature encoding (ACTA hex, Asqav-native base64) plus the
Asqav-only `anchors` envelope key. `_is_lower_hex` is the shared discriminator the
asqav-native adapter also imports so neither claims the other's receipts.

The chain field previousReceiptHash is SHA-256 of the JCS of the FULL predecessor
receipt INCLUDING its signature (the inverse of AERF, which excludes the
signature). It lives inside `payload`, so it is covered by the signature; removing
or altering it breaks the signature. Genesis OMITS the field; a present
`previousReceiptHash: null` is a malformed genesis, not a link. Keys resolve from
an injected JWK Set (OKP/Ed25519, base64url `x`), the same no-live-fetch model the
other adapters use.
"""
from __future__ import annotations

from typing import Any

from asqav.verifier import verify_receipt as _vr

from ..adapter import ChainStep, FormatAdapter, SignatureMaterial
from ..canonical import jcs
from ..core import sha256_hex


def _safe_hex(value: Any) -> bytes:
    """Decode a hex string; b'' on any malformed input so verify FAILs, never crashes."""
    if not isinstance(value, str):
        return b""
    try:
        return bytes.fromhex(value)
    except ValueError:
        return b""


def _is_lower_hex(s: Any) -> bool:
    """True for a non-empty even-length lowercase-hex string (an ACTA signature).

    A safe POSITIVE test: base64 (the Asqav-native encoding) carries uppercase
    and/or `+` `/` `=`, all hex-invalid, so this never matches a base64 signature.
    """
    return (
        isinstance(s, str)
        and len(s) > 0
        and len(s) % 2 == 0
        and all(c in "0123456789abcdef" for c in s)
    )


class ActaAdapter(FormatAdapter):
    """ACTA signed receipt - Ed25519 over JCS(payload), hex sig, chain incl sig."""

    name = "acta"

    def detect(self, doc: dict) -> bool:
        payload = doc.get("payload")
        sig = doc.get("signature")
        if not isinstance(payload, dict) or not isinstance(sig, dict):
            return False
        if "anchors" in doc:  # Asqav-native envelope marker; never ACTA.
            return False
        return _is_lower_hex(sig.get("sig"))

    def extract_signature(self, doc: dict) -> SignatureMaterial:
        sig = doc.get("signature", {})
        return SignatureMaterial(
            sig=_safe_hex(sig.get("sig", "")),
            alg=sig.get("alg", "EdDSA"),
            kid=sig.get("kid", ""),
        )

    def resolve_key(self, doc: dict, key_provider: Any) -> tuple[bytes | None, str]:
        kid = doc.get("signature", {}).get("kid", "")
        for jwk in (key_provider or {}).get("keys", []):
            if jwk.get("kid") != kid:
                continue
            if jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519":
                return None, f"kid {kid!r} is not an OKP/Ed25519 JWK"
            raw = _vr._b64decode(jwk.get("x", ""))
            if len(raw) != 32:
                return None, f"kid {kid!r} Ed25519 key is {len(raw)} bytes, expected 32"
            return raw, f"resolved kid {kid} from ACTA JWK Set"
        return None, f"kid {kid!r} not in ACTA JWK Set"

    def signing_input(self, doc: dict) -> bytes:
        # Baseline: Ed25519 signs the JCS bytes of the payload directly, no pre-hash.
        return jcs(doc.get("payload", {}))

    def chain_step(self, doc: dict) -> ChainStep:
        payload = doc.get("payload", {})
        prev = payload.get("previousReceiptHash")
        is_genesis = "previousReceiptHash" not in payload
        # Chain hash covers the full predecessor receipt, signature included.
        return ChainStep(
            prev_field=prev,
            is_genesis=is_genesis,
            recompute=lambda pred: sha256_hex(jcs(pred)),
        )

    def schema(self, doc: dict) -> tuple[str, str]:
        payload = doc.get("payload")
        sig = doc.get("signature")
        if not isinstance(payload, dict) or not isinstance(sig, dict):
            return "FAIL", "ACTA receipt needs object payload and signature"
        # Require only the temporal anchor + signature triple; `type`/`issuer_id` are
        # OPTIONAL Asqav-profile fields.
        missing = [f for f in ("issued_at",) if f not in payload]
        missing += [f"signature.{f}" for f in ("alg", "kid", "sig") if f not in sig]
        if missing:
            return "FAIL", f"missing required fields: {','.join(missing)}"
        if "previousReceiptHash" in payload and payload["previousReceiptHash"] is None:
            return "FAIL", "genesis must omit previousReceiptHash, not set it null"
        # False-attestation guard: when `issuer_id` is present it must match the signing kid.
        if "issuer_id" in payload and payload["issuer_id"] != sig.get("kid"):
            return "FAIL", "issuer_id must match signature.kid"
        # Bind alg to the EdDSA baseline; reject registry algs this verifier does not check.
        if sig.get("alg") not in ("EdDSA", "Ed25519"):
            return "FAIL", f"unsupported ACTA alg {sig.get('alg')!r}; only EdDSA baseline is implemented"
        return "PASS", "issued_at + signature triple present; issuer_id (when present) matches kid; EdDSA baseline"
