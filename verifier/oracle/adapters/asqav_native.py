"""Asqav-native adapter - the existing verify_receipt logic behind the seam.

Behaviour-preserving wrapper over ``verify_receipt``: the envelope is the 3-key
``{payload, signature, anchors}`` shape, the signature is ML-DSA-65 over the
canonical bytes of ``payload`` signed DIRECTLY (no pre-hash, no field strip - the
signature sits beside the payload, not inside it), and the chain hash is the
SHA-256 of the predecessor payload's canonical bytes. Genesis is the all-zero
seed sentinel.

Key resolution and the structure check delegate to ``verify_receipt`` so the two
surfaces stay byte-for-byte identical; the oracle does not re-implement them.
"""
from __future__ import annotations

from typing import Any

import verify_receipt as _vr

from ..adapter import ChainStep, FormatAdapter, SignatureMaterial
from ..core import sha256_hex
from .acta import _is_lower_hex


def _payload(doc: dict) -> dict:
    """Normalise to the signed payload regardless of envelope nesting."""
    env = _vr.normalise_envelope(doc)
    return env.get("payload", env)


class AsqavNativeAdapter(FormatAdapter):
    """Asqav Compliance Receipt - ML-DSA-65 over canonical payload bytes."""

    name = "asqav-native"

    def detect(self, doc: dict) -> bool:
        sig = doc.get("signature")
        # An ACTA receipt carries a lowercase-hex sig; decline it so the formats stay disjoint.
        if isinstance(sig, dict) and _is_lower_hex(sig.get("sig")):
            return False
        payload = doc.get("payload")
        if isinstance(payload, dict) and "previousReceiptHash" in payload:
            return True
        # A bare payload (the conformance-vector shape) is also Asqav-native.
        return "previousReceiptHash" in doc and "issuer_id" in doc

    def extract_signature(self, doc: dict) -> SignatureMaterial:
        env = _vr.normalise_envelope(doc)
        sig_obj = env.get("signature", {})
        if isinstance(sig_obj, str):
            sig_obj = {"alg": "ML-DSA-65", "kid": _payload(doc).get("issuer_id", ""), "sig": sig_obj}
        return SignatureMaterial(
            sig=_vr._b64decode(sig_obj.get("sig", "")),
            alg=sig_obj.get("alg", "ML-DSA-65"),
            kid=sig_obj.get("kid", ""),
        )

    def resolve_key(self, doc: dict, key_provider: Any) -> tuple[bytes | None, str]:
        jwks = key_provider or {"keys": []}
        kid = self.extract_signature(doc).kid
        pk, status, _alg = _vr.resolve_key(jwks, kid)
        if pk is None:
            return None, f"kid {kid!r} not in jwks directory"
        return pk, f"resolved kid {kid} (status={status})"

    def signing_input(self, doc: dict) -> bytes:
        # Asqav signs the canonical bytes of the payload directly, no pre-hash.
        return _vr.canonical_json(_payload(doc))

    def chain_step(self, doc: dict) -> ChainStep:
        prev = _payload(doc).get("previousReceiptHash")
        is_genesis = prev == _vr.FIRST_RECEIPT_SEED
        return ChainStep(
            prev_field=prev,
            is_genesis=is_genesis,
            recompute=lambda pred: sha256_hex(_vr.canonical_json(_payload(pred))),
        )

    def schema(self, doc: dict) -> tuple[str, str]:
        return _vr.check_structure(_payload(doc))
