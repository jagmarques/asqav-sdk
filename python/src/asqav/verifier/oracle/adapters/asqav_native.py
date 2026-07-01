"""Asqav-native adapter - the existing verify_receipt logic behind the seam.

Two real Asqav wire shapes route here, kept mutually exclusive at detection:

  - COMPLIANCE mode: the 3-key ``{payload, signature, anchors}`` envelope whose
    ``payload`` carries ``previousReceiptHash`` / ``issuer_id`` (the protectmcp:*
    shape). ML-DSA-65 (or Ed25519 for the test vectors) signs the canonical bytes
    of ``payload`` DIRECTLY (no pre-hash, no field strip); the chain hash is the
    SHA-256 of the predecessor payload's canonical bytes.
  - HASH mode: the default ``/sign`` output - a FLAT receipt with ``mode:"hash"``,
    ``payload:null``, a ``signature_b64`` (or ``signature``), and a ``hash`` over
    the action bytes. The signing input is the flat 11-field object the cloud
    rebuilds, not the receipt itself; ``_hash_mode_signing_input`` reconstructs it.

Scope: this adapter verifies the issuer signature, the hash-chain link, and
structural presence of the required fields. It does NOT check anchor liveness or
issued_at clock skew - the standalone ``verify_receipt`` carries those axes.

Key resolution and the compliance-mode structure check delegate to
``verify_receipt`` so the two surfaces stay byte-for-byte identical; the oracle
does not re-implement them. Canonicalisation goes through ``asqav_jcs``, asserted
byte-identical to ``verify_receipt.canonical_json`` by the cloud-parity test.
"""
from __future__ import annotations

from typing import Any

from asqav.verifier import verify_receipt as _vr

from ..adapter import ChainStep, FormatAdapter, SignatureMaterial
from ..canonical import asqav_jcs
from ..core import sha256_hex
from .acta import _is_lower_hex

#: Field set the cloud's hash-mode signer canonicalises, in _build_signing_message order.
_HASH_MODE_FIELDS = (
    "v",
    "mode",
    "hash",
    "hash_algo",
    "metadata",
    "server_timestamp",
    "action_id",
    "agent_id",
    "org_id",
    "policy_digest",
    "policy_decision",
)


def _is_hash_mode(doc: dict) -> bool:
    """True for a flat hash-mode signature receipt (mode=hash, null payload, a sig)."""
    if doc.get("mode") != "hash" or doc.get("payload") is not None:
        return False
    return bool(doc.get("signature_b64") or doc.get("signature"))


def _payload(doc: dict) -> dict:
    """Normalise to the signed payload regardless of envelope nesting."""
    env = _vr.normalise_envelope(doc)
    return env.get("payload", env)


def _safe_b64(value: Any) -> bytes:
    """Decode signature material; b'' on any malformed input so verify FAILs, never crashes."""
    if not isinstance(value, str):
        return b""
    try:
        return _vr._b64decode(value)
    except Exception:
        return b""


class AsqavNativeAdapter(FormatAdapter):
    """Asqav Compliance Receipt - ML-DSA-65 over canonical bytes (compliance or hash mode)."""

    name = "asqav-native"

    def detect(self, doc: dict) -> bool:
        if _is_hash_mode(doc):
            return True
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
        if _is_hash_mode(doc):
            return SignatureMaterial(
                sig=_safe_b64(doc.get("signature_b64") or doc.get("signature", "")),
                alg=doc.get("algorithm", "ML-DSA-65"),
                kid=doc.get("key_id", ""),
            )
        env = _vr.normalise_envelope(doc)
        sig_obj = env.get("signature", {})
        if isinstance(sig_obj, str):
            sig_obj = {"alg": "ML-DSA-65", "kid": _payload(doc).get("issuer_id", ""), "sig": sig_obj}
        return SignatureMaterial(
            sig=_safe_b64(sig_obj.get("sig", "")),
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
        if _is_hash_mode(doc):
            return self._hash_mode_signing_input(doc)
        # Asqav signs the canonical bytes of the payload directly, no pre-hash.
        return asqav_jcs(_payload(doc))

    def _hash_mode_signing_input(self, doc: dict) -> bytes:
        """Rebuild the flat object the cloud's hash-mode path signs, then canonicalise.

        Mirrors ``agents.py::_build_signing_message`` hash-mode branch field-for-field;
        ``asqav_jcs`` sorts the keys, so insertion order is cosmetic but kept aligned.
        """
        flat = {
            "v": 1,
            "mode": "hash",
            "hash": doc.get("hash"),
            "hash_algo": doc.get("hash_algo") or "sha256",
            "metadata": doc.get("metadata") or {},
            "server_timestamp": doc.get("server_timestamp"),
            "action_id": doc.get("action_id"),
            "agent_id": doc.get("agent_id"),
            "org_id": doc.get("org_id"),
            "policy_digest": doc.get("policy_digest"),
            "policy_decision": doc.get("policy_decision"),
        }
        return asqav_jcs(flat)

    def chain_step(self, doc: dict) -> ChainStep:
        if _is_hash_mode(doc):
            # A hash-mode signature receipt carries no in-band chain link of its own.
            return ChainStep(prev_field=None, is_genesis=True, recompute=lambda _pred: "")
        prev = _payload(doc).get("previousReceiptHash")
        is_genesis = prev == _vr.FIRST_RECEIPT_SEED
        return ChainStep(
            prev_field=prev,
            is_genesis=is_genesis,
            recompute=lambda pred: sha256_hex(asqav_jcs(_payload(pred))),
        )

    def schema(self, doc: dict) -> tuple[str, str]:
        if _is_hash_mode(doc):
            missing = [f for f in _HASH_MODE_FIELDS if doc.get(f) is None and f != "policy_digest"]
            if missing:
                return "FAIL", f"hash-mode receipt missing fields: {','.join(missing)}"
            return "PASS", "hash-mode signature receipt; required flat fields present"
        return _vr.check_structure(_payload(doc))

    def extra_axes(self, doc: dict, key_provider: Any) -> list[tuple[str, str, str]]:
        """Gate the verdict on the signing key's revocation status.

        A receipt signed by a revoked key must not PASS offline, matching the
        hosted /verify. No key resolved means the signature axis already FAILs,
        so this axis only weighs in once a key is found.
        """
        jwks = key_provider or {"keys": []}
        kid = self.extract_signature(doc).kid
        pk, status, _alg = _vr.resolve_key(jwks, kid)
        if pk is None:
            return []
        if _is_hash_mode(doc):
            issued_at = doc.get("server_timestamp", "")
        else:
            issued_at = _payload(doc).get("issued_at", "")
        revoked_at = _vr.resolve_revoked_at(jwks, kid)
        res, note = _vr.check_key_status(status, issued_at, revoked_at)
        return [("key_status", res, note)]

    def attestation(self, doc: dict) -> dict[str, Any]:
        """Surface the v:2 in-body ``signer``. None for v:1 and hash-mode.

        Read only from the signed payload, so a signer appended as loose
        metadata outside the canonical body is never surfaced (and would also
        break the signature).
        """
        if _is_hash_mode(doc):
            return {}
        signer = _payload(doc).get("signer")
        return {"signer": signer} if signer is not None else {}
