"""AERF adapter - Ed25519 over RFC 8785 JCS, signed directly (no pre-hash).

AERF receipts are a flat object (not an envelope). The signature is Ed25519 over
the full RFC 8785 JCS bytes of the receipt with the non-payload fields stripped -
``signature``, ``timestamp``, ``parent_signature``, ``parent_key_id``,
``log_inclusion_proof`` - signed DIRECTLY, with no pre-hash.

Chain rule follows the AERF SPEC, not the agentmint reference producer: the
``previous_receipt_hash`` is the SHA-256 of the predecessor's signable payload
with the SAME field set stripped, i.e. the signature is EXCLUDED from the chain
hash. (The agentmint producer includes the signature; the published spec vectors
exclude it, so this adapter targets the spec.) Genesis OMITS the field entirely;
a present ``previous_receipt_hash: null`` is a malformed genesis, not a link.

Public keys are the raw 32-byte Ed25519 form; callers pass either raw bytes or an
RFC 8410 SPKI value, decoded by ``_raw_ed25519``.
"""
from __future__ import annotations

from typing import Any

from ..adapter import ChainStep, FormatAdapter, SignatureMaterial
from ..canonical import jcs
from ..core import sha256_hex

#: Fields removed before canonicalising for both the signature and the chain hash.
_STRIP = ("signature", "timestamp", "parent_signature", "parent_key_id", "log_inclusion_proof")

#: RFC 8410 Ed25519 SPKI prefix; an SPKI key is this 12-byte header + 32 raw bytes.
_SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")


def _signable(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in _STRIP}


def _raw_ed25519(key: bytes) -> bytes:
    """Return the raw 32-byte key from raw input or an RFC 8410 SPKI value."""
    if len(key) == 44 and key.startswith(_SPKI_PREFIX):
        return key[12:]
    return key


class AerfAdapter(FormatAdapter):
    """AERF notarised-evidence receipt - Ed25519 over JCS, sig excluded from chain."""

    name = "aerf"

    def detect(self, doc: dict) -> bool:
        return doc.get("type") == "notarised_evidence" and "evidence_hash_sha512" in doc

    def extract_signature(self, doc: dict) -> SignatureMaterial:
        return SignatureMaterial(
            sig=bytes.fromhex(doc.get("signature", "")),
            alg="Ed25519",
            kid=doc.get("key_id", ""),
        )

    def resolve_key(self, doc: dict, key_provider: Any) -> tuple[bytes | None, str]:
        keys = key_provider or {}
        kid = doc.get("key_id", "")
        material = keys.get(kid)
        if material is None:
            return None, f"key_id {kid!r} not supplied to the key provider"
        raw = _raw_ed25519(material if isinstance(material, bytes) else bytes.fromhex(material))
        return raw, f"resolved key_id {kid}"

    def signing_input(self, doc: dict) -> bytes:
        # AERF signs the JCS canonical bytes of the stripped payload directly.
        return jcs(_signable(doc))

    def chain_step(self, doc: dict) -> ChainStep:
        prev = doc.get("previous_receipt_hash")
        is_genesis = "previous_receipt_hash" not in doc
        return ChainStep(
            prev_field=prev,
            is_genesis=is_genesis,
            recompute=lambda pred: sha256_hex(jcs(_signable(pred))),
        )

    def schema(self, doc: dict) -> tuple[str, str]:
        required = (
            "id",
            "type",
            "plan_id",
            "agent",
            "action",
            "in_policy",
            "evidence_hash_sha512",
            "observed_at",
            "key_id",
            "signature",
        )
        missing = [f for f in required if f not in doc]
        if missing:
            return "FAIL", f"missing required fields: {','.join(missing)}"
        if "previous_receipt_hash" in doc and doc["previous_receipt_hash"] is None:
            return "FAIL", "genesis must omit previous_receipt_hash, not set it null"
        return "PASS", "required fields present; type notarised_evidence"
