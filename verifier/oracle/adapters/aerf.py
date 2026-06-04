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

Beyond the issuer signature and chain, AERF layers two conditionally-REQUIRED
signature checks the neutral verifier reproduces in ``extra_axes`` so a
multi-signer receipt cannot pass on the issuer signature alone:
  - parent counter-signature: REQUIRED when ``impact_tags`` is non-empty; the
    parent key signs the SAME stripped canonical payload as the issuer.
  - PDP binding: REQUIRED when ``impact_tags`` is non-empty; the PDP key signs the
    canonical tuple ``{context_hash_sha256, in_policy, policy_hash}``, which binds
    the verdict to one context so a verdict for context A cannot be replayed onto a
    receipt claiming context B.
A required-but-missing or present-but-invalid layer FAILs; an unverifiable layer
(its key not supplied) reports SKIPPED, which the core treats as not-fully-verified,
never a PASS.

Public keys are the raw 32-byte Ed25519 form; callers pass either raw bytes or an
RFC 8410 SPKI value, decoded by ``_raw_ed25519``.
"""
from __future__ import annotations

from typing import Any

from ..adapter import ChainStep, FormatAdapter, SignatureMaterial
from ..canonical import jcs
from ..core import sha256_hex
from ..crypto import FAIL, PASS, SKIPPED, verify_signature

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


def _resolve_raw(key_provider: Any, kid: str) -> bytes | None:
    """Return the raw 32-byte Ed25519 key for ``kid`` from the provider, or None."""
    material = (key_provider or {}).get(kid)
    if material is None:
        return None
    return _raw_ed25519(material if isinstance(material, bytes) else bytes.fromhex(material))


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

    def extra_axes(self, doc: dict, key_provider: Any) -> list[tuple[str, str, str]]:
        """Parent counter-signature and PDP binding per the AERF procedure."""
        axes: list[tuple[str, str, str]] = []
        has_impact = bool(doc.get("impact_tags"))
        if has_impact or doc.get("parent_signature"):
            axes.append(self._parent_axis(doc, key_provider, has_impact))
        if has_impact or doc.get("pdp_signature"):
            axes.append(self._pdp_axis(doc, key_provider, has_impact))
        return axes

    def _parent_axis(self, doc: dict, key_provider: Any, required: bool) -> tuple[str, str, str]:
        sig_hex = doc.get("parent_signature")
        if not sig_hex:
            if required:
                return ("parent_signature", FAIL, "parent_signature absent")
            return ("parent_signature", SKIPPED, "no parent_signature on this receipt")
        pk = _resolve_raw(key_provider, doc.get("parent_key_id", ""))
        if pk is None:
            return ("parent_signature", SKIPPED, "parent_key_id not supplied to the key provider")
        res, why = verify_signature("Ed25519", pk, self.signing_input(doc), bytes.fromhex(sig_hex))
        note = "parent counter-signature valid" if res == PASS else f"parent signature verification FAILED: {why}"
        return ("parent_signature", res, note)

    def _pdp_axis(self, doc: dict, key_provider: Any, required: bool) -> tuple[str, str, str]:
        sig_hex = doc.get("pdp_signature")
        if not sig_hex:
            if required:
                return ("pdp_signature", FAIL, "pdp_signature absent")
            return ("pdp_signature", SKIPPED, "no pdp_signature on this receipt")
        pk = _resolve_raw(key_provider, doc.get("pdp_key_id", ""))
        if pk is None:
            return ("pdp_signature", SKIPPED, "pdp_key_id not supplied to the key provider")
        tuple_bytes = jcs(
            {
                "context_hash_sha256": doc.get("context_hash_sha256"),
                "in_policy": doc.get("in_policy"),
                "policy_hash": doc.get("policy_hash"),
            }
        )
        res, why = verify_signature("Ed25519", pk, tuple_bytes, bytes.fromhex(sig_hex))
        note = "pdp binding valid" if res == PASS else f"pdp signature verification FAILED: {why}"
        return ("pdp_signature", res, note)
