"""agent-receipts adapter - W3C-VC AgentReceipt, Ed25519 over strict RFC 8785 JCS.

An AgentReceipt is a W3C Verifiable Credential 2.0 envelope: top-level ``@context``,
``id``, ``type`` (``["VerifiableCredential", "AgentReceipt"]``), ``issuer``,
``issuanceDate``, ``credentialSubject``, and a ``proof``. The signature is Ed25519
over the strict JCS bytes of the receipt with the ``proof`` member removed, signed
DIRECTLY (no pre-hash) per the agent-receipts signing rule. The ``proof.type`` is
``Ed25519Signature2020`` and ``proof.proofValue`` is multibase ``u`` (base64url, no
padding).

This adapter uses ``jcs_rfc8785`` (UTF-16 key sort, ECMAScript numbers), NOT the
AERF/ACTA ``jcs`` dialect: the agent-receipts producers sign true strict-JCS bytes and
the upstream canonicalization vectors pin that. The signing key is resolved from
``proof.verificationMethod`` through the shared DID resolver - did:key inline, every
other method from an injected map (the oracle never fetches a DID document).

Chain rule: ``credentialSubject.chain.previous_receipt_hash`` is ``"sha256:"`` +
hex(SHA-256(JCS(predecessor without proof))). Genesis carries the field present and
explicitly ``null`` (the field MUST always be present; ``null`` is not the same as
omitting it) - the opposite of the AERF omit-on-genesis rule.
"""
from __future__ import annotations

import base64
from typing import Any

from ..adapter import ChainStep, FormatAdapter, SignatureMaterial
from ..canonical import jcs_rfc8785
from ..core import sha256_hex
from ..did import resolve_ed25519_key

#: SHA-256 hash prefix the chain field carries before the hex digest.
_SHA256_PREFIX = "sha256:"


def _signable(doc: dict) -> dict:
    """The receipt with ``proof`` removed - the exact object the signature covers."""
    return {k: v for k, v in doc.items() if k != "proof"}


def _multibase_u_decode(value: str) -> bytes:
    """Decode a multibase ``u`` (base64url, no padding) proofValue to raw bytes."""
    if not value or value[0] != "u":
        raise ValueError("proofValue is not multibase 'u' (base64url) encoded")
    body = value[1:]
    return base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))


def _chain(doc: dict) -> dict:
    sub = doc.get("credentialSubject")
    return sub.get("chain", {}) if isinstance(sub, dict) else {}


class AgentReceiptsAdapter(FormatAdapter):
    """agent-receipts AgentReceipt - Ed25519 over strict RFC 8785 JCS, proof excluded."""

    name = "agentreceipts"

    def detect(self, doc: dict) -> bool:
        types = doc.get("type")
        proof = doc.get("proof")
        if not isinstance(types, list) or "AgentReceipt" not in types:
            return False
        return isinstance(proof, dict) and isinstance(proof.get("proofValue"), str)

    def extract_signature(self, doc: dict) -> SignatureMaterial:
        proof = doc.get("proof", {})
        try:
            sig = _multibase_u_decode(proof.get("proofValue", ""))
        except (ValueError, base64.binascii.Error):
            sig = b""
        return SignatureMaterial(sig=sig, alg="Ed25519", kid=proof.get("verificationMethod", ""))

    def resolve_key(self, doc: dict, key_provider: Any) -> tuple[bytes | None, str]:
        kid = doc.get("proof", {}).get("verificationMethod", "")
        return resolve_ed25519_key(kid, key_provider)

    def signing_input(self, doc: dict) -> bytes:
        # Strict-JCS bytes of the receipt minus proof, signed directly (no pre-hash).
        return jcs_rfc8785(_signable(doc))

    def chain_step(self, doc: dict) -> ChainStep:
        chain = _chain(doc)
        present = "previous_receipt_hash" in chain
        prev = chain.get("previous_receipt_hash")
        is_genesis = present and prev is None
        return ChainStep(
            prev_field=prev,
            is_genesis=is_genesis,
            recompute=lambda pred: _SHA256_PREFIX + sha256_hex(jcs_rfc8785(_signable(pred))),
        )

    def schema(self, doc: dict) -> tuple[str, str]:
        """Structural check; returns ``(result, note)``.

        Issuer binding: the strict ``verificationMethod`` DID == ``issuer.id`` check is
        intentionally stricter than the spec, which leaves controller indirection to DID
        resolution the oracle does not perform. It is fail-closed and revisited when
        DID-document resolution lands.
        """
        required = (
            "@context",
            "id",
            "type",
            "version",
            "issuer",
            "issuanceDate",
            "credentialSubject",
            "proof",
        )
        missing = [f for f in required if f not in doc]
        if missing:
            return "FAIL", f"missing required VC fields: {','.join(missing)}"
        types = doc.get("type")
        if not isinstance(types, list) or "VerifiableCredential" not in types or "AgentReceipt" not in types:
            return "FAIL", "type must include VerifiableCredential and AgentReceipt"
        if doc.get("proof", {}).get("type") != "Ed25519Signature2020":
            return "FAIL", "proof.type must be Ed25519Signature2020"
        issuer = doc.get("issuer")
        issuer_did = issuer.get("id") if isinstance(issuer, dict) else issuer
        vm_did = doc.get("proof", {}).get("verificationMethod", "").split("#")[0]
        if not vm_did or vm_did != issuer_did:
            # did:key carries its key in-receipt; bind to issuer or anyone self-signs as a victim
            return "FAIL", "proof.verificationMethod is not controlled by issuer (signing-key DID != issuer DID)"
        chain = _chain(doc)
        if "previous_receipt_hash" not in chain:
            return "FAIL", "chain.previous_receipt_hash must be present (null on genesis), never omitted"
        return "PASS", "required VC fields present; type AgentReceipt; chain link well-formed"
