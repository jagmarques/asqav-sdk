"""W3C VC 2.0 adapter - DataIntegrityProof with the eddsa-jcs-2022 cryptosuite.

A Verifiable Credential 2.0 envelope (``@context`` starting with the credentials/v2
URL, ``type`` including ``VerifiableCredential``, ``issuer``, ``credentialSubject``)
secured by a ``DataIntegrityProof`` whose ``cryptosuite`` is ``eddsa-jcs-2022``
(W3C TR vc-di-eddsa): Ed25519 (RFC 8032) signs
``SHA-256(JCS(proofOptions)) || SHA-256(JCS(unsecuredDocument))`` where JCS is
strict RFC 8785, proofOptions is ``proof`` with ``proofValue`` removed, and
unsecuredDocument is the credential with ``proof`` removed. ``proofValue`` is the
raw 64-byte signature multibase base58btc ('z') encoded.

When the proof options carry their own ``@context`` the spec's transform requires
the document ``@context`` to start with it in order and canonicalises the document
with the proof's ``@context`` substituted; both rules are enforced here.

The oracle performs NO network DID resolution: ``proof.verificationMethod``
resolves through the shared DID resolver (did:key inline; did:web and every other
method from the injected map, either a raw key or the DID document the fetch would
have returned). Structure is intentionally stricter than the suite spec, which does
not normatively check proofPurpose: this adapter requires ``assertionMethod`` and
binds the verificationMethod DID to the issuer DID, fail-closed, mirroring the
agentreceipts adapter. A W3C VC carries no in-band chain link, so the chain axis
reports genesis.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from ..adapter import ChainStep, FormatAdapter, SignatureMaterial
from ..canonical import jcs_rfc8785
from ..did import b58btc_decode, resolve_ed25519_key

#: First @context value a VC 2.0 credential must carry.
_VC_V2_CONTEXT = "https://www.w3.org/ns/credentials/v2"
_PROOF_TYPE = "DataIntegrityProof"
_CRYPTOSUITE = "eddsa-jcs-2022"


    # The doc's proof member when it is a single object; {} for absent/list proofs.
def _proof(doc: dict) -> dict:
    proof = doc.get("proof")
    return proof if isinstance(proof, dict) else {}


    # The issuer DID whether issuer is a bare string or an object with an id.
def _issuer_did(doc: dict) -> str | None:
    issuer = doc.get("issuer")
    if isinstance(issuer, str):
        return issuer
    if isinstance(issuer, dict) and isinstance(issuer.get("id"), str):
        return issuer["id"]
    return None


    # xsd:datetime (RFC 3339 profile) -> aware datetime; None when unparseable.
def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


    # proof minus proofValue - the exact object the spec canonicalises for hash one.
def _proof_options(doc: dict) -> dict:
    return {k: v for k, v in _proof(doc).items() if k != "proofValue"}


    # Document minus proof; the proof's @context substitutes the document's when present.
def _unsecured(doc: dict) -> dict:
    body = {k: v for k, v in doc.items() if k != "proof"}
    ctx = _proof(doc).get("@context")
    if ctx is not None:
        body["@context"] = ctx
    return body


    # True when the proof's @context is absent or prefixes the document @context in order.
def _context_prefix_ok(doc: dict) -> bool:
    ctx = _proof(doc).get("@context")
    if ctx is None:
        return True
    if not isinstance(ctx, list):
        return False
    doc_ctx = doc.get("@context")
    return isinstance(doc_ctx, list) and doc_ctx[: len(ctx)] == ctx


    # Decode a multibase 'z' (base58btc) proofValue to raw signature bytes.
def _multibase_z_decode(value: str) -> bytes:
    if not value or value[0] != "z":
        raise ValueError("proofValue is not multibase 'z' (base58btc) encoded")
    return b58btc_decode(value[1:])


    # W3C VC 2.0 - DataIntegrityProof eddsa-jcs-2022, Ed25519 over RFC 8785 JCS.
class W3cVcAdapter(FormatAdapter):

    name = "w3c-vc"

    def detect(self, doc: dict) -> bool:
        types = doc.get("type")
        if not isinstance(types, list) or "VerifiableCredential" not in types:
            return False
        proof = doc.get("proof")
        candidates = [proof] if isinstance(proof, dict) else (proof if isinstance(proof, list) else [])
        # Any DataIntegrityProof on a VC routes here; the cryptosuite check belongs
        # to the schema axis so a sibling suite reports as an algorithm mismatch
        return any(isinstance(p, dict) and p.get("type") == _PROOF_TYPE for p in candidates)

    def extract_signature(self, doc: dict) -> SignatureMaterial:
        proof = _proof(doc)
        value = proof.get("proofValue")
        try:
            sig = _multibase_z_decode(value if isinstance(value, str) else "")
        except ValueError:
            sig = b""
        vm = proof.get("verificationMethod")
        return SignatureMaterial(sig=sig, alg="EdDSA", kid=vm if isinstance(vm, str) else "")

    def resolve_key(self, doc: dict, key_provider: Any) -> tuple[bytes | None, str]:
        vm = _proof(doc).get("verificationMethod")
        return resolve_ed25519_key(vm if isinstance(vm, str) else "", key_provider)

    def signing_input(self, doc: dict) -> bytes:
        # hashData = SHA-256(JCS(proofOptions)) || SHA-256(JCS(unsecured)) (TR vc-di-eddsa)
        options_hash = hashlib.sha256(jcs_rfc8785(_proof_options(doc))).digest()
        document_hash = hashlib.sha256(jcs_rfc8785(_unsecured(doc))).digest()
        return options_hash + document_hash

    def chain_step(self, doc: dict) -> ChainStep:
        # A W3C VC carries no in-band chain link of its own.
        return ChainStep(prev_field=None, is_genesis=True, recompute=lambda _pred: "")

    def schema(self, doc: dict) -> tuple[str, str]:
        ctx = doc.get("@context")
        if not isinstance(ctx, list) or not ctx or ctx[0] != _VC_V2_CONTEXT:
            return "FAIL", "first @context must be the W3C VC 2.0 credentials context"
        types = doc.get("type")
        if not isinstance(types, list) or "VerifiableCredential" not in types:
            return "FAIL", "type must include VerifiableCredential"
        proof = doc.get("proof")
        if isinstance(proof, list):
            return "FAIL", "proof sets are not supported; exactly one proof object is required"
        if not isinstance(proof, dict):
            return "FAIL", "missing required VC fields: proof"
        if proof.get("type") != _PROOF_TYPE:
            return "FAIL", "proof.type must be DataIntegrityProof"
        suite = proof.get("cryptosuite")
        if suite != _CRYPTOSUITE:
            return (
                "FAIL",
                f"unsupported signature algorithm: cryptosuite {suite!r} "
                f"(this verifier checks {_CRYPTOSUITE!r})",
            )
        vm = proof.get("verificationMethod")
        if not isinstance(vm, str) or not vm.startswith("did:"):
            return "FAIL", "proof.verificationMethod must be a DID URL"
        if proof.get("proofPurpose") != "assertionMethod":
            return "FAIL", "proof.proofPurpose must be assertionMethod"
        issuer = _issuer_did(doc)
        if issuer is None or not issuer.startswith("did:"):
            return "FAIL", "issuer must be a DID"
        if vm.split("#", 1)[0] != issuer:
            # bind the signing key to the issuer or anyone self-signs as a victim
            return "FAIL", "proof.verificationMethod is not controlled by issuer (signing-key DID != issuer DID)"
        subject = doc.get("credentialSubject")
        if not isinstance(subject, dict) and not (isinstance(subject, list) and subject):
            return "FAIL", "missing required VC fields: credentialSubject"
        created = proof.get("created")
        if created is not None and _parse_datetime(created) is None:
            return "FAIL", f"unreadable proof.created: {created!r}"
        if not _context_prefix_ok(doc):
            return "FAIL", "proof @context is not a prefix of the document @context"
        return "PASS", "required VC 2.0 fields present; DataIntegrityProof eddsa-jcs-2022"

    def extra_axes(self, doc: dict, key_provider: Any) -> list[tuple[str, str, str]]:
        """validFrom/validUntil bound validity; the expiry axis never folds the verdict (426)."""
        now = datetime.now(timezone.utc)
        valid_from = doc.get("validFrom")
        if valid_from is not None:
            parsed = _parse_datetime(valid_from)
            if parsed is None:
                return [("expiry", "FAIL", f"unreadable expires_at (validFrom {valid_from!r})")]
            if now < parsed:
                return [("expiry", "FAIL", f"not yet valid: validFrom {valid_from}")]
        valid_until = doc.get("validUntil")
        if valid_until is not None:
            parsed = _parse_datetime(valid_until)
            if parsed is None:
                return [("expiry", "FAIL", f"unreadable expires_at (validUntil {valid_until!r})")]
            if now >= parsed:
                return [("expiry", "FAIL", f"expired at {valid_until}")]
        return [("expiry", "PASS", "no validFrom/validUntil constraint breached")]
