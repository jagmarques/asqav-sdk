"""Export-only shim: an Asqav receipt in W3C VC 2.0 shape. EU-lane
presentation, not a wire token, internal type, or verification path."""
from __future__ import annotations

from typing import Any

from .oracle import ADAPTERS

#: The VC ``@context`` entries - v2 base plus the Asqav presentation vocabulary.
VC_V2_CONTEXT = "https://www.w3.org/ns/credentials/v2"
ASQAV_CONTEXT = "https://asqav.com/context/compliance-receipt/v1"

#: The non-standard, self-descriptive proof type. NOT a VC cryptosuite.
ASQAV_PROOF_TYPE = "AsqavReceiptSignature2026"

#: The credential type pair. NOT ``AgentReceipt`` - that token belongs to a
#: different, standard-suite-signed format and must not be borrowed here.
VC_CREDENTIAL_TYPE = ["VerifiableCredential", "AsqavComplianceReceipt"]

#: Registered VC proof suites the export must never claim to be.
STANDARD_VC_PROOF_TYPES = frozenset(
    {
        "Ed25519Signature2020",
        "Ed25519Signature2018",
        "DataIntegrityProof",
        "JsonWebSignature2020",
        "EcdsaSecp256k1Signature2019",
        "BbsBlsSignature2020",
        "RsaSignature2018",
    }
)


    # Reuse the oracle's detection to label the source format, or ``unknown``.
def _detect_format(receipt: dict) -> str:
    for adapter in ADAPTERS:
        try:
            if adapter.detect(receipt):
                return adapter.name
        except Exception:
            continue
    return "unknown"


def _signing_input_b64(receipt: dict, fmt: str) -> str | None:
    """Base64 of the exact canonical bytes the source signature covers,
    if reproducible; the oracle adapter encodes the signing-input rule."""
    import base64

    for adapter in ADAPTERS:
        if adapter.name != fmt:
            continue
        try:
            return base64.b64encode(adapter.signing_input(receipt)).decode("ascii")
        except Exception:
            return None
    return None


def _native_subject(receipt: dict) -> tuple[dict, dict, str | None]:
    """Map an Asqav-native receipt (compliance or hash) to subject, proofbits,
    issuer."""
    if receipt.get("mode") == "hash" and receipt.get("payload") is None:
        # Flat hash-mode /sign output.
        subject = {
            "agent": {"id": receipt.get("agent_id")},
            "organization": {"id": receipt.get("org_id")},
            "action": {
                "id": receipt.get("action_id"),
                "contentHash": receipt.get("hash"),
                "hashAlgorithm": receipt.get("hash_algo"),
                "timestamp": receipt.get("server_timestamp"),
            },
            "policy": {
                "digest": receipt.get("policy_digest"),
                "decision": receipt.get("policy_decision"),
            },
        }
        proofbits = {
            "sig": receipt.get("signature_b64") or receipt.get("signature"),
            "alg": receipt.get("algorithm", "ML-DSA-65"),
            "kid": receipt.get("key_id"),
        }
        return subject, proofbits, receipt.get("org_id")

    # Compliance-mode envelope: signed payload + signature object.
    payload = receipt.get("payload") if isinstance(receipt.get("payload"), dict) else receipt
    sig_obj = receipt.get("signature")
    if isinstance(sig_obj, dict):
        proofbits = {"sig": sig_obj.get("sig"), "alg": sig_obj.get("alg"), "kid": sig_obj.get("kid")}
    else:
        proofbits = {"sig": sig_obj, "alg": "ML-DSA-65", "kid": payload.get("issuer_id")}
    subject = {
        "agent": {"id": payload.get("agent_id")},
        "action": {
            "type": payload.get("type"),
            "reference": payload.get("action_ref"),
            "toolName": payload.get("tool_name"),
            "timestamp": payload.get("issued_at"),
        },
        "policy": {
            "digest": payload.get("policy_digest"),
            "decision": payload.get("decision"),
        },
    }
    return subject, proofbits, payload.get("issuer_id")


def _agentreceipts_subject(receipt: dict) -> tuple[dict, dict, str | None]:
    """Map a W3C-VC AgentReceipt's fields into Asqav subject + proofbits."""
    sub = receipt.get("credentialSubject", {}) if isinstance(receipt.get("credentialSubject"), dict) else {}
    proof = receipt.get("proof", {}) if isinstance(receipt.get("proof"), dict) else {}
    issuer = receipt.get("issuer")
    issuer_id = issuer.get("id") if isinstance(issuer, dict) else issuer
    subject = {
        "principal": sub.get("principal"),
        "agent": {"id": issuer_id},
        "action": sub.get("action"),
        "outcome": sub.get("outcome"),
        "chain": sub.get("chain"),
    }
    proofbits = {
        "sig": proof.get("proofValue"),
        "alg": "Ed25519",
        "kid": proof.get("verificationMethod"),
        "sourceProofType": proof.get("type"),
    }
    return subject, proofbits, issuer_id


def _subject_for(receipt: dict, fmt: str) -> tuple[dict, dict, str | None]:
    if fmt == "agentreceipts":
        return _agentreceipts_subject(receipt)
    if fmt == "asqav-native":
        return _native_subject(receipt)
    # Best-effort generic mapping for the other oracle formats; the proof still
    # carries the raw signature and stays non-standard: no standard claim.
    payload = receipt.get("payload") if isinstance(receipt.get("payload"), dict) else receipt
    sig = receipt.get("signature")
    if isinstance(sig, dict):
        proofbits = {"sig": sig.get("sig"), "alg": sig.get("alg"), "kid": sig.get("kid")}
    else:
        proofbits = {"sig": sig, "alg": payload.get("alg"), "kid": payload.get("key_id")}
    subject = {"agent": {"id": payload.get("agent") or payload.get("agent_id")}, "source": payload}
    return subject, proofbits, payload.get("issuer") or payload.get("issuer_id")


def to_vc_envelope(receipt: dict, *, format: str | None = None) -> dict:
    """Present an Asqav receipt as a W3C VC 2.0 envelope (export only).
    Pure and offline: no signing, no network, no mutation of ``receipt``."""
    if not isinstance(receipt, dict):
        raise TypeError("receipt must be a parsed JSON object (dict)")

    fmt = format or _detect_format(receipt)
    subject, proofbits, issuer_id = _subject_for(receipt, fmt)
    valid_from = _valid_from(subject)

    proof: dict[str, Any] = {
        "type": ASQAV_PROOF_TYPE,
        "proofPurpose": "assertionMethod",
        "asqavAlgorithm": proofbits.get("alg"),
        "verificationKeyId": proofbits.get("kid"),
        "signatureValue": proofbits.get("sig"),
        "signingInputBase64": _signing_input_b64(receipt, fmt),
        "sourceFormat": fmt,
        "note": (
            "Asqav-native signature over the Asqav canonical bytes of the source receipt; "
            "re-check requires an Asqav-aware verifier, not a standard VC suite."
        ),
    }
    if proofbits.get("sourceProofType"):
        proof["sourceProofType"] = proofbits["sourceProofType"]

    envelope: dict[str, Any] = {
        "@context": [VC_V2_CONTEXT, ASQAV_CONTEXT],
        "type": list(VC_CREDENTIAL_TYPE),
        "issuer": {"id": issuer_id} if issuer_id is not None else {},
        "validFrom": valid_from,
        "issuanceDate": valid_from,  # carried for VC 1.1 readers; v2 uses validFrom
        "credentialSubject": subject,
        "proof": proof,
        "proofIsAsqavNative": True,
    }
    return envelope


    # Best-effort issuance timestamp from the action or payload, else None.
def _valid_from(subject: dict) -> str | None:
    action = subject.get("action")
    if isinstance(action, dict) and action.get("timestamp"):
        return action.get("timestamp")
    source = subject.get("source")
    if isinstance(source, dict):
        for key in ("issued_at", "issuanceDate", "timestamp", "validFrom"):
            if source.get(key):
                return source[key]
    return None
