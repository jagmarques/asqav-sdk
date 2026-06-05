"""Export-only shim: an Asqav receipt presented in W3C Verifiable Credential 2.0 shape.

EU-lane presentation. This is NOT a wire token, NOT a first-class internal type, and
NOT a verification path. ``to_vc_envelope`` maps an Asqav receipt's action/agent fields
into a VC ``credentialSubject`` so a VC-shaped consumer can read them, but it carries the
ORIGINAL Asqav signature verbatim - it does not re-sign anything.

WHY THE PROOF CANNOT BE A STANDARD VC SUITE
-------------------------------------------
An Asqav receipt is signed with ML-DSA-65 (or Ed25519 / ES256) over the Asqav canonical
bytes of the *receipt's own* payload. A registered VC proof (``Ed25519Signature2020``,
``DataIntegrityProof`` with ``eddsa-rdfc-2022``, ...) is a signature over the *VC's* own
canonical bytes. Those two byte strings differ, so relabelling the Asqav signature as a
standard suite would produce a credential that any conformant VC verifier would reject -
a forged-looking VC. To stay honest the export emits an explicit, self-descriptive
NON-STANDARD proof type (``AsqavReceiptSignature2026``) that:

  - carries the original signature value (base64 / hex) and its algorithm and key id,
  - points at the EXACT canonical bytes the signature covers (``signingInput``) so an
    Asqav-aware verifier can re-check, and
  - never claims to be a registered VC cryptosuite.

A top-level ``proofIsAsqavNative: true`` marker makes the boundary unmistakable: this is
an Asqav receipt wearing a VC envelope, not a standard-suite-signed VC. Verifying it
needs an Asqav-aware verifier; a generic VC verifier MUST decline it.
"""
from __future__ import annotations

from typing import Any

from .oracle import ADAPTERS

#: The VC ``@context`` entries - the v2 base plus the Asqav presentation vocabulary.
VC_V2_CONTEXT = "https://www.w3.org/ns/credentials/v2"
ASQAV_CONTEXT = "https://asqav.com/context/compliance-receipt/v1"

#: The non-standard, self-descriptive proof type. NOT a registered VC cryptosuite.
ASQAV_PROOF_TYPE = "AsqavReceiptSignature2026"

#: The credential type pair. Deliberately NOT ``AgentReceipt`` - that token belongs to a
#: different, standard-suite-signed format and must not be borrowed here.
VC_CREDENTIAL_TYPE = ["VerifiableCredential", "AsqavComplianceReceipt"]

#: Registered VC proof suites the export must never claim to be (honesty guard reference).
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


def _detect_format(receipt: dict) -> str:
    """Reuse the oracle's detection to label the source format, or ``unknown``."""
    for adapter in ADAPTERS:
        try:
            if adapter.detect(receipt):
                return adapter.name
        except Exception:
            continue
    return "unknown"


def _signing_input_b64(receipt: dict, fmt: str) -> str | None:
    """Base64 of the exact canonical bytes the source signature covers, if reproducible.

    The oracle's adapter for this format already encodes the signing input rule. Reusing
    it keeps the pointer honest: it is the same bytes the verifier would re-derive, not a
    second guess. ``None`` when the format is unknown or the bytes cannot be rebuilt.
    """
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
    """Map an Asqav-native receipt (compliance or hash mode) to (subject, proofbits, issuer).

    ``proofbits`` is ``{sig, alg, kid}`` pulled verbatim from the source so the proof block
    preserves the original signature byte-for-byte.
    """
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
    """Map a W3C-VC AgentReceipt's fields into the Asqav presentation subject + proofbits.

    The source proof is Ed25519 over the receipt's strict-JCS bytes; its ``proofValue`` is
    carried verbatim. The export still re-labels it Asqav-native, because once the
    envelope shape changes (type + credentialSubject re-keyed) the original proof no longer
    verifies as a standard VC over THESE bytes - so presenting it as standard would lie.
    """
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
    # Best-effort generic mapping for the other oracle formats; the proof still carries the
    # raw signature and stays non-standard, so no false standard-VC claim is possible.
    payload = receipt.get("payload") if isinstance(receipt.get("payload"), dict) else receipt
    sig = receipt.get("signature")
    if isinstance(sig, dict):
        proofbits = {"sig": sig.get("sig"), "alg": sig.get("alg"), "kid": sig.get("kid")}
    else:
        proofbits = {"sig": sig, "alg": payload.get("alg"), "kid": payload.get("key_id")}
    subject = {"agent": {"id": payload.get("agent") or payload.get("agent_id")}, "source": payload}
    return subject, proofbits, payload.get("issuer") or payload.get("issuer_id")


def to_vc_envelope(receipt: dict, *, format: str | None = None) -> dict:
    """Present an Asqav receipt as a W3C Verifiable Credential 2.0 envelope (export only).

    Pure and offline: no signing, no network, no mutation of ``receipt``. The returned
    credential's ``proof`` is the NON-STANDARD ``AsqavReceiptSignature2026`` carrying the
    original Asqav signature verbatim and a pointer to the bytes it covers. It is NOT a
    standard-suite-signed VC and a generic VC verifier MUST decline it.

    Args:
      receipt: a parsed Asqav (or oracle-recognised) receipt dict.
      format: optional format hint; when omitted the oracle's detection is reused.

    Returns:
      A VC 2.0 dict with ``proofIsAsqavNative: True``.
    """
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


def _valid_from(subject: dict) -> str | None:
    """Best-effort issuance timestamp from the mapped action or source payload, or ``None``."""
    action = subject.get("action")
    if isinstance(action, dict) and action.get("timestamp"):
        return action.get("timestamp")
    source = subject.get("source")
    if isinstance(source, dict):
        for key in ("issued_at", "issuanceDate", "timestamp", "validFrom"):
            if source.get(key):
                return source[key]
    return None
