"""Standards-interop "doors": one Asqav receipt, many native envelopes.

Each door wraps the SAME receipt object verbatim into a standard envelope an agent
framework already consumes: a W3C Verifiable Credential, a CloudEvents event, an OTel
GenAI attribute map, a C2PA assertion, and an ERC-8004 validationRequest shape.

The inner receipt is byte-identical across every door: the receipt dict is embedded
unchanged (VC ``credentialSubject``, CloudEvents ``data``, C2PA ``data.receipt``,
ERC-8004 ``receipt``) or, for the flat OTel attribute map, carried as its JCS string
under ``asqav.receipt``. Every door has an inverse that returns the exact same receipt.

All functions are pure and offline: no signing, no network, no mutation of the input,
no wall-clock reads. Timestamps and ids come only from the receipt itself, so the same
input always yields the same envelope in both the Python and TypeScript SDKs.

This is presentation only. A door does NOT re-sign; the authoritative Asqav signature
stays inside the embedded receipt. A generic VC / C2PA verifier reads the shape but must
use an Asqav-aware verifier to check the signature.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from ._jcs import canonical_json

__all__ = [
    "canonical_receipt_bytes",
    "receipt_digest",
    "to_w3c_vc",
    "to_cloudevent",
    "to_otel_genai_attributes",
    "to_c2pa_assertion",
    "to_erc8004_validation_request",
    "wrap_all",
    "receipt_from_vc",
    "receipt_from_cloudevent",
    "receipt_from_otel_genai_attributes",
    "receipt_from_c2pa_assertion",
    "receipt_from_erc8004_validation_request",
    "extract_receipt",
]

VC_V2_CONTEXT = "https://www.w3.org/ns/credentials/v2"
ASQAV_DOORS_CONTEXT = "https://asqav.com/context/doors/v1"
VC_CREDENTIAL_TYPE = ["VerifiableCredential", "AsqavReceiptCredential"]
VC_PROOF_TYPE = "AsqavReceiptSignature2026"

CLOUDEVENT_TYPE = "com.asqav.receipt.v1"
CLOUDEVENT_DIGEST_ATTR = "asqavreceiptdigest"

C2PA_ASSERTION_LABEL = "com.asqav.receipt.v1"

OTEL_RECEIPT_ATTR = "asqav.receipt"
OTEL_DIGEST_ATTR = "asqav.receipt.digest"
OTEL_SIGNATURE_ATTR = "asqav.signature"

ERC8004_ZERO_ADDRESS = "0x" + "00" * 20


def canonical_receipt_bytes(receipt: dict) -> bytes:
    """Return the JCS canonical bytes of ``receipt``; the same bytes both SDKs hash."""
    return canonical_json(receipt)


def receipt_digest(receipt: dict) -> str:
    """``sha256:<hex>`` over the raw canonical receipt bytes (the C2PA raw-bytes rule)."""
    return "sha256:" + hashlib.sha256(canonical_receipt_bytes(receipt)).hexdigest()


def _require_dict(receipt: Any) -> dict:
    if not isinstance(receipt, dict):
        raise TypeError("receipt must be a parsed JSON object (dict)")
    return receipt


def _payload(receipt: dict) -> dict:
    payload = receipt.get("payload")
    return payload if isinstance(payload, dict) else receipt


def _signature_bits(receipt: dict) -> tuple[Any, Any, Any]:
    """Return ``(sig, alg, kid)`` for compliance-mode or flat hash-mode receipts."""
    sig = receipt.get("signature")
    if isinstance(sig, dict):
        return sig.get("sig"), sig.get("alg"), sig.get("kid")
    if sig is not None:
        return sig, receipt.get("algorithm"), receipt.get("key_id")
    return (
        receipt.get("signature_b64"),
        receipt.get("algorithm"),
        receipt.get("key_id"),
    )


def _field(receipt: dict, *keys: str) -> Any:
    payload = _payload(receipt)
    for key in keys:
        if payload.get(key) is not None:
            return payload[key]
        if receipt.get(key) is not None:
            return receipt[key]
    return None


def _agent_id(receipt: dict) -> Any:
    return _field(receipt, "agent_id")


def _action_id(receipt: dict) -> Any:
    return _field(receipt, "action_id", "action_ref")


def _issuer_id(receipt: dict) -> Any:
    return _field(receipt, "issuer_id", "org_id")


def _timestamp(receipt: dict) -> Any:
    return _field(receipt, "issued_at", "server_timestamp")


def _verify_urn(receipt: dict) -> str:
    """A deterministic offline pointer to the receipt by digest. Not a live endpoint."""
    return "urn:asqav:receipt:" + receipt_digest(receipt).split(":", 1)[1]


def to_w3c_vc(receipt: dict, *, issuer: str | None = None) -> dict:
    """Wrap the receipt as a W3C VC 2.0 with the whole receipt as ``credentialSubject``.

    ``proof`` is the non-standard ``AsqavReceiptSignature2026`` referencing the receipt's
    own signature. It is not a registered VC cryptosuite, so a generic VC verifier must
    decline it and defer to an Asqav-aware verifier.
    """
    _require_dict(receipt)
    sig, alg, kid = _signature_bits(receipt)
    issuer_id = issuer if issuer is not None else _issuer_id(receipt)
    vc: dict[str, Any] = {
        "@context": [VC_V2_CONTEXT, ASQAV_DOORS_CONTEXT],
        "type": list(VC_CREDENTIAL_TYPE),
        "issuer": issuer_id if issuer_id is not None else "https://asqav.com",
        "credentialSubject": receipt,
        "proof": {
            "type": VC_PROOF_TYPE,
            "proofPurpose": "assertionMethod",
            "signatureAlgorithm": alg,
            "verificationKeyId": kid,
            "signatureValue": sig,
            "receiptDigest": receipt_digest(receipt),
        },
    }
    ts = _timestamp(receipt)
    if ts is not None:
        vc["validFrom"] = ts
    return vc


def to_cloudevent(receipt: dict) -> dict:
    """Wrap the receipt as a CloudEvents 1.0 event with the receipt as ``data``.

    Adds one Asqav extension context attribute, ``asqavreceiptdigest``, binding the
    envelope to the exact receipt bytes.
    """
    _require_dict(receipt)
    agent = _agent_id(receipt)
    action = _action_id(receipt)
    event: dict[str, Any] = {
        "specversion": "1.0",
        "id": str(action) if action is not None else receipt_digest(receipt),
        "source": "urn:asqav:agent:" + str(agent) if agent is not None else "urn:asqav",
        "type": CLOUDEVENT_TYPE,
        "datacontenttype": "application/json",
        CLOUDEVENT_DIGEST_ATTR: receipt_digest(receipt),
        "data": receipt,
    }
    ts = _timestamp(receipt)
    if ts is not None:
        event["time"] = ts
    return event


def to_otel_genai_attributes(receipt: dict) -> dict:
    """Map the receipt to an OTel GenAI ``gen_ai.*`` attribute dict.

    The byte-identical inner receipt rides as the ``asqav.receipt`` JCS string so a
    consumer recovers the exact receipt from the attribute map.
    """
    _require_dict(receipt)
    sig, _alg, _kid = _signature_bits(receipt)
    attrs: dict[str, Any] = {}
    op = _field(receipt, "type", "action_type")
    if op is not None:
        attrs["gen_ai.operation.name"] = op
    agent = _agent_id(receipt)
    if agent is not None:
        attrs["gen_ai.agent.id"] = agent
    tool = _field(receipt, "tool_name")
    if tool is not None:
        attrs["gen_ai.tool.name"] = tool
    call_id = _action_id(receipt)
    if call_id is not None:
        attrs["gen_ai.tool.call.id"] = call_id
    conversation = _field(receipt, "session_id")
    if conversation is not None:
        attrs["gen_ai.conversation.id"] = conversation
    attrs[OTEL_RECEIPT_ATTR] = canonical_receipt_bytes(receipt).decode("utf-8")
    attrs[OTEL_DIGEST_ATTR] = receipt_digest(receipt)
    if sig is not None:
        attrs[OTEL_SIGNATURE_ATTR] = sig
    return attrs


def to_c2pa_assertion(receipt: dict) -> dict:
    """Wrap the receipt as a C2PA third-party assertion using the raw-bytes rule.

    ``hash_binding.hash`` is sha256 over the raw canonical receipt bytes directly, not a
    hash of a pre-computed digest, so it is the asset's own hash.
    """
    _require_dict(receipt)
    return {
        "label": C2PA_ASSERTION_LABEL,
        "data": {
            "receipt": receipt,
            "verify": _verify_urn(receipt),
        },
        "hash_binding": {
            "alg": "sha256",
            "rule": "raw-bytes",
            "hash": hashlib.sha256(canonical_receipt_bytes(receipt)).hexdigest(),
        },
    }


def to_erc8004_validation_request(
    receipt: dict,
    *,
    validator: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Wrap the receipt as an ERC-8004 ``validationRequest`` call shape.

    ``requestHash`` is a bytes32 sha256 over the canonical receipt bytes. The full receipt
    rides under ``receipt`` so the inner object stays byte-identical to the other doors.
    """
    _require_dict(receipt)
    resolved_agent = agent_id if agent_id is not None else _agent_id(receipt)
    digest_hex = hashlib.sha256(canonical_receipt_bytes(receipt)).hexdigest()
    return {
        "function": "validationRequest",
        "args": {
            "validator": validator if validator is not None else ERC8004_ZERO_ADDRESS,
            "agentId": resolved_agent,
            "requestURI": _verify_urn(receipt),
            "requestHash": "0x" + digest_hex,
        },
        "hashAlgorithm": "sha256",
        "receipt": receipt,
    }


def wrap_all(receipt: dict) -> dict:
    """Return every door for ``receipt`` keyed by standard name; one inner object, N skins."""
    _require_dict(receipt)
    return {
        "w3c_vc": to_w3c_vc(receipt),
        "cloudevent": to_cloudevent(receipt),
        "otel_genai": to_otel_genai_attributes(receipt),
        "c2pa": to_c2pa_assertion(receipt),
        "erc8004": to_erc8004_validation_request(receipt),
    }


def receipt_from_vc(vc: dict) -> dict:
    """Recover the inner receipt from a VC door."""
    return vc["credentialSubject"]


def receipt_from_cloudevent(event: dict) -> dict:
    """Recover the inner receipt from a CloudEvents door."""
    return event["data"]


def receipt_from_otel_genai_attributes(attrs: dict) -> dict:
    """Recover the inner receipt from an OTel GenAI attribute map."""
    return json.loads(attrs[OTEL_RECEIPT_ATTR])


def receipt_from_c2pa_assertion(assertion: dict) -> dict:
    """Recover the inner receipt from a C2PA door."""
    return assertion["data"]["receipt"]


def receipt_from_erc8004_validation_request(request: dict) -> dict:
    """Recover the inner receipt from an ERC-8004 door."""
    return request["receipt"]


def extract_receipt(envelope: dict) -> dict:
    """Recover the inner receipt from any door envelope, detected by shape."""
    if not isinstance(envelope, dict):
        raise TypeError("envelope must be a dict")
    if "credentialSubject" in envelope:
        return receipt_from_vc(envelope)
    if envelope.get("type") == CLOUDEVENT_TYPE or ("specversion" in envelope and "data" in envelope):
        return receipt_from_cloudevent(envelope)
    if OTEL_RECEIPT_ATTR in envelope:
        return receipt_from_otel_genai_attributes(envelope)
    if envelope.get("label") == C2PA_ASSERTION_LABEL:
        return receipt_from_c2pa_assertion(envelope)
    if envelope.get("function") == "validationRequest":
        return receipt_from_erc8004_validation_request(envelope)
    raise ValueError("unrecognised door envelope")
