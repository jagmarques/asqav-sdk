"""Oversight-attestation states the platform reports, gated on the signature."""

from __future__ import annotations

from .oracle.adapters.asqav_native import _safe_b64
from .oracle.canonical import asqav_jcs
from .oracle.crypto import PASS, verify_signature

#: The token every axis and report uses when a check does not apply.
NOT_APPLICABLE = "not_applicable"

STATES = {
    "missing": "missing/unverified",
    "asserted": "asserted/unverified",
    "signature_verified": "signature-verified",
}
MISSING = STATES["missing"]
ASSERTED = STATES["asserted"]
SIGNATURE_VERIFIED = STATES["signature_verified"]

METHODS = frozenset(
    {"manual_assertion", "sso", "hardware_key", "signed_statement"}
)

#: The vector file pinning the reviewer key; the receipt never supplies it.
REVIEWER_KEY_FILE = "reviewer_key.json"
#: The attestation signature members, each a non-blank string.
SIGNATURE_MEMBERS = ("alg", "kid", "sig")


def attestation_message(attestation: dict) -> bytes:
    """Canonical bytes of the attestation minus its own signature envelope."""
    return asqav_jcs({k: v for k, v in attestation.items() if k != "signature"})


def _signature_verifies(attestation: dict, reviewer_key: dict) -> bool:
    signature = attestation.get("signature")
    if not isinstance(signature, dict) or not all(
        isinstance(signature.get(k), str) and signature[k].strip()
        for k in SIGNATURE_MEMBERS
    ):
        return False
    alg = reviewer_key.get("alg")
    if not isinstance(alg, str) or alg.upper() != signature["alg"].upper():
        return False
    result, _note = verify_signature(
        alg,
        _safe_b64(reviewer_key.get("public_key")),
        attestation_message(attestation),
        _safe_b64(signature["sig"]),
    )
    return result == PASS


def oversight_attestation(payload: object, reviewer_key: object = None) -> str:
    """Verified only when the attestation signature checks under the pinned
    reviewer key; every other listed method is asserted."""
    if not isinstance(payload, dict):
        return NOT_APPLICABLE
    if payload.get("type") != "protectmcp:lifecycle:oversight_ruling":
        return NOT_APPLICABLE
    reviewer = payload.get("reviewer")
    attestation = reviewer.get("attestation") if isinstance(reviewer, dict) else None
    if not isinstance(attestation, dict):
        return MISSING
    method = attestation.get("method")
    if not isinstance(method, str) or method not in METHODS:
        return MISSING
    if method == "manual_assertion" or not isinstance(reviewer_key, dict):
        return ASSERTED
    return SIGNATURE_VERIFIED if _signature_verifies(attestation, reviewer_key) else ASSERTED
