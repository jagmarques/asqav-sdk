"""Oversight-attestation states the platform reports, gated on the signature."""

from __future__ import annotations

from .oracle.adapters.asqav_native import _safe_b64
from .oracle.canonical import asqav_jcs
from .oracle.crypto import PASS, verify_signature
from .verify_receipt import check_key_status, revoked_at_of

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

#: The attestation signature members, each a non-blank string.
SIGNATURE_MEMBERS = ("alg", "kid", "sig")


#: The reviewer members the signature binds beside the attestation.
REVIEWER_MEMBERS = ("principal", "role", "authority")

#: The ruling members the attestation signature binds.
RULING_MEMBERS = ("issued_at", "judged_action_refs", "ruling")


def attestation_message(payload: dict, attestation: dict) -> bytes:
    """Canonical bytes of the attestation minus its own signature envelope,
    bound to the reviewer's identity and the ruling it backs."""
    reviewer = payload.get("reviewer")
    reviewer = reviewer if isinstance(reviewer, dict) else {}
    return asqav_jcs(
        {
            "attestation": {k: v for k, v in attestation.items() if k != "signature"},
            "reviewer": {k: reviewer.get(k) for k in REVIEWER_MEMBERS},
            **{k: payload.get(k) for k in RULING_MEMBERS},
        }
    )


def reviewer_key(keys: object, principal: object) -> dict | None:
    """The one trusted key bound to the reviewer's principal, not a wire key."""
    if not isinstance(keys, dict) or not isinstance(principal, str) or not principal.strip():
        return None
    entries = keys.get("keys")
    bound = [
        entry
        for entry in (entries if isinstance(entries, list) else [])
        if isinstance(entry, dict) and entry.get("principal") == principal
    ]
    return bound[0] if len(bound) == 1 else None


def _signature_verifies(payload: dict, attestation: dict, reviewer_key: dict) -> bool:
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
        attestation_message(payload, attestation),
        _safe_b64(signature["sig"]),
    )
    return result == PASS


def oversight_attestation(payload: object, keys: object = None) -> str:
    """Verified only when the attestation signature checks under the trusted
    key bound to the reviewer; every other listed method is asserted."""
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
    key = reviewer_key(keys, reviewer.get("principal"))
    if method == "manual_assertion" or key is None:
        return ASSERTED
    issued_at = payload.get("issued_at")
    status, _note = check_key_status(
        key.get("status"),
        issued_at if isinstance(issued_at, str) else "",
        revoked_at_of(key),
    )
    if status != PASS:
        return ASSERTED
    return SIGNATURE_VERIFIED if _signature_verifies(payload, attestation, key) else ASSERTED
