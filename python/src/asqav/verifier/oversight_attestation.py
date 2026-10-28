"""Oversight-attestation states the platform reports, gated on the signature."""

from __future__ import annotations

from .oracle.runner import NOT_APPLICABLE

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


def oversight_attestation(payload: object, signature_ok: bool) -> str:
    """Report the reviewer's claim; only a verified signature backs it."""
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
    if method == "manual_assertion":
        return ASSERTED
    return SIGNATURE_VERIFIED if signature_ok else ASSERTED
