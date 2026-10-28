"""Oversight-attestation states the platform reports, gated on the signature."""

from __future__ import annotations

from .oracle.runner import NOT_APPLICABLE

MISSING = "missing/unverified"
ASSERTED = "asserted/unverified"
SIGNATURE_VERIFIED = "signature-verified"

METHODS = frozenset(
    {"manual_assertion", "sso", "hardware_key", "signed_statement"}
)
MANUAL_ASSERTION = "manual_assertion"


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
    if method == MANUAL_ASSERTION:
        return ASSERTED
    return SIGNATURE_VERIFIED if signature_ok else ASSERTED
