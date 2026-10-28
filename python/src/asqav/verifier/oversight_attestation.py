"""Oversight-attestation states the platform reports, gated on the signature."""

from __future__ import annotations

from .oracle.core import VerifyResult
from .oracle.crypto import PASS

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

#: The receipt type whose reviewer claim these states report.
TYPE = "protectmcp:lifecycle:oversight_ruling"
MANUAL_ASSERTION = "manual_assertion"
METHODS = frozenset(
    {MANUAL_ASSERTION, "sso", "hardware_key", "signed_statement"}
)
#: The expected.json member a vector declares its oversight state in.
EXPECTED_KEY = "oversight_attestation"


def oversight_attestation(payload: object, result: VerifyResult) -> str:
    """Report the reviewer's claim; only a verified signature backs it."""
    if not isinstance(payload, dict):
        return NOT_APPLICABLE
    if payload.get("type") != TYPE:
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
    signature = result.axis("signature")
    signature_ok = signature is not None and signature.result == PASS
    return SIGNATURE_VERIFIED if signature_ok else ASSERTED
