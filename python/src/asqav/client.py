"""
asqav API Client - Thin SDK that connects to asqav.com.

This module provides the API client for connecting to asqav Cloud.
All ML-DSA cryptography happens server-side; this SDK handles
API communication and response parsing.

Example:
    import asqav

    # Initialize with API key
    asqav.init(api_key="sk_...")

    # Create an agent (server generates identity)
    agent = asqav.Agent.create("my-agent")

    # Sign an action (server signs with ML-DSA)
    signature = agent.sign("read:data", {"file": "config.json"})
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import json as _json
import logging
import os
import sys
import time
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import urljoin

if TYPE_CHECKING:
    from .phases import PhaseChain
    from .scope import ScopeToken

from .patterns import resolve_pattern
from .retry import with_retry

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger("asqav")

# Retry configuration for API calls
_MAX_RETRIES = 5
_RETRY_DELAYS = [0.5, 1.0, 2.0, 4.0, 8.0]  # Exponential backoff


def _with_retry(func: Callable[[], Any]) -> Any:
    """Execute function with exponential backoff retry on rate limit/network errors."""
    last_error: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS):
        try:
            return func()
        except (RateLimitError, ConnectionError, TimeoutError) as e:
            last_error = e
            if attempt < len(_RETRY_DELAYS) - 1:
                time.sleep(delay)
    raise last_error


def _hash_value(value: Any) -> str:
    """Hash an arbitrary value with SHA-256 for output verification."""
    if isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, bytes):
        raw = value
    else:
        raw = _json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _parse_timestamp(value: Any) -> float:
    """Parse a timestamp from API response (ISO string or float)."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        from datetime import datetime

        # Parse ISO format datetime string
        try:
            # Handle both with and without timezone
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            dt = datetime.fromisoformat(value)
            return dt.timestamp()
        except ValueError:
            return 0.0
    return 0.0


try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    httpx = None  # type: ignore

# API configuration
_api_key: str | None = None
_api_base: str = os.environ.get("ASQAV_API_URL", "https://api.asqav.com/api/v1")
_client: Any = None
# Hybrid GDPR mode: resolved at init() time, overridable per-call. See _mode.py.
_mode: str = "full-payload"
_org_salt: bytes | None = None
# Metadata keys allowed to ride alongside a hash-only request. The server
# enforces its own whitelist; this mirrors the conservative default so we
# don't accidentally leak fields the customer didn't intend to share.
_HASH_ONLY_METADATA_WHITELIST: frozenset[str] = frozenset(
    {"agent_id", "session_id", "action_type", "model_name", "tool_name", "parent_id"}
)


class AsqavError(Exception):
    """Base exception for asqav errors."""

    pass


class AuthenticationError(AsqavError):
    """Raised when API key is missing or invalid."""

    pass


class RateLimitError(AsqavError):
    """Raised when rate limit is exceeded."""

    pass


class APIError(AsqavError):
    """Raised for general API errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class AgentResponse:
    """Response from agent creation."""

    agent_id: str
    name: str
    public_key: str
    key_id: str
    algorithm: str
    capabilities: list[str]
    created_at: float


@dataclass
class TokenResponse:
    """Response from token issuance."""

    token: str
    expires_at: float
    algorithm: str


def _parse_bitcoin_anchor(raw: dict[str, Any] | None) -> "BitcoinAnchor | None":
    """Turn the JSON bitcoin_anchor block from the API into a typed value."""
    if not raw:
        return None
    return BitcoinAnchor(
        status=raw.get("status") or "none",
        bitcoin_tx=raw.get("bitcoin_tx"),
        bitcoin_block=raw.get("bitcoin_block"),
    )


@dataclass
class BitcoinAnchor:
    """Bitcoin anchor state for a signed action.

    Branch on ``status`` before treating the signature as Bitcoin-anchored:
    - ``none``: no OpenTimestamps proof was attached (free tier path).
    - ``pending``: OTS stamp submitted, not yet included in a Bitcoin block.
    - ``confirmed``: OTS calendar upgraded the proof to a specific block.
    - ``failed``: OTS upgrade failed permanently; treat as unanchored.
    """

    status: str
    bitcoin_tx: str | None = None
    bitcoin_block: int | None = None


# IETF Compliance Receipts profile (draft-marques-asqav-compliance-receipts-00 §5).
# Receipt type namespace mirrored from the cloud's SignRequest validator
# (`src/asqav_cloud/api/routes/agents.py`). Kept here so the SDK can reject
# bad inputs before a network roundtrip.
RECEIPT_TYPE_NAMESPACE: frozenset[str] = frozenset(
    {
        "protectmcp:decision",
        "protectmcp:restraint",
        "protectmcp:lifecycle",
    }
)

# DORA RTS JC 2024-33 Annex II field 3.23 canonical incident classification.
# The Joint Committee of the European Supervisory Authorities published
# the final RTS on classification of major ICT-related incidents on
# 17 July 2024 (JC 2024-33). Annex II field 3.23 fixes the controlled
# vocabulary of `incident_class` to exactly six values. Earlier DORA
# preliminary drafts (and the SDK's previous releases) circulated a
# 12-value list; the cloud accepts those legacy tokens and normalises
# them to the canonical six server-side, so this SDK forwards the
# caller's value verbatim and only rejects tokens that are neither
# canonical nor a known legacy alias.
DORA_INCIDENT_CLASS_NAMESPACE: frozenset[str] = frozenset(
    {
        "cybersecurity_related",
        "process_failure",
        "system_failure",
        "external_event",
        "payment_related",
        "other",
    }
)

# Legacy 12-value vocabulary from pre-final DORA drafts. Mirrored from
# the cloud's alias dict (cloud PR #194) so the SDK can accept either
# form without a network roundtrip. The cloud performs the authoritative
# normalisation; this map is purely for client-side acceptance checks.
LEGACY_DORA_ALIASES: dict[str, str] = {
    "malicious_actions": "cybersecurity_related",
    "cyberattack": "cybersecurity_related",
    "unauthorised_access": "cybersecurity_related",
    "process_failure_internal": "process_failure",
    "human_error": "process_failure",
    "system_failure_internal": "system_failure",
    "software_malfunction": "system_failure",
    "hardware_failure": "system_failure",
    "third_party_failure": "external_event",
    "natural_disaster": "external_event",
    "payment_fraud": "payment_related",
    "unknown": "other",
}

# Verifier rejects receipts further than this from the wall clock.
SKEW_BOUND_SECONDS: int = 300

# Spec wire vocabulary {"allow", "deny", "rate_limit"} vs legacy
# `policy_decision` {"permit", "deny", "rate_limit"}; unknown -> "deny".
DECISION_MAP: dict[str, str] = {
    "permit": "allow",
    "allow": "allow",
    "deny": "deny",
    "rate_limit": "rate_limit",
}


def _map_policy_decision_to_decision(policy_decision: str | None) -> str:
    """Translate internal `policy_decision` to spec-shape `decision`.

    Used when emitting compliance receipts. Falls back to "deny" so a
    misconfigured caller never publishes an out-of-vocabulary token.
    """
    return DECISION_MAP.get((policy_decision or "").lower(), "deny")

# REQUIRED fields on a Compliance Receipt envelope per §5 of
# draft-marques-asqav-compliance-receipts-00. Used by the local
# `verify_compliance_receipt` helper.
_COMPLIANCE_REQUIRED_FIELDS: tuple[str, ...] = (
    "receipt_type",
    "issuer_id",
    "agent_id",
    "action_ref",
    "payload_digest",
    "policy_digest",
    "previousReceiptHash",
    "policy_decision",
    "issued_at",
)


@dataclass
class SignatureResponse:
    """Response from signing operation.

    The ML-DSA ``signature`` is always present. ``bitcoin_anchor`` carries
    the anchoring state separately so callers can distinguish a
    cryptographically signed action from one that is also Bitcoin-anchored.
    """

    # Polymorphic: str (b64) for legacy, {alg, kid, sig} dict under
    # compliance_mode. Use signature_envelope() for the dict form.
    signature: str | dict[str, str]
    signature_id: str
    action_id: str
    timestamp: float
    verification_url: str
    algorithm: str | None = None
    chain_hash: str | None = None
    rfc3161_tsa: str | None = None
    rfc3161_serial: str | None = None
    scan_result: dict[str, Any] | None = None
    policy_digest: str | None = None
    policy_decision: str = "permit"
    authorization_ref: str | None = None
    bitcoin_anchor: BitcoinAnchor | None = None
    # Multi-party countersigning (optional). When the caller passed
    # `co_signers=[...]` to `agent.sign()`, the server records the list and
    # surfaces it back so peers know they need to countersign.
    required_co_signers: list[str] | None = None
    co_signatures: list[dict[str, Any]] | None = None
    countersign_url: str | None = None
    # True when the server validated a user_intent envelope on this record.
    user_intent_verified: bool | None = None
    # Compliance Receipts profile fields. Populated by the cloud only
    # when `compliance_mode=True` was passed on the sign call. None on
    # legacy receipts so callers can branch cleanly.
    compliance_mode: bool = False
    receipt_type: str | None = None
    action_ref: str | None = None
    # payload_digest accepts both shapes for back-compat: legacy
    # "sha256:<hex>" string or the wire object form {"hash", "size"}.
    payload_digest: str | dict[str, Any] | None = None
    issuer_id: str | None = None
    iteration_id: str | None = None
    sandbox_state: str | None = None
    risk_class: str | None = None
    # str or list of strings for multi-regime cases.
    incident_class: str | list[str] | None = None
    reason: str | None = None
    previous_receipt_hash: str | None = None
    # Spec `decision` token (allow | deny | rate_limit). None on
    # non-compliance receipts.
    decision: str | None = None
    # Three-key Compliance Receipts envelope. `payload` is the canonical
    # signed dict; `anchors` is the type-discriminated array. None on
    # legacy receipts.
    payload: dict[str, Any] | None = None
    anchors: list[dict[str, Any]] | None = None

    def signature_envelope(self) -> dict[str, str] | None:
        """Return the spec-shape signature envelope `{alg, kid, sig}`.

        Returns the cloud-supplied object form when present (compliance
        mode); None otherwise.
        """
        if isinstance(self.signature, dict):
            keys = set(self.signature.keys())
            if keys >= {"alg", "kid", "sig"}:
                return self.signature  # type: ignore[return-value]
        return None

    def anchors_array(self) -> list[dict[str, Any]] | None:
        """Return the cloud-supplied `anchors[]` array. None on
        non-compliance receipts; [] under compliance_mode before
        anchors land.
        """
        return self.anchors


@dataclass
class SessionResponse:
    """Response from session operations."""

    session_id: str
    agent_id: str
    status: str
    started_at: str  # ISO datetime string
    ended_at: str | None = None


@dataclass
class SDTokenResponse:
    """Response from SD-JWT token issuance (Enterprise tier).

    SD-JWT tokens allow selective disclosure of claims to external services.
    Use present() to create a proof with only specific claims revealed.
    """

    token: str  # Full SD-JWT with all disclosures
    jwt: str  # Just the signed JWT part
    disclosures: dict[str, str]  # claim_name -> encoded disclosure
    expires_at: float

    def present(self, disclose: list[str]) -> str:
        """Create a presentation with only specified claims disclosed.

        Args:
            disclose: List of claim names to reveal.

        Returns:
            SD-JWT string with only specified disclosures.

        Example:
            # Full token has: tier, org, capabilities
            # Present only tier to partner:
            proof = sd_token.present(["tier"])
        """
        parts = [self.jwt]
        for claim_name in disclose:
            if claim_name in self.disclosures:
                parts.append(self.disclosures[claim_name])
        return "~".join(parts) + "~"

    def full(self) -> str:
        """Return full SD-JWT with all disclosures."""
        return self.token


@dataclass
class CertificateResponse:
    """Agent identity certificate."""

    agent_id: str
    agent_name: str
    algorithm: str
    public_key_pem: str
    key_id: str
    created_at: float
    is_revoked: bool


@dataclass
class VerificationDetail:
    """Granular per-axis verification outcome (May 2026 release).

    Optional on the response; older servers do not return it.
    `validation_label` is one of: valid, invalid_signature,
    signature_expired, signer_key_changed, algorithm_mismatch,
    agent_inactive, agent_unknown, chain_break, anchor_invalid,
    missing_required_field, signed_at_skew.

    The IETF Compliance Receipts profile fields (`signed_at_skew_seconds`,
    `chain_valid`, `anchor_status_ots`, `anchor_status_rfc3161`,
    `missing_fields`) are populated only when the cloud returns them on
    a compliance-mode receipt; older receipts and older deployments
    leave them None for back-compat.
    """

    signer_key_match: bool
    signature_valid: bool
    algorithm_match: bool
    agent_active: bool
    validation_label: str
    # IETF profile sub-axes (cloud may omit on legacy receipts).
    signed_at_skew_seconds: float | None = None
    chain_valid: bool | None = None
    anchor_status_ots: str | None = None
    anchor_status_rfc3161: str | None = None
    missing_fields: list[str] | None = None


@dataclass
class VerificationResponse:
    """Public verification response."""

    signature_id: str
    agent_id: str
    agent_name: str
    action_id: str
    action_type: str
    payload: dict[str, Any] | None
    signature: str
    algorithm: str
    signed_at: float
    verified: bool
    verification_url: str
    verification_detail: VerificationDetail | None = None


@dataclass
class SignedActionResponse:
    """Signed action from a session."""

    signature_id: str
    agent_id: str
    action_id: str
    action_type: str
    payload: dict[str, Any] | None
    algorithm: str
    signed_at: float
    signature_preview: str
    verification_url: str


@dataclass
class SignatureDetail:
    """Details of a single entity signature within a signing session."""

    entity_id: str
    entity_name: str
    entity_class: str
    signed_at: str


@dataclass
class SigningSessionResponse:
    """Response from signing session operations."""

    session_id: str
    config_id: str
    agent_id: str
    action_type: str
    action_params: dict[str, Any] | None
    approvals_required: int
    signatures_collected: int
    status: str
    signatures: list[SignatureDetail]
    policy_attestation_hash: str | None
    created_at: str
    expires_at: str
    resolved_at: str | None = None


@dataclass
class ApprovalResponse:
    """Response from signing approval operation."""

    session_id: str
    entity_id: str
    signatures_collected: int
    approvals_required: int
    status: str
    approved: bool


@dataclass
class SigningGroupResponse:
    """Response from signing group operations."""

    id: str
    agent_id: str
    min_approvals: int
    total_shares: int
    is_active: bool
    created_at: str
    updated_at: str | None = None


@dataclass
class SigningEntityResponse:
    """Response from signing entity operations."""

    id: str
    config_id: str
    entity_class: str
    label: str
    is_active: bool
    created_at: str


@dataclass
class GroupKeypairResponse:
    """Response from group keypair operations."""

    id: str
    config_id: str
    public_key_hex: str
    min_approvals: int
    total_shares: int
    created_at: str
    status: str = "active"


@dataclass
class GroupSignResponse:
    """Response from group signing operation."""

    signature_hex: str
    message_hex: str
    keypair_id: str
    verified: bool
    audit_record_id: str | None = None


@dataclass
class RiskRuleResponse:
    """Response from risk rule operations."""

    id: str
    name: str
    action_pattern: str
    risk_level: str
    approval_override: int | None
    priority: int
    entity_weights: dict[str, float] | None = None
    time_schedule: dict[str, Any] | None = None
    created_at: str = ""


@dataclass
class DelegationResponse:
    """Response from delegation operations."""

    id: str
    config_id: str
    delegator_entity_id: str
    delegate_entity_id: str
    expires_at: str
    is_active: bool
    created_at: str


@dataclass
class KeyRefreshResponse:
    """Response from key share refresh operation."""

    keypair_id: str
    refreshed_at: str
    delegations_invalidated: int


@dataclass
class ShareRecoveryResponse:
    """Response from key share recovery operation."""

    keypair_id: str
    recovered_entity_id: str
    recovered_at: str


@dataclass
class PreflightResult:
    """Result of a pre-flight check combining agent status and policy checks.

    If cleared is True the agent is allowed to proceed with the action.
    When cleared is False, reasons lists the blocking factors.
    The explanation field provides a human-readable summary of the result.
    """

    cleared: bool
    agent_active: bool
    policy_allowed: bool
    reasons: list[str]
    explanation: str = ""


@dataclass
class Span:
    """A single traced operation.

    Spans track the duration and context of operations. When ended,
    they are signed server-side with ML-DSA for cryptographic proof.
    """

    span_id: str
    name: str
    start_time: float
    attributes: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    end_time: float | None = None
    status: str = "ok"
    signature: str | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Add an attribute to the span."""
        self.attributes[key] = value

    def set_status(self, status: str) -> None:
        """Set span status (ok, error)."""
        self.status = status


# Global tracer state
_current_span: Span | None = None
_span_stack: list[Span] = []
_completed_spans: list[Span] = []
_otel_endpoint: str | None = None


@contextmanager
def span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Span, None, None]:
    """Create a traced span with automatic signing.

    Example:
        with asqav.span("api:openai", {"model": "gpt-4"}) as s:
            response = openai.chat.completions.create(...)
            s.set_attribute("tokens", response.usage.total_tokens)
    """
    global _current_span, _span_stack

    span_obj = Span(
        span_id=str(uuid.uuid4()),
        name=name,
        start_time=time.time(),
        attributes=attributes or {},
        parent_id=_current_span.span_id if _current_span else None,
    )

    _span_stack.append(span_obj)
    _current_span = span_obj

    try:
        yield span_obj
        span_obj.status = "ok"
    except Exception as e:
        span_obj.status = "error"
        span_obj.set_attribute("error.message", str(e))
        span_obj.set_attribute("error.type", type(e).__name__)
        raise
    finally:
        span_obj.end_time = time.time()
        _span_stack.pop()
        _current_span = _span_stack[-1] if _span_stack else None

        # Sign the span via API
        try:
            agent = get_agent()
            result = agent.sign(
                action_type=f"span:{name}",
                context={
                    "span_id": span_obj.span_id,
                    "parent_id": span_obj.parent_id,
                    "start_time": span_obj.start_time,
                    "end_time": span_obj.end_time,
                    "duration_ms": (span_obj.end_time - span_obj.start_time) * 1000,
                    "status": span_obj.status,
                    "attributes": span_obj.attributes,
                },
            )
            span_obj.signature = result.signature
        except Exception:
            pass  # Don't fail user code if signing fails

        # Add to completed spans for OTEL export
        _completed_spans.append(span_obj)


def get_current_span() -> Span | None:
    """Get the currently active span, if any."""
    return _current_span


def configure_otel(endpoint: str | None = None) -> None:
    """Configure OpenTelemetry export.

    Args:
        endpoint: OTEL collector endpoint (e.g., "http://localhost:4318/v1/traces").
                  Set to None to disable export.

    Example:
        asqav.configure_otel("http://localhost:4318/v1/traces")
    """
    global _otel_endpoint
    _otel_endpoint = endpoint


def span_to_otel(s: Span) -> dict[str, Any]:
    """Convert a Span to OTEL format."""
    return {
        "traceId": s.span_id.replace("-", "")[:32].ljust(32, "0"),
        "spanId": s.span_id.replace("-", "")[:16],
        "parentSpanId": s.parent_id.replace("-", "")[:16] if s.parent_id else None,
        "name": s.name,
        "kind": 1,  # INTERNAL
        "startTimeUnixNano": int(s.start_time * 1_000_000_000),
        "endTimeUnixNano": int((s.end_time or s.start_time) * 1_000_000_000),
        "attributes": [
            {"key": k, "value": {"stringValue": str(v)}} for k, v in s.attributes.items()
        ]
        + (
            [{"key": "asqav.signature", "value": {"stringValue": s.signature}}]
            if s.signature
            else []
        ),
        "status": {"code": 1 if s.status == "ok" else 2},
    }


def export_spans() -> list[dict[str, Any]]:
    """Export completed spans in OTEL format.

    Returns:
        List of spans in OTEL format.
    """
    global _completed_spans
    spans = [span_to_otel(s) for s in _completed_spans]
    _completed_spans = []
    return spans


def flush_spans() -> None:
    """Flush spans to configured OTEL endpoint."""
    global _otel_endpoint, _completed_spans

    if not _otel_endpoint or not _completed_spans:
        return

    from . import __version__

    spans = export_spans()
    payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "asqav"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "asqav", "version": __version__},
                        "spans": spans,
                    }
                ],
            }
        ]
    }

    try:
        import json
        import urllib.request

        req = urllib.request.Request(
            _otel_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # Don't fail on export errors


def _compute_action_ref(
    action_type: str, context: dict[str, Any] | None
) -> str:
    """Client-side ``sha256:<hex>`` of the canonical Action object.

    The Action shape mirrors `core/canonical.py:hash_action` on the cloud
    so the SDK and cloud agree byte-for-byte under JCS encoding.
    """
    from .canonicalize import canonicalize_action

    return "sha256:" + hashlib.sha256(canonicalize_action(action_type, context)).hexdigest()


def _build_sign_body(
    *,
    action_type: str,
    context: dict[str, Any] | None,
    session_id: str | None,
    agent_id: str,
    user_intent: dict[str, Any] | None = None,
    compliance_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON body for ``POST /agents/{id}/sign``.

    full-payload sends ``context``; hash-only sends a digest plus a
    whitelisted metadata bag.
    """
    if _mode == "hash-only":
        from .canonicalize import canonicalize_action

        canonical_bytes = canonicalize_action(action_type, context)
        payload_size = len(canonical_bytes)
        if _org_salt is not None:
            digest_hex = hmac.new(
                _org_salt, canonical_bytes, hashlib.sha256
            ).hexdigest()
        else:
            digest_hex = hashlib.sha256(canonical_bytes).hexdigest()
        digest = f"sha256:{digest_hex}"
        metadata: dict[str, Any] = {
            "agent_id": agent_id,
            "action_type": action_type,
        }
        if session_id is not None:
            metadata["session_id"] = session_id
        # Opt-in via underscored sentinel keys; nothing else from context travels.
        for src_key, dst_key in (
            ("_model_name", "model_name"),
            ("_tool_name", "tool_name"),
            ("_parent_id", "parent_id"),
        ):
            if context and src_key in context:
                value = context[src_key]
                if isinstance(value, str):
                    metadata[dst_key] = value
        metadata = {k: v for k, v in metadata.items() if k in _HASH_ONLY_METADATA_WHITELIST}
        body: dict[str, Any] = {
            "action_type": action_type,
            "hash": digest,
            "hash_algo": "sha256",
            "payload_size": payload_size,
            "metadata": metadata,
            "session_id": session_id,
        }
        if user_intent is not None:
            body["user_intent"] = user_intent
        if compliance_fields:
            for key, value in compliance_fields.items():
                if value is not None:
                    body[key] = value
        return body

    # full-payload (default for self-hosted)
    body = {
        "action_type": action_type,
        "context": context or {},
        "session_id": session_id,
    }
    if user_intent is not None:
        body["user_intent"] = user_intent
    if compliance_fields:
        for key, value in compliance_fields.items():
            if value is not None:
                body[key] = value
    return body


@dataclass
class Agent:
    """Agent representation from asqav Cloud.

    All ML-DSA cryptography happens server-side.
    This is a thin client that wraps API calls.
    """

    agent_id: str
    name: str
    public_key: str
    key_id: str
    algorithm: str
    capabilities: list[str]
    created_at: float
    _session_id: str | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        name: str,
        algorithm: str = "ml-dsa-65",
        capabilities: list[str] | None = None,
    ) -> Agent:
        """Create a new agent via asqav Cloud.

        The server generates the keypair. The private key never leaves
        the server.

        Args:
            name: Human-readable name for the agent.
            algorithm: Receipt-signing algorithm. One of `ml-dsa-65`
                (FIPS 204; default), `ed25519`, or `es256`
                (ECDSA-P256 per RFC 7518). The cloud also accepts
                `ml-dsa-44` and `ml-dsa-87`.
            capabilities: List of capabilities/permissions.

        Returns:
            An Agent instance.

        Raises:
            AuthenticationError: If API key is missing or invalid.
            APIError: If the request fails.
        """
        data = _post(
            "/agents/create",
            {
                "name": name,
                "algorithm": algorithm,
                "capabilities": capabilities or [],
            },
        )

        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            public_key=data["public_key"],
            key_id=data["key_id"],
            algorithm=data["algorithm"],
            capabilities=data["capabilities"],
            created_at=_parse_timestamp(data["created_at"]),
        )

    @classmethod
    def get(cls, agent_id: str) -> Agent:
        """Get an existing agent by ID.

        Args:
            agent_id: The agent ID to retrieve.

        Returns:
            An Agent instance.
        """
        data = _get(f"/agents/{agent_id}")

        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            public_key=data["public_key"],
            key_id=data["key_id"],
            algorithm=data["algorithm"],
            capabilities=data["capabilities"],
            created_at=_parse_timestamp(data["created_at"]),
        )

    def issue_token(
        self,
        scope: list[str] | None = None,
        ttl: int = 3600,
    ) -> TokenResponse:
        """Issue a PQC-JWT token for this agent.

        The token is signed server-side with ML-DSA.

        Args:
            scope: Capabilities to include (default: all).
            ttl: Token time-to-live in seconds.

        Returns:
            TokenResponse with the signed token.
        """
        data = _post(
            f"/agents/{self.agent_id}/tokens",
            {
                "scope": scope or self.capabilities,
                "ttl": ttl,
            },
        )

        return TokenResponse(
            token=data["token"],
            expires_at=_parse_timestamp(data["expires_at"]),
            algorithm=data["algorithm"],
        )

    def issue_sd_token(
        self,
        claims: dict[str, Any] | None = None,
        disclosable: list[str] | None = None,
        ttl: int = 3600,
    ) -> SDTokenResponse:
        """Issue a PQC-SD-JWT token with selective disclosure (Enterprise tier).

        SD-JWT tokens allow agents to selectively reveal claims when
        presenting the token to external services, maintaining privacy.

        Args:
            claims: Claims to include in the token.
            disclosable: List of claim names that can be selectively disclosed.
            ttl: Token time-to-live in seconds.

        Returns:
            SDTokenResponse with the token and disclosures.

        Example:
            sd_token = agent.issue_sd_token(
                claims={"tier": "pro", "org": "acme"},
                disclosable=["tier", "org"]
            )

            # Present to partner - only show tier
            proof = sd_token.present(["tier"])
        """
        data = _post(
            f"/agents/{self.agent_id}/tokens/sd",
            {
                "claims": claims or {},
                "disclosable": disclosable or [],
                "ttl": ttl,
            },
        )

        return SDTokenResponse(
            token=data["token"],
            jwt=data["jwt"],
            disclosures=data["disclosures"],
            expires_at=_parse_timestamp(data["expires_at"]),
        )

    def sign(
        self,
        action_type: str,
        context: dict[str, Any] | None = None,
        system_prompt_hash: str | None = None,
        model_params: dict | None = None,
        tool_inputs_hash: str | None = None,
        trace_id: str | None = None,
        parent_id: str | None = None,
        counterparty: dict[str, Any] | None = None,
        tool_name: str | None = None,
        model_name: str | None = None,
        *,
        co_signers: list[str] | None = None,
        user_intent: dict[str, Any] | None = None,
        nonce: str | None = None,
        valid_seconds: int | None = None,
        expected_executor_pubkey_b64: str | None = None,
        compliance_mode: bool = False,
        receipt_type: str | None = None,
        action_ref: str | None = None,
        payload_digest: str | dict[str, Any] | None = None,
        issuer_id: str | None = None,
        iteration_id: str | None = None,
        sandbox_state: str | None = None,
        risk_class: str | None = None,
        incident_class: str | list[str] | None = None,
        reason: str | None = None,
        policy_decision: str = "permit",
    ) -> SignatureResponse:
        """Sign an action cryptographically.

        The signature is created server-side with ML-DSA.

        Args:
            action_type: Type of action (e.g., "read:data", "api:call").
                Accepts semantic pattern names like "sql-read" which
                auto-expand to their glob equivalents.
            context: Additional context for the action.
            system_prompt_hash: Optional hash of the system prompt for
                reproducibility tracking.
            model_params: Optional dict of model parameters (temperature,
                top_p, etc.) for reproducibility tracking.
            tool_inputs_hash: Optional hash of tool inputs for
                reproducibility tracking.
            trace_id: Optional trace ID for correlating actions across
                a multi-step workflow. Use generate_trace_id() to create one.
            parent_id: Optional parent action ID for linking child actions
                to their parent in a workflow. Powers the agent graph view.
            counterparty: Optional identity of the endpoint the agent is
                calling. Included in the signed record so an auditor can
                verify the agent reached a specific endpoint, not an
                imposter. Recommended shape::

                    {"kind": "tls_spki_sha256", "value": "<hex>"}
                    {"kind": "pubkey_fingerprint", "value": "<hex>"}
                    {"kind": "url_sni", "value": "api.example.com"}

                Any dict is accepted; unknown kinds are recorded verbatim.
            tool_name: Optional name of the tool being invoked when this
                action is a tool call. Surfaced in the dashboard graph.
                Only the name string travels; tool arguments do not.
            model_name: Optional name of the model being called when this
                action is an LLM call (e.g. "gpt-4", "claude-opus-4-7").
                Only the name string travels; the prompt does not.

        Returns:
            SignatureResponse with the signature.
        """
        action_type = resolve_pattern(action_type)

        if (
            system_prompt_hash
            or model_params
            or tool_inputs_hash
            or trace_id
            or parent_id
            or counterparty
            or tool_name
            or model_name
        ):
            context = dict(context) if context else {}
            if system_prompt_hash:
                context["_system_prompt_hash"] = system_prompt_hash
            if model_params:
                context["_model_params"] = model_params
            if tool_inputs_hash:
                context["_tool_inputs_hash"] = tool_inputs_hash
            if trace_id:
                context["_trace_id"] = trace_id
            if parent_id:
                context["_parent_id"] = parent_id
            if counterparty:
                context["_counterparty"] = counterparty
            if tool_name:
                context["_tool_name"] = tool_name
            if model_name:
                context["_model_name"] = model_name

        # Dispatch before-hooks (fail-open).
        try:
            from . import hooks as _hooks_mod
            context = _hooks_mod._dispatch_before(
                action_type, dict(context) if context else {}
            )
        except Exception:
            logger.warning("asqav before-hook dispatch failed (fail-open)", exc_info=True)

        # Surface bad arguments client-side; the cloud re-validates.
        if receipt_type is not None and receipt_type not in RECEIPT_TYPE_NAMESPACE:
            raise ValueError(
                "invalid_receipt_type: must be one of "
                f"{sorted(RECEIPT_TYPE_NAMESPACE)}"
            )
        if policy_decision in {"deny", "rate_limit"} and not reason:
            raise ValueError(
                "missing_reason: policy_decision=deny|rate_limit "
                "requires a `reason` code."
            )
        # DORA RTS JC 2024-33 Annex II field 3.23 vocabulary check (six
        # canonical values, plus the legacy 12-value alias set the cloud
        # still accepts). Reject anything else before the HTTP roundtrip.
        if incident_class is not None and (
            incident_class not in DORA_INCIDENT_CLASS_NAMESPACE
            and incident_class not in LEGACY_DORA_ALIASES
        ):
            raise ValueError(
                "invalid_incident_class: must be one of "
                f"{sorted(DORA_INCIDENT_CLASS_NAMESPACE)} "
                "(per DORA RTS JC 2024-33 Annex II field 3.23) "
                "or a known legacy alias "
                f"{sorted(LEGACY_DORA_ALIASES)}."
            )
        # Derive action_ref under compliance_mode when caller omits it.
        if compliance_mode and action_ref is None:
            action_ref = _compute_action_ref(action_type, context)

        compliance_fields: dict[str, Any] = {}
        if compliance_mode:
            compliance_fields["compliance_mode"] = True
            compliance_fields["receipt_type"] = (
                receipt_type or "protectmcp:decision"
            )
            compliance_fields["policy_decision"] = policy_decision
            compliance_fields["action_ref"] = action_ref
            for k, v in (
                ("payload_digest", payload_digest),
                ("issuer_id", issuer_id),
                ("iteration_id", iteration_id),
                ("sandbox_state", sandbox_state),
                ("risk_class", risk_class),
                ("incident_class", incident_class),
                ("reason", reason),
            ):
                if v is not None:
                    compliance_fields[k] = v

        body = _build_sign_body(
            action_type=action_type,
            context=context,
            session_id=self._session_id,
            agent_id=self.agent_id,
            user_intent=user_intent,
            compliance_fields=compliance_fields or None,
        )
        if co_signers:
            body["co_signers"] = list(co_signers)
        # Replay protection.
        if nonce is not None:
            body["nonce"] = nonce
        if valid_seconds is not None:
            body["valid_seconds"] = valid_seconds
        # Counterparty pin: server decodes b64.
        if expected_executor_pubkey_b64 is not None:
            body["expected_executor_pubkey_b64"] = expected_executor_pubkey_b64
        data = _post(f"/agents/{self.agent_id}/sign", body)

        response = SignatureResponse(
            signature=data["signature"],
            signature_id=data["signature_id"],
            action_id=data["action_id"],
            timestamp=data["timestamp"],
            verification_url=data["verification_url"],
            algorithm=data.get("algorithm"),
            chain_hash=data.get("chain_hash") or data.get("record_hash"),
            rfc3161_tsa=(data.get("rfc3161_timestamp") or {}).get("tsa"),
            rfc3161_serial=(data.get("rfc3161_timestamp") or {}).get("serial_number"),
            scan_result=data.get("scan_result"),
            policy_digest=data.get("policy_digest"),
            policy_decision=data.get("policy_decision", "permit"),
            authorization_ref=data.get("authorization_ref"),
            bitcoin_anchor=_parse_bitcoin_anchor(data.get("bitcoin_anchor")),
            required_co_signers=data.get("required_co_signers"),
            co_signatures=data.get("co_signatures"),
            countersign_url=data.get("countersign_url"),
            user_intent_verified=data.get("user_intent_verified"),
            # IETF profile fields. Cloud emits both `previousReceiptHash`
            # (camelCase per spec) and `previous_receipt_hash` (snake_case
            # alias). Accept either so the SDK works against both shapes.
            compliance_mode=bool(data.get("compliance_mode", False)),
            receipt_type=data.get("receipt_type"),
            action_ref=data.get("action_ref"),
            payload_digest=data.get("payload_digest"),
            issuer_id=data.get("issuer_id"),
            iteration_id=data.get("iteration_id"),
            sandbox_state=data.get("sandbox_state"),
            risk_class=data.get("risk_class"),
            incident_class=data.get("incident_class"),
            reason=data.get("reason"),
            previous_receipt_hash=(
                data.get("previousReceiptHash")
                or data.get("previous_receipt_hash")
            ),
            # Spec-shape `decision`; fall back to local translation for
            # older clouds that only emit `policy_decision`.
            decision=(
                data.get("decision")
                or (
                    _map_policy_decision_to_decision(
                        data.get("policy_decision")
                    )
                    if data.get("compliance_mode")
                    else None
                )
            ),
            payload=data.get("payload"),
            anchors=data.get("anchors"),
        )

        # Dispatch after-hooks (fail-open).
        try:
            from . import hooks as _hooks_mod
            _hooks_mod._dispatch_after(action_type, response)
        except Exception:
            logger.warning("asqav after-hook dispatch failed (fail-open)", exc_info=True)

        return response

    def countersign(self, signature_id: str) -> SignatureResponse:
        """Countersign an existing signature record.

        The server looks up ``signature_id``, verifies that this agent is
        in the original ``required_co_signers`` list and shares the same
        organization, then signs the SAME canonical bytes the original
        signer signed. No new prompts or tool args travel over the wire;
        only signatures are added.

        Args:
            signature_id: The ``sig_...`` id returned by the original
                ``sign(co_signers=[...])`` call.

        Returns:
            SignatureResponse describing the updated record, including
            the full ``co_signatures`` list after this agent's signature
            was appended.
        """
        data = _post(
            f"/agents/{self.agent_id}/countersign/{signature_id}",
            {},
        )
        return SignatureResponse(
            signature=data["signature"],
            signature_id=data["signature_id"],
            action_id=data["action_id"],
            timestamp=data["timestamp"],
            verification_url=data["verification_url"],
            algorithm=data.get("algorithm"),
            chain_hash=data.get("chain_hash") or data.get("record_hash"),
            rfc3161_tsa=(data.get("rfc3161_timestamp") or {}).get("tsa"),
            rfc3161_serial=(data.get("rfc3161_timestamp") or {}).get("serial_number"),
            policy_digest=data.get("policy_digest"),
            policy_decision=data.get("policy_decision", "permit"),
            authorization_ref=data.get("authorization_ref"),
            bitcoin_anchor=_parse_bitcoin_anchor(data.get("bitcoin_anchor")),
            required_co_signers=data.get("required_co_signers"),
            co_signatures=data.get("co_signatures"),
            countersign_url=data.get("countersign_url"),
            user_intent_verified=data.get("user_intent_verified"),
            decision=(
                data.get("decision")
                or (
                    _map_policy_decision_to_decision(
                        data.get("policy_decision")
                    )
                    if data.get("compliance_mode")
                    else None
                )
            ),
        )

    def sign_batch(
        self,
        actions: list[dict[str, Any]],
    ) -> list[SignatureResponse | None]:
        """Sign multiple actions in one HTTP roundtrip.

        Each item in ``actions`` is a dict with ``action_type`` (required) and
        optional ``context``. The server signs each action through the full
        single-sign pipeline. One bad action does not abort the batch.

        Returns a list parallel to ``actions``. Each element is either a
        SignatureResponse on success or None on failure. To inspect failures,
        compare positions to the original input.

        Args:
            actions: List of {"action_type": str, "context": dict | None}.
                    Max 100 per call.

        Returns:
            List of SignatureResponse or None, same length as ``actions``.
        """
        if not actions:
            raise ValueError("actions must contain at least one item")
        if len(actions) > 100:
            raise ValueError("actions must contain at most 100 items per call")

        body_actions: list[dict[str, Any]] = []
        for a in actions:
            if "action_type" not in a:
                raise ValueError("each action must include 'action_type'")
            body_actions.append(
                {
                    "action_type": resolve_pattern(a["action_type"]),
                    "context": a.get("context") or {},
                    "session_id": self._session_id,
                }
            )

        data = _post(
            f"/agents/{self.agent_id}/sign-batch",
            {"actions": body_actions},
        )

        results: list[SignatureResponse | None] = []
        for sig in data.get("signatures", []):
            if sig is None:
                results.append(None)
            else:
                results.append(
                    SignatureResponse(
                        signature=sig["signature"],
                        signature_id=sig["signature_id"],
                        action_id=sig["action_id"],
                        timestamp=sig["timestamp"],
                        verification_url=sig["verification_url"],
                        algorithm=sig.get("algorithm"),
                        chain_hash=sig.get("chain_hash") or sig.get("record_hash"),
                        rfc3161_tsa=(sig.get("rfc3161_timestamp") or {}).get("tsa"),
                        rfc3161_serial=(sig.get("rfc3161_timestamp") or {}).get("serial_number"),
                        scan_result=sig.get("scan_result"),
                        policy_digest=sig.get("policy_digest"),
                        policy_decision=sig.get("policy_decision", "permit"),
                        authorization_ref=sig.get("authorization_ref"),
                        bitcoin_anchor=_parse_bitcoin_anchor(sig.get("bitcoin_anchor")),
                        decision=(
                            sig.get("decision")
                            or (
                                _map_policy_decision_to_decision(
                                    sig.get("policy_decision")
                                )
                                if sig.get("compliance_mode")
                                else None
                            )
                        ),
                    )
                )
        return results

    def sign_output(
        self,
        action_type: str,
        input_hash: str,
        output: Any,
        context: dict[str, Any] | None = None,
    ) -> SignatureResponse:
        """Sign a tool output with a hash of the original input for binding.

        Creates a signed record that binds the output to a specific input,
        proving the output hasn't been modified after generation.

        Args:
            action_type: Type of action (e.g., "tool:search", "api:call").
            input_hash: SHA-256 hash of the original input for binding.
            output: The output to hash and sign.
            context: Additional context to include in the signed payload.

        Returns:
            SignatureResponse with the signature.
        """
        try:
            output_hash = _hash_value(output)
        except Exception:
            import warnings
            warnings.warn("asqav: failed to hash output, signing without output_hash")
            output_hash = ""

        sign_context: dict[str, Any] = {
            "output_hash": output_hash,
            "input_hash": input_hash,
        }
        if context:
            sign_context.update(context)

        return self.sign(
            action_type=action_type,
            context=sign_context,
        )

    def start_session(self) -> SessionResponse:
        """Start a new session.

        Returns:
            SessionResponse with session details.
        """
        data = _post("/sessions/", {"agent_id": self.agent_id})

        self._session_id = data["session_id"]

        return SessionResponse(
            session_id=data["session_id"],
            agent_id=data["agent_id"],
            status=data["status"],
            started_at=data["started_at"],
        )

    def end_session(self, status: str = "completed") -> SessionResponse:
        """End the current session.

        Args:
            status: Final status (completed, error, timeout).

        Returns:
            SessionResponse with final details.
        """
        if not self._session_id:
            raise AsqavError("No active session")

        data = _patch(
            f"/sessions/{self._session_id}",
            {"status": status},
        )

        session_id = self._session_id
        self._session_id = None

        return SessionResponse(
            session_id=session_id,
            agent_id=data["agent_id"],
            status=data["status"],
            started_at=data["started_at"],
            ended_at=data.get("ended_at"),
        )

    def get_session_signatures(self) -> list[SignedActionResponse]:
        """Get signed actions for the current session.

        Returns:
            List of SignedActionResponse with signature details.
        """
        if not self._session_id:
            raise AsqavError("No active session")

        return get_session_signatures(self._session_id)

    def revoke(self, reason: str = "manual") -> None:
        """Revoke this agent's credentials permanently.

        Revocation propagates globally across the asqav network.
        Use suspend() for temporary disable that can be reversed.

        Args:
            reason: Reason for revocation.
        """
        _post(
            f"/agents/{self.agent_id}/revoke",
            {"reason": reason},
        )

    def suspend(
        self,
        reason: str = "manual",
        note: str | None = None,
    ) -> dict[str, Any]:
        """Temporarily suspend this agent.

        Suspended agents cannot sign, issue tokens, or delegate.
        Use unsuspend() to restore access. For permanent revocation,
        use revoke() instead.

        Args:
            reason: Reason for suspension (investigation, maintenance,
                    policy_violation, manual, anomaly_detected).
            note: Optional note about the suspension.

        Returns:
            Dict with suspension details.
        """
        payload: dict[str, Any] = {"reason": reason}
        if note:
            payload["note"] = note
        return _post(f"/agents/{self.agent_id}/suspend", payload)

    def unsuspend(self) -> dict[str, Any]:
        """Remove suspension from this agent."""
        return _post(f"/agents/{self.agent_id}/unsuspend", {})

    def decommission(self, reason: str = "manual") -> dict[str, Any]:
        """Permanently retire this agent (keys archived, no further activity)."""
        return _post(
            f"/agents/{self.agent_id}/decommission",
            {"reason": reason},
        )

    def quarantine(self) -> dict[str, Any]:
        """Block signing and token issuance, keep read + verify. Business+."""
        return _post(f"/agents/{self.agent_id}/quarantine", {})

    def unquarantine(self) -> dict[str, Any]:
        """Restore normal signing and token issuance after quarantine."""
        return _post(f"/agents/{self.agent_id}/unquarantine", {})

    def delegate(
        self,
        name: str,
        scope: list[str] | None = None,
        ttl: int = 86400,
    ) -> "Agent":
        """Create a delegated child agent with limited scope.

        Args:
            name: Name for the child agent.
            scope: List of capabilities to delegate.
            ttl: Time-to-live in seconds (default 24h, max 7 days).

        Returns:
            Agent: The delegated child agent.
        """
        data = _post(
            f"/agents/{self.agent_id}/delegate",
            {
                "name": name,
                "scope": scope or [],
                "ttl": ttl,
            },
        )

        return Agent(
            agent_id=data["child_id"],
            name=data["child_name"],
            public_key="",  # Use Agent.get() to fetch full details
            key_id="",
            algorithm=self.algorithm,
            capabilities=data["scope"],
            created_at=_parse_timestamp(data["created_at"]),
        )

    @property
    def is_revoked(self) -> bool:
        """Check if this agent is revoked."""
        data = _get(f"/agents/{self.agent_id}/status")
        return bool(data.get("revoked", False))

    @property
    def is_suspended(self) -> bool:
        """Check if this agent is suspended."""
        data = _get(f"/agents/{self.agent_id}/status")
        return bool(data.get("suspended", False))

    def preflight(self, action_type: str) -> PreflightResult:
        """Pre-flight check combining revocation, suspension, and policy status.

        Returns a single result indicating whether the agent is cleared to
        perform the given action type. Combines:
        - Agent status (not revoked, not suspended)
        - Policy check (action allowed by org policies)

        Use before executing sensitive actions to catch issues early.

        Fail-open: if any individual check errors out, the agent is not
        blocked. A warning is added to reasons instead.

        Args:
            action_type: The action to check (e.g., "data:read", "api:call").
                Accepts semantic pattern names like "sql-read" which
                auto-expand to their glob equivalents.

        Returns:
            PreflightResult with combined check outcome.
        """
        action_type = resolve_pattern(action_type)
        agent_active = True
        policy_allowed = True
        reasons: list[str] = []

        # Check agent status (revoked / suspended).
        try:
            status_data = _get(f"/agents/{self.agent_id}/status")
            if status_data.get("revoked", False):
                agent_active = False
                reasons.append("agent is revoked")
            if status_data.get("suspended", False):
                agent_active = False
                reasons.append("agent is suspended")
        except Exception as exc:
            reasons.append(f"status check failed ({exc}) - skipped")

        # Check organization policies.
        try:
            policies = _get("/policies")
            for p in policies if isinstance(policies, list) else []:
                if not p.get("is_active"):
                    continue
                pattern = p.get("action_pattern", "")
                if pattern == "*" or action_type.startswith(pattern.rstrip("*")):
                    if p.get("action") in ("block", "block_and_alert"):
                        policy_allowed = False
                        reasons.append(f"blocked by policy: {p.get('name', 'unknown')}")
        except Exception as exc:
            reasons.append(f"policy check failed ({exc}) - skipped")

        cleared = agent_active and policy_allowed

        # Build human-readable explanation from check results.
        if cleared:
            explanation = "Allowed: agent is active and action is permitted by policy"
        elif not agent_active and "agent is revoked" in reasons:
            explanation = "Blocked: agent has been revoked"
        elif not agent_active and "agent is suspended" in reasons:
            suspend_reason = ""
            for r in reasons:
                if r.startswith("agent is suspended"):
                    suspend_reason = r
                    break
            explanation = f"Blocked: agent is suspended ({suspend_reason})"
        elif not policy_allowed:
            explanation = "Blocked: action not permitted by current policy rules"
        else:
            explanation = "Blocked: " + "; ".join(reasons) if reasons else "Blocked"

        return PreflightResult(
            cleared=cleared,
            agent_active=agent_active,
            policy_allowed=policy_allowed,
            reasons=reasons,
            explanation=explanation,
        )

    def get_certificate(self) -> CertificateResponse:
        """Get the agent's identity certificate.

        The certificate contains the ML-DSA public key in PEM format
        for independent verification.

        Returns:
            CertificateResponse with certificate details.
        """
        data = _get(f"/agents/{self.agent_id}/certificate")

        return CertificateResponse(
            agent_id=data["agent_id"],
            agent_name=data["agent_name"],
            algorithm=data["algorithm"],
            public_key_pem=data["public_key_pem"],
            key_id=data["key_id"],
            created_at=_parse_timestamp(data["created_at"]),
            is_revoked=data["is_revoked"],
        )

    def create_scope_token(
        self,
        actions: list[str],
        ttl: int = 3600,
        metadata: dict[str, Any] | None = None,
    ) -> "ScopeToken":
        """Issue a portable scope token for cross-org verification.

        Wraps issue_sd_token() with resolved action patterns and returns
        a ScopeToken that can be attached to outbound HTTP requests.

        Args:
            actions: Action patterns (semantic names or raw globs).
            ttl: Token time-to-live in seconds.
            metadata: Optional extra claims to embed.

        Returns:
            ScopeToken ready for to_header() or present().

        Example:
            token = agent.create_scope_token(["sql-read", "http-external"])
            requests.get(url, headers=token.to_header())
        """
        from .scope import create_scope_token as _create

        return _create(self, actions, ttl=ttl, metadata=metadata)

    def sign_with_phases(
        self,
        action_type: str,
        context: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> "PhaseChain":
        """Execute the full three-phase signing flow (intent, decision, execution).

        See :func:`asqav.phases.sign_with_phases` for details.
        """
        from .phases import sign_with_phases

        return sign_with_phases(self, action_type, context, trace_id)


def health_check() -> dict[str, Any]:
    """Check API connectivity and return server status.

    Returns:
        Dict with server status info.

    Raises:
        AuthenticationError: If API key is invalid.
        APIError: If the API is unreachable.

    Example:
        status = asqav.health_check()
        print(f"API version: {status['version']}")
    """
    return _get("/health")


def generate_trace_id() -> str:
    """Generate a unique trace ID for correlating actions across a workflow."""
    return uuid.uuid4().hex


def emergency_halt(reason: str = "emergency") -> list[dict]:
    """Revoke all agents immediately. Use in emergencies only."""
    return _post("/agents/emergency-halt", {"reason": reason})


def init(
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    mode: str = "auto",
    org_salt: bytes | None = None,
) -> None:
    """Initialize the asqav SDK.

    Args:
        api_key: Your asqav API key. Can also be set via ASQAV_API_KEY env var.
        base_url: Override API base URL (for testing).
        mode: Wire format for ``Agent.sign``. One of ``"auto"`` (default),
            ``"hash-only"``, or ``"full-payload"``. ``auto`` resolves to
            ``hash-only`` for the asqav cloud (``*.asqav.com``) and
            ``full-payload`` for self-hosted deployments. The
            ``ASQAV_MODE`` env var is consulted when ``mode="auto"``.
        org_salt: Optional 32-byte per-organization salt. When set, the
            SDK uses HMAC-SHA-256 instead of plain SHA-256 for hash-only
            requests. Fetch yours from the asqav dashboard.

    Raises:
        AuthenticationError: If no API key is provided.

    Example:
        import asqav

        # Using parameter
        asqav.init(api_key="sk_...")

        # Using environment variable
        os.environ["ASQAV_API_KEY"] = "sk_..."
        asqav.init()

        # Force hash-only against a custom URL
        asqav.init(api_key="sk_...", mode="hash-only", org_salt=bytes.fromhex("..."))
    """
    global _api_key, _api_base, _client, _mode, _org_salt

    _api_key = api_key or os.environ.get("ASQAV_API_KEY")

    if not _api_key:
        raise AuthenticationError(
            "API key required. Set ASQAV_API_KEY or pass api_key to init(). Get yours at asqav.com"
        )

    if base_url:
        _api_base = base_url

    from ._mode import _resolve_mode

    _mode = _resolve_mode(
        api_base_url=_api_base,
        env=os.environ.get("ASQAV_MODE"),
        explicit=mode,
    )
    _org_salt = org_salt

    # Initialize HTTP client
    if _HTTPX_AVAILABLE:
        _client = httpx.Client(
            base_url=_api_base,
            headers={"X-API-Key": _api_key},
            timeout=30.0,
        )


@with_retry()
def _get(path: str) -> dict[str, Any]:
    """Make a GET request to the API."""
    _ensure_initialized()

    if _HTTPX_AVAILABLE and _client:
        response = _client.get(path)
        _handle_response(response)
        result: dict[str, Any] = response.json()
        return result
    else:
        # Use stdlib urllib if httpx not installed
        return _urllib_request("GET", path)


@with_retry()
def _post(path: str, data: dict[str, Any]) -> dict[str, Any]:
    """Make a POST request to the API."""
    _ensure_initialized()

    if _HTTPX_AVAILABLE and _client:
        response = _client.post(path, json=data)
        _handle_response(response)
        result: dict[str, Any] = response.json()
        return result
    else:
        return _urllib_request("POST", path, data)


@with_retry()
def _patch(path: str, data: dict[str, Any]) -> dict[str, Any]:
    """Make a PATCH request to the API."""
    _ensure_initialized()

    if _HTTPX_AVAILABLE and _client:
        response = _client.patch(path, json=data)
        _handle_response(response)
        result: dict[str, Any] = response.json()
        return result
    else:
        return _urllib_request("PATCH", path, data)


@with_retry()
def _put(path: str, data: dict[str, Any]) -> dict[str, Any]:
    """Make a PUT request to the API."""
    _ensure_initialized()

    if _HTTPX_AVAILABLE and _client:
        response = _client.put(path, json=data)
        _handle_response(response)
        result: dict[str, Any] = response.json()
        return result
    else:
        return _urllib_request("PUT", path, data)


@with_retry()
def _delete(path: str) -> dict[str, Any]:
    """Make a DELETE request to the API."""
    _ensure_initialized()

    if _HTTPX_AVAILABLE and _client:
        response = _client.delete(path)
        _handle_response(response)
        result: dict[str, Any] = response.json()
        return result
    else:
        return _urllib_request("DELETE", path)


def _ensure_initialized() -> None:
    """Ensure the SDK is initialized."""
    if not _api_key:
        raise AuthenticationError("Call asqav.init() first. Get your API key at asqav.com")


def _handle_response(response: Any) -> None:
    """Handle API response errors."""
    if response.status_code == 401:
        raise AuthenticationError("Invalid API key")
    elif response.status_code == 429:
        raise RateLimitError("Rate limit exceeded. Upgrade at asqav.com/pricing")
    elif response.status_code >= 400:
        try:
            error = response.json().get("error", "Unknown error")
        except Exception:
            error = response.text
        raise APIError(error, response.status_code)


def _urllib_request(
    method: str,
    path: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """HTTP client using stdlib urllib (used when httpx not installed)."""
    import json
    import urllib.error
    import urllib.request

    url = urljoin(_api_base, path)
    headers = {
        "X-API-Key": _api_key,
        "Content-Type": "application/json",
    }

    body = json.dumps(data).encode("utf-8") if data else None

    def _do_request() -> dict[str, Any]:
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=30) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return result

    try:
        return _with_retry(_do_request)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise AuthenticationError("Invalid API key") from e
        elif e.code == 429:
            raise RateLimitError("Rate limit exceeded") from e
        else:
            raise APIError(str(e), e.code) from e


# Global agent for decorators
_global_agent: Agent | None = None


def get_agent() -> Agent:
    """Get the global agent for decorator use.

    Returns:
        The global Agent instance.

    Raises:
        AsqavError: If no agent is configured.
    """
    global _global_agent
    if _global_agent is None:
        # Auto-create an agent with default name
        name = _auto_generate_name()
        _global_agent = Agent.create(name)
    return _global_agent


def list_agents() -> list[Agent]:
    """Return every agent visible to the current API key.

    Public surface for the same data ``asqav agents list`` shows. Use this
    in dashboards, scripts, and notebooks instead of reaching into
    ``asqav.client._get('/agents')``.
    """
    data = _get("/agents")
    raw = data if isinstance(data, list) else data.get("agents", [])
    out: list[Agent] = []
    for a in raw:
        out.append(
            Agent(
                agent_id=a["agent_id"],
                name=a["name"],
                public_key=a.get("public_key", ""),
                key_id=a.get("key_id", ""),
                algorithm=a.get("algorithm", ""),
                capabilities=a.get("capabilities", []),
                created_at=_parse_timestamp(a.get("created_at", 0)),
            )
        )
    return out


def _auto_generate_name() -> str:
    """Generate an agent name from environment."""
    env_name = os.environ.get("ASQAV_AGENT_NAME")
    if env_name:
        return env_name

    if hasattr(sys, "argv") and sys.argv:
        import pathlib

        script_path = pathlib.Path(sys.argv[0])
        return f"agent-{script_path.stem}"

    return "asqav-agent"


def secure(func: F) -> F:
    """Decorator to secure function calls with cryptographic signing.

    Wraps the function in an asqav session, signing the call as an action.
    All ML-DSA cryptography happens server-side.

    Args:
        func: The function to secure.

    Returns:
        The wrapped function.

    Example:
        import asqav

        asqav.init(api_key="sk_...")

        @asqav.secure
        def process_data(data: dict) -> dict:
            # This call is signed with cryptographic proof
            return {"processed": True}
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        agent = get_agent()

        # Start session if not active
        if agent._session_id is None:
            agent.start_session()

        # Log the function call
        agent.sign(
            action_type="function:call",
            context={
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
            },
        )

        try:
            result = func(*args, **kwargs)

            # Log success
            agent.sign(
                action_type="function:result",
                context={
                    "function": func.__name__,
                    "success": True,
                },
            )

            return result

        except Exception as e:
            # Log error
            agent.sign(
                action_type="function:error",
                context={
                    "function": func.__name__,
                    "error": str(e),
                },
            )
            raise

    return wrapper  # type: ignore


def secure_async(func: F) -> F:
    """Async version of the @secure decorator.

    Args:
        func: The async function to secure.

    Returns:
        The wrapped async function.

    Example:
        import asqav

        asqav.init(api_key="sk_...")

        @asqav.secure_async
        async def fetch_data(url: str) -> dict:
            # This call is signed with cryptographic proof
            return {"data": "..."}
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        agent = get_agent()

        if agent._session_id is None:
            agent.start_session()

        agent.sign(
            action_type="function:call",
            context={
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
            },
        )

        try:
            result = await func(*args, **kwargs)

            agent.sign(
                action_type="function:result",
                context={
                    "function": func.__name__,
                    "success": True,
                },
            )

            return result

        except Exception as e:
            agent.sign(
                action_type="function:error",
                context={
                    "function": func.__name__,
                    "error": str(e),
                },
            )
            raise

    return wrapper  # type: ignore


def get_session_signatures(session_id: str) -> list[SignedActionResponse]:
    """Get signed actions for a session.

    Args:
        session_id: The session ID.

    Returns:
        List of SignedActionResponse with signature details.
    """
    data = _get(f"/sessions/{session_id}/signatures")

    return [
        SignedActionResponse(
            signature_id=sig["signature_id"],
            agent_id=sig["agent_id"],
            action_id=sig["action_id"],
            action_type=sig["action_type"],
            payload=sig.get("payload"),
            algorithm=sig["algorithm"],
            signed_at=_parse_timestamp(sig["signed_at"]),
            signature_preview=sig["signature_preview"],
            verification_url=sig["verification_url"],
        )
        for sig in data
    ]


def verify_signature(signature_id: str) -> VerificationResponse:
    """Publicly verify a signature by ID.

    This endpoint requires no authentication. Anyone with the signature_id
    can verify that the signature is valid and was created by the agent.

    Args:
        signature_id: The signature ID to verify.

    Returns:
        VerificationResponse with verification details.

    Example:
        result = asqav.verify_signature("sig_abc123")
        if result.verified:
            print(f"Signature valid for agent {result.agent_name}")
    """
    url = f"{_api_base}/verify/{signature_id}"

    # Use httpx if available (better SSL handling)
    if _HTTPX_AVAILABLE:
        response = httpx.get(url, timeout=30.0)
        if response.status_code == 404:
            raise APIError("Signature not found", 404)
        if response.status_code >= 400:
            raise APIError(response.text, response.status_code)
        data: dict[str, Any] = response.json()
    else:
        # Fallback to urllib
        import json
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise APIError("Signature not found", 404) from e
            raise APIError(str(e), e.code) from e

    detail_raw = data.get("verification_detail")
    detail = (
        VerificationDetail(
            signer_key_match=detail_raw["signer_key_match"],
            signature_valid=detail_raw["signature_valid"],
            algorithm_match=detail_raw["algorithm_match"],
            agent_active=detail_raw["agent_active"],
            validation_label=detail_raw["validation_label"],
            signed_at_skew_seconds=detail_raw.get("signed_at_skew_seconds"),
            chain_valid=detail_raw.get("chain_valid"),
            anchor_status_ots=detail_raw.get("anchor_status_ots"),
            anchor_status_rfc3161=detail_raw.get("anchor_status_rfc3161"),
            missing_fields=detail_raw.get("missing_fields"),
        )
        if isinstance(detail_raw, dict)
        else None
    )
    return VerificationResponse(
        signature_id=data["signature_id"],
        agent_id=data["agent_id"],
        agent_name=data["agent_name"],
        action_id=data["action_id"],
        action_type=data["action_type"],
        payload=data.get("payload"),
        signature=data["signature"],
        algorithm=data["algorithm"],
        signed_at=_parse_timestamp(data["signed_at"]),
        verified=data["verified"],
        verification_url=data["verification_url"],
        verification_detail=detail,
    )


@dataclass
class ComplianceReceiptVerification:
    """Outcome of the SDK-side Compliance Receipt sanity check.

    The cloud is the authoritative verifier; this helper is a
    convenience for callers who want to reject obviously bad receipts
    before involving the network. Each boolean reflects one MUST clause
    on draft-marques-asqav-compliance-receipts-00 §5.

    `errors` carries a per-clause failure code so a CLI / dashboard can
    show the user which clause failed. Codes mirror the cloud's
    `validation_label` vocabulary in `routes/verify.py`.
    """

    valid: bool
    fields_present: bool
    receipt_type_in_namespace: bool
    skew_within_bound: bool
    chain_link_rederives: bool
    errors: list[str]


def verify_compliance_receipt(
    envelope: dict[str, Any],
    *,
    predecessor_envelope: dict[str, Any] | None = None,
    now: float | None = None,
) -> ComplianceReceiptVerification:
    """Local-side sanity-check on a Compliance Receipt envelope.

    Validates the four MUSTs the SDK can check without round-tripping
    to the cloud:

    1. All REQUIRED fields are present (§5).
    2. ``receipt_type`` is in the `protectmcp:*` namespace.
    3. ``signed_at`` (or ``issued_at``) is within
       ``SKEW_BOUND_SECONDS`` of ``now``.
    4. ``previousReceiptHash`` rederives over the predecessor envelope
       under JCS, when one is supplied (§5.7). When the receipt is the
       first on its chain (`previousReceiptHash == "0" * 64`) the
       rederivation step is skipped.

    The cloud is authoritative for cryptographic signature checks,
    policy_digest resolution against the Audit Pack, and anchor
    verification. This helper is a convenience.

    Args:
        envelope: The signed receipt envelope, as a dict.
        predecessor_envelope: Optional predecessor envelope. When
            provided, the chain-link is rederived and compared to
            ``envelope["previousReceiptHash"]``.
        now: Optional override for the verifier wall clock (UNIX
            seconds). Defaults to ``time.time()``.

    Returns:
        A :class:`ComplianceReceiptVerification` capturing each MUST.
    """
    from datetime import datetime

    from ._jcs import canonical_json
    from .replay import FIRST_RECEIPT_SEED

    errors: list[str] = []

    # 1. REQUIRED fields present.
    missing = [
        f for f in _COMPLIANCE_REQUIRED_FIELDS if f not in envelope
    ]
    fields_present = not missing
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")

    # 2. receipt_type namespace.
    rt = envelope.get("receipt_type") or envelope.get("type")
    receipt_type_in_namespace = rt in RECEIPT_TYPE_NAMESPACE
    if not receipt_type_in_namespace:
        errors.append("invalid_receipt_type")

    # 3. 300-second skew bound.
    issued_at = envelope.get("issued_at") or envelope.get("signed_at")
    skew_within_bound = False
    if issued_at is None:
        errors.append("missing_issued_at")
    else:
        try:
            if isinstance(issued_at, (int, float)):
                ts = float(issued_at)
            else:
                # ISO 8601; tolerate trailing "Z".
                s = str(issued_at).replace("Z", "+00:00")
                ts = datetime.fromisoformat(s).timestamp()
            wall = now if now is not None else time.time()
            skew = abs(ts - wall)
            skew_within_bound = skew <= SKEW_BOUND_SECONDS
            if not skew_within_bound:
                errors.append("signed_at_skew")
        except (ValueError, TypeError):
            errors.append("invalid_issued_at")

    # 4. Chain link rederives. Only checked when caller supplies the
    # predecessor envelope; first-record seeds skip this.
    chain_link_rederives = True
    expected_prev = envelope.get("previousReceiptHash")
    if expected_prev == FIRST_RECEIPT_SEED:
        # First record on the chain; nothing to rederive.
        pass
    elif predecessor_envelope is not None:
        actual_prev = hashlib.sha256(
            canonical_json(predecessor_envelope)
        ).hexdigest()
        if actual_prev != expected_prev:
            chain_link_rederives = False
            errors.append("chain_link_mismatch")
    # else: caller did not pass a predecessor; cannot rederive. We
    # leave `chain_link_rederives=True` since "did not check" is not the
    # same as "failed to check". Callers needing strictness pass the
    # predecessor.

    valid = (
        fields_present
        and receipt_type_in_namespace
        and skew_within_bound
        and chain_link_rederives
    )
    return ComplianceReceiptVerification(
        valid=valid,
        fields_present=fields_present,
        receipt_type_in_namespace=receipt_type_in_namespace,
        skew_within_bound=skew_within_bound,
        chain_link_rederives=chain_link_rederives,
        errors=errors,
    )


def post_applied_attestation(
    signature_id: str,
    *,
    applied_at: str,
    outcome: str,
    executor_pubkey_b64: str,
    executor_algorithm: str,
    signature_b64: str,
    error_code: str | None = None,
) -> dict[str, Any]:
    """Post the executor side of an action.

    The downstream service that actually carried out the action signs
    `{signature_id, applied_at, outcome, error_code}` with its identity
    key and posts back. If the original signer pinned an executor public
    key (`expected_executor_pubkey_b64` on sign), the server rejects an
    attestation signed by any other key.

    Args:
        signature_id: The sig_... id from the original sign().
        applied_at: ISO timestamp when the action ran.
        outcome: success | fail | partial.
        executor_pubkey_b64: Base64 of the executor public key.
        executor_algorithm: e.g. "ML-DSA-65".
        signature_b64: Base64 of the executor signature over the
            canonical `{signature_id, applied_at, outcome, error_code}`
            JSON (sorted keys, separators (",",":")).
        error_code: Optional error code string.

    Returns:
        Dict with `signature_id` and `applied_attestation` keys.
    """
    body = {
        "applied_at": applied_at,
        "outcome": outcome,
        "executor_pubkey_b64": executor_pubkey_b64,
        "executor_algorithm": executor_algorithm,
        "signature_b64": signature_b64,
    }
    if error_code is not None:
        body["error_code"] = error_code
    return _post(f"/signatures/{signature_id}/applied-attestation", body)


def list_rejected_attempts(
    *,
    failure_reason: str | None = None,
    agent_id: str | None = None,
    hours: int = 24,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """List rejected sign / verify / replay attempts for the org.

    Pro+ feature. Org-scoped via the API key. Returns the most recent
    rejections first.
    """
    params = [f"hours={hours}", f"limit={limit}", f"offset={offset}"]
    if failure_reason:
        params.append(f"failure_reason={failure_reason}")
    if agent_id:
        params.append(f"agent_id={agent_id}")
    return _get("/observability/rejected-attempts?" + "&".join(params))


def verify_output(signature_id: str, expected_output: Any) -> dict[str, Any]:
    """Verify a signed output matches the expected content.

    Checks that the signature is valid and that the output hash stored
    in the signed payload matches a fresh hash of expected_output.

    Args:
        signature_id: The signature ID from sign_output().
        expected_output: The output to verify against the signed hash.

    Returns:
        Dict with verified, signature_valid, and output_matches bools.

    Example:
        result = asqav.verify_output("sig_abc123", {"answer": 42})
        if result["verified"]:
            print("Output is authentic and unmodified")
    """
    try:
        verification = verify_signature(signature_id)
        signature_valid = verification.verified
    except Exception:
        return {
            "verified": False,
            "signature_valid": False,
            "output_matches": False,
        }

    # Compare output hash from the signed payload to a fresh hash.
    output_matches = False
    try:
        stored_hash = ""
        if verification.payload and isinstance(verification.payload, dict):
            context = verification.payload.get("context", verification.payload)
            stored_hash = context.get("output_hash", "")
        if stored_hash:
            expected_hash = _hash_value(expected_output)
            output_matches = stored_hash == expected_hash
    except Exception:
        output_matches = False

    return {
        "verified": signature_valid and output_matches,
        "signature_valid": signature_valid,
        "output_matches": output_matches,
    }


def request_action(
    agent_id: str,
    action_type: str,
    params: dict[str, Any] | None = None,
) -> SigningSessionResponse:
    """Request a multi-party authorized action for an agent.

    Creates a signing session that requires multiple entity approvals
    before the action is authorized (multi-party signing).

    Args:
        agent_id: The agent requesting the action.
        action_type: Type of action (e.g., "deploy:production").
        params: Optional parameters for the action.

    Returns:
        SigningSessionResponse with session details and approval status.

    Raises:
        AuthenticationError: If API key is missing or invalid.
        APIError: If the request fails (e.g., no signing group for agent).

    Example:
        session = asqav.request_action(
            agent_id="agent_abc",
            action_type="deploy:production",
            params={"target": "us-east-1"},
        )
        print(f"Session {session.session_id}: needs {session.approvals_required} approvals")
    """
    body: dict[str, Any] = {
        "agent_id": agent_id,
        "action_type": action_type,
    }
    if params is not None:
        body["params"] = params

    data = _post("/signing-groups/sessions", body)
    return _parse_signing_session(data)


def approve_action(
    session_id: str,
    entity_id: str,
) -> ApprovalResponse:
    """Approve a pending signing action session.

    Adds an entity signature to the session. When enough entities
    have approved (meeting the required count), the action is authorized.

    Args:
        session_id: The signing session to approve.
        entity_id: The signing entity providing approval.

    Returns:
        ApprovalResponse with signing approval status and progress.

    Raises:
        AuthenticationError: If API key is missing or invalid.
        APIError: If the request fails (e.g., session expired, entity already signed).

    Example:
        result = asqav.approve_action(
            session_id="thr_abc123",
            entity_id="ent_xyz789",
        )
        if result.approved:
            print("Action authorized!")
    """
    data = _post(
        f"/signing-groups/sessions/{session_id}/approve",
        {"entity_id": entity_id},
    )
    return ApprovalResponse(
        session_id=data["session_id"],
        entity_id=data["entity_id"],
        signatures_collected=data["signatures_collected"],
        approvals_required=data["approvals_required"],
        status=data["status"],
        approved=data["approved"],
    )


def get_action_status(session_id: str) -> SigningSessionResponse:
    """Get the current status of a signing action session.

    Args:
        session_id: The signing session to check.

    Returns:
        SigningSessionResponse with full session details.

    Raises:
        AuthenticationError: If API key is missing or invalid.
        APIError: If the request fails (e.g., session not found).

    Example:
        status = asqav.get_action_status("thr_abc123")
        print(f"{status.signatures_collected}/{status.approvals_required} approvals")
    """
    data = _get(f"/signing-groups/sessions/{session_id}")
    return _parse_signing_session(data)


def _parse_signing_session(data: dict[str, Any]) -> SigningSessionResponse:
    """Parse API response into a SigningSessionResponse."""
    signatures = [
        SignatureDetail(
            entity_id=s["entity_id"],
            entity_name=s["entity_name"],
            entity_class=s["entity_class"],
            signed_at=s["signed_at"],
        )
        for s in data.get("signatures", [])
    ]
    return SigningSessionResponse(
        session_id=data["session_id"],
        config_id=data["config_id"],
        agent_id=data["agent_id"],
        action_type=data["action_type"],
        action_params=data.get("action_params"),
        approvals_required=data["approvals_required"],
        signatures_collected=data["signatures_collected"],
        status=data["status"],
        signatures=signatures,
        policy_attestation_hash=data.get("policy_attestation_hash"),
        created_at=data["created_at"],
        expires_at=data["expires_at"],
        resolved_at=data.get("resolved_at"),
    )


# --- Signing Groups ---

def create_signing_group(
    agent_id: str,
    min_approvals: int,
    total_shares: int,
) -> SigningGroupResponse:
    """Create a signing group for an agent.

    Args:
        agent_id: The agent to create a signing group for.
        min_approvals: Minimum approvals required (t in t-of-n).
        total_shares: Total key shares to generate (n in t-of-n).

    Returns:
        SigningGroupResponse with config details.

    Example:
        config = asqav.create_signing_group("agt_xxx", min_approvals=2, total_shares=3)
    """
    data = _post("/signing-groups/configs", {
        "agent_id": agent_id,
        "min_approvals": min_approvals,
        "total_shares": total_shares,
    })
    return _parse_signing_group(data)


def get_signing_group(agent_id: str) -> SigningGroupResponse:
    """Get the active signing group for an agent.

    Args:
        agent_id: The agent ID.

    Returns:
        SigningGroupResponse with config details.
    """
    data = _get(f"/signing-groups/configs/{agent_id}")
    return _parse_signing_group(data)


def update_signing_group(
    config_id: str,
    min_approvals: int | None = None,
    is_active: bool | None = None,
) -> SigningGroupResponse:
    """Update a signing group.

    Args:
        config_id: The config to update.
        min_approvals: New minimum approvals value.
        is_active: Activate or deactivate the config.

    Returns:
        SigningGroupResponse with updated config.
    """
    body: dict[str, Any] = {}
    if min_approvals is not None:
        body["min_approvals"] = min_approvals
    if is_active is not None:
        body["is_active"] = is_active
    data = _put(f"/signing-groups/configs/{config_id}", body)
    return _parse_signing_group(data)


def _parse_signing_group(data: dict[str, Any]) -> SigningGroupResponse:
    """Parse API response into a SigningGroupResponse."""
    return SigningGroupResponse(
        id=data["id"],
        agent_id=data["agent_id"],
        min_approvals=data["min_approvals"],
        total_shares=data["total_shares"],
        is_active=data["is_active"],
        created_at=data["created_at"],
        updated_at=data.get("updated_at"),
    )


# --- Signing Entities ---

def add_entity(
    config_id: str,
    entity_class: str,
    label: str,
) -> SigningEntityResponse:
    """Add a signing entity to a signing group.

    Entity classes: A (Agent), B (Human Operator), C (Policy Engine),
    D (Compliance Verifier), E (Organizational Authority).

    Args:
        config_id: The signing group to add the entity to.
        entity_class: Entity class (A, B, C, D, or E).
        label: Human-readable label for the entity.

    Returns:
        SigningEntityResponse with entity details.

    Example:
        entity = asqav.add_entity("cfg_xxx", entity_class="B", label="operator-1")
    """
    data = _post(f"/signing-groups/configs/{config_id}/entities", {
        "entity_class": entity_class,
        "label": label,
    })
    return SigningEntityResponse(
        id=data["id"],
        config_id=data["config_id"],
        entity_class=data["entity_class"],
        label=data["label"],
        is_active=data["is_active"],
        created_at=data["created_at"],
    )


def list_entities(config_id: str) -> list[SigningEntityResponse]:
    """List signing entities for a signing group.

    Args:
        config_id: The signing group.

    Returns:
        List of SigningEntityResponse.
    """
    data = _get(f"/signing-groups/configs/{config_id}/entities")
    return [
        SigningEntityResponse(
            id=e["id"],
            config_id=e["config_id"],
            entity_class=e["entity_class"],
            label=e["label"],
            is_active=e["is_active"],
            created_at=e["created_at"],
        )
        for e in data
    ]


def remove_entity(entity_id: str) -> dict[str, Any]:
    """Remove a signing entity (soft delete).

    Args:
        entity_id: The entity to remove.

    Returns:
        Confirmation dict.
    """
    return _delete(f"/signing-groups/entities/{entity_id}")


# --- Group Keypairs ---

def generate_keypair(config_id: str) -> GroupKeypairResponse:
    """Generate a multi-party ML-DSA keypair with Shamir secret sharing.

    Creates a keypair where the private key is split into shares
    distributed to the signing entities in the config.

    Args:
        config_id: The signing group to generate a keypair for.

    Returns:
        GroupKeypairResponse with public key and metadata.

    Example:
        keypair = asqav.generate_keypair("cfg_xxx")
        print(f"Public key: {keypair.public_key_hex[:32]}...")
    """
    data = _post("/signing-groups/keypairs", {"config_id": config_id})
    return _parse_group_keypair(data)


def get_keypair(keypair_id: str) -> GroupKeypairResponse:
    """Get group keypair details (no raw shares).

    Args:
        keypair_id: The keypair ID.

    Returns:
        GroupKeypairResponse with keypair details.
    """
    data = _get(f"/signing-groups/keypairs/{keypair_id}")
    return _parse_group_keypair(data)


def group_sign(
    keypair_id: str,
    message_hex: str,
) -> GroupSignResponse:
    """Perform multi-party ML-DSA signing.

    Requires that the minimum number of entity approvals has been met.
    Output is a standard FIPS 204 ML-DSA-65 signature.

    Args:
        keypair_id: The group keypair to sign with.
        message_hex: The message to sign (hex-encoded).

    Returns:
        GroupSignResponse with signature and verification status.

    Example:
        result = asqav.group_sign("kp_xxx", message_hex="deadbeef")
        assert result.verified  # Standard ML-DSA verifier works
    """
    data = _post(f"/signing-groups/keypairs/{keypair_id}/sign", {
        "message_hex": message_hex,
    })
    return GroupSignResponse(
        signature_hex=data["signature_hex"],
        message_hex=data["message_hex"],
        keypair_id=data["keypair_id"],
        verified=data["verified"],
        audit_record_id=data.get("audit_record_id"),
    )


def refresh_keypair(keypair_id: str) -> KeyRefreshResponse:
    """Refresh key shares without changing the public key.

    Generates new Shamir shares while preserving the same public key.
    Invalidates all active delegations.

    Args:
        keypair_id: The keypair to refresh.

    Returns:
        KeyRefreshResponse with refresh details.
    """
    data = _post(f"/signing-groups/keypairs/{keypair_id}/refresh", {})
    return KeyRefreshResponse(
        keypair_id=data["keypair_id"],
        refreshed_at=data["refreshed_at"],
        delegations_invalidated=data["delegations_invalidated"],
    )


def recover_share(
    keypair_id: str,
    entity_id: str,
    contributing_entity_ids: list[str],
) -> ShareRecoveryResponse:
    """Recover a lost key share for an entity.

    Requires exactly t contributing entities to reconstruct the share.

    Args:
        keypair_id: The keypair containing the lost share.
        entity_id: The entity whose share is being recovered.
        contributing_entity_ids: Entity IDs providing their shares for recovery.

    Returns:
        ShareRecoveryResponse with recovery details.
    """
    data = _post(f"/signing-groups/keypairs/{keypair_id}/recover", {
        "entity_id": entity_id,
        "contributing_entity_ids": contributing_entity_ids,
    })
    return ShareRecoveryResponse(
        keypair_id=data["keypair_id"],
        recovered_entity_id=data["recovered_entity_id"],
        recovered_at=data["recovered_at"],
    )


def _parse_group_keypair(data: dict[str, Any]) -> GroupKeypairResponse:
    """Parse API response into a GroupKeypairResponse."""
    return GroupKeypairResponse(
        id=data["id"],
        config_id=data["config_id"],
        public_key_hex=data["public_key_hex"],
        min_approvals=data["min_approvals"],
        total_shares=data["total_shares"],
        created_at=data["created_at"],
        status=data.get("status", "active"),
    )


# --- Risk Rules ---

def create_risk_rule(
    name: str,
    action_pattern: str,
    risk_level: str,
    approval_override: int | None = None,
    priority: int = 0,
    entity_weights: dict[str, float] | None = None,
    time_schedule: dict[str, Any] | None = None,
) -> RiskRuleResponse:
    """Create a risk classification rule.

    Risk rules dynamically adjust approval requirements based on action patterns.

    Args:
        name: Rule name.
        action_pattern: Glob pattern to match action types.
        risk_level: Risk level (low, medium, high, critical).
        approval_override: Override the default approval count for matching actions.
        priority: Rule priority (higher = evaluated first).
        entity_weights: Per-class weight multipliers (e.g., {"D": 2.0}).
        time_schedule: Time-dependent approval adjustment.

    Returns:
        RiskRuleResponse with rule details.

    Example:
        rule = asqav.create_risk_rule(
            name="High-risk finance",
            action_pattern="finance.*",
            risk_level="critical",
            approval_override=3,
            priority=100,
        )
    """
    body: dict[str, Any] = {
        "name": name,
        "action_pattern": action_pattern,
        "risk_level": risk_level,
        "priority": priority,
    }
    if approval_override is not None:
        body["approval_override"] = approval_override
    if entity_weights is not None:
        body["entity_weights"] = entity_weights
    if time_schedule is not None:
        body["time_schedule"] = time_schedule
    data = _post("/risk-rules", body)
    return _parse_risk_rule(data)


def list_risk_rules() -> list[RiskRuleResponse]:
    """List all risk rules for the organization.

    Returns:
        List of RiskRuleResponse.
    """
    data = _get("/risk-rules")
    return [_parse_risk_rule(r) for r in data]


def get_risk_rule(rule_id: str) -> RiskRuleResponse:
    """Get a risk rule by ID.

    Args:
        rule_id: The rule ID.

    Returns:
        RiskRuleResponse with rule details.
    """
    data = _get(f"/risk-rules/{rule_id}")
    return _parse_risk_rule(data)


def update_risk_rule(
    rule_id: str,
    **kwargs: Any,
) -> RiskRuleResponse:
    """Update a risk rule.

    Args:
        rule_id: The rule to update.
        **kwargs: Fields to update (name, action_pattern, risk_level,
                  approval_override, priority, entity_weights, time_schedule).

    Returns:
        RiskRuleResponse with updated rule.
    """
    data = _put(f"/risk-rules/{rule_id}", kwargs)
    return _parse_risk_rule(data)


def delete_risk_rule(rule_id: str) -> dict[str, Any]:
    """Delete a risk rule.

    Args:
        rule_id: The rule to delete.

    Returns:
        Confirmation dict.
    """
    return _delete(f"/risk-rules/{rule_id}")


def _parse_risk_rule(data: dict[str, Any]) -> RiskRuleResponse:
    """Parse API response into RiskRuleResponse."""
    return RiskRuleResponse(
        id=data["id"],
        name=data["name"],
        action_pattern=data["action_pattern"],
        risk_level=data["risk_level"],
        approval_override=data.get("approval_override"),
        priority=data["priority"],
        entity_weights=data.get("entity_weights"),
        time_schedule=data.get("time_schedule"),
        created_at=data.get("created_at", ""),
    )


# --- Delegations ---

def create_delegation(
    config_id: str,
    delegator_entity_id: str,
    delegate_entity_id: str,
    expires_in: int = 86400,
) -> DelegationResponse:
    """Create a temporary delegation between Class B entities.

    Allows one entity to delegate their signing authority to another
    with cryptographic expiry.

    Args:
        config_id: The signing group.
        delegator_entity_id: Entity delegating their authority.
        delegate_entity_id: Entity receiving delegated authority.
        expires_in: Delegation duration in seconds (default 24h).

    Returns:
        DelegationResponse with delegation details.

    Example:
        delegation = asqav.create_delegation(
            config_id="cfg_xxx",
            delegator_entity_id="ent_alice",
            delegate_entity_id="ent_bob",
            expires_in=3600,  # 1 hour
        )
    """
    data = _post("/signing-groups/delegations", {
        "config_id": config_id,
        "delegator_entity_id": delegator_entity_id,
        "delegate_entity_id": delegate_entity_id,
        "expires_in": expires_in,
    })
    return _parse_delegation(data)


def list_delegations() -> list[DelegationResponse]:
    """List all active delegations.

    Returns:
        List of DelegationResponse.
    """
    data = _get("/signing-groups/delegations")
    return [_parse_delegation(d) for d in data]


def get_delegation(delegation_id: str) -> DelegationResponse:
    """Get a delegation by ID.

    Args:
        delegation_id: The delegation ID.

    Returns:
        DelegationResponse.
    """
    data = _get(f"/signing-groups/delegations/{delegation_id}")
    return _parse_delegation(data)


def revoke_delegation(delegation_id: str) -> dict[str, Any]:
    """Revoke an active delegation.

    Args:
        delegation_id: The delegation to revoke.

    Returns:
        Confirmation dict.
    """
    return _delete(f"/signing-groups/delegations/{delegation_id}")


def _parse_delegation(data: dict[str, Any]) -> DelegationResponse:
    """Parse API response into a DelegationResponse."""
    return DelegationResponse(
        id=data["id"],
        config_id=data["config_id"],
        delegator_entity_id=data["delegator_entity_id"],
        delegate_entity_id=data["delegate_entity_id"],
        expires_at=data["expires_at"],
        is_active=data["is_active"],
        created_at=data["created_at"],
    )


# --- List Signing Sessions ---

def list_sessions(
    status: str | None = None,
    agent_id: str | None = None,
    limit: int = 100,
) -> list[SigningSessionResponse]:
    """List signing sessions.

    Args:
        status: Filter by status (pending, approved, denied, expired).
        agent_id: Filter by agent ID.
        limit: Max results (default 100).

    Returns:
        List of SigningSessionResponse.
    """
    params = [f"limit={limit}"]
    if status:
        params.append(f"status={status}")
    if agent_id:
        params.append(f"agent_id={agent_id}")
    path = "/signing-groups/sessions?" + "&".join(params)
    data = _get(path)
    return [_parse_signing_session(s) for s in data]


def export_audit_json(
    start_date: str | None = None,
    end_date: str | None = None,
    agent_id: str | None = None,
) -> dict[str, Any]:
    """Export signed actions as JSON for audit trail export (Pro+ tier).

    Args:
        start_date: Filter by start date (ISO format).
        end_date: Filter by end date (ISO format).
        agent_id: Filter by agent ID.

    Returns:
        Dict with export data including signatures and verification URLs.
    """
    params = []
    if start_date:
        params.append(f"start_date={start_date}")
    if end_date:
        params.append(f"end_date={end_date}")
    if agent_id:
        params.append(f"agent_id={agent_id}")

    path = "/export/json"
    if params:
        path += "?" + "&".join(params)

    return _get(path)


def export_audit_csv(
    start_date: str | None = None,
    end_date: str | None = None,
    agent_id: str | None = None,
) -> str:
    """Export signed actions as CSV for audit trail export (Pro+ tier).

    Args:
        start_date: Filter by start date (ISO format).
        end_date: Filter by end date (ISO format).
        agent_id: Filter by agent ID.

    Returns:
        CSV string with signed actions.
    """
    _ensure_initialized()

    params = []
    if start_date:
        params.append(f"start_date={start_date}")
    if end_date:
        params.append(f"end_date={end_date}")
    if agent_id:
        params.append(f"agent_id={agent_id}")

    path = "/export/csv"
    if params:
        path += "?" + "&".join(params)

    if _HTTPX_AVAILABLE and _client:
        response = _client.get(path)
        _handle_response(response)
        return response.text
    else:
        import urllib.error
        import urllib.request

        url = urljoin(_api_base, path)
        headers = {
            "Authorization": f"Bearer {_api_key}",
        }
        request = urllib.request.Request(url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise AuthenticationError("Invalid API key") from e
            elif e.code == 429:
                raise RateLimitError("Rate limit exceeded") from e
            else:
                raise APIError(str(e), e.code) from e


# --- Trust Signal Export ---

def generate_attestation(
    agent_id: str,
    session_id: str | None = None,
    format: str = "json",
) -> dict[str, Any]:
    """Generate a portable attestation document proving an agent's governance status.

    The attestation includes: agent identity, algorithm, signed action count,
    session summary, verification URLs, and the agent's public key for
    independent verification.

    Args:
        agent_id: The agent to generate an attestation for.
        session_id: Optional session to include signature details from.
        format: Output format (currently only "json" is supported).

    Returns:
        Dict containing the attestation document with an embedded signature
        and attestation hash for independent verification.

    Example:
        attestation = asqav.generate_attestation("agt_abc123")
        print(attestation["attestation_hash"])

        # With session details
        attestation = asqav.generate_attestation(
            "agt_abc123",
            session_id="sess_xyz",
        )
    """
    import hashlib
    import json
    from datetime import datetime, timezone

    agent = Agent.get(agent_id)

    # Build signature list from session if provided
    signatures_list: list[dict[str, Any]] = []
    if session_id:
        session_sigs = get_session_signatures(session_id)
        signatures_list = [
            {
                "signature_id": sig.signature_id,
                "action_type": sig.action_type,
                "signed_at": sig.signed_at,
                "verification_url": sig.verification_url,
            }
            for sig in session_sigs
        ]

    # Build the attestation document content (before hashing)
    doc_content: dict[str, Any] = {
        "agent_id": agent.agent_id,
        "agent_name": agent.name,
        "algorithm": agent.algorithm,
        "public_key": agent.public_key,
        "total_signatures": len(signatures_list),
        "session_id": session_id,
        "signatures": signatures_list,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "format_version": "1.0",
    }

    # Hash the document content for tamper detection
    content_bytes = json.dumps(doc_content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    attestation_hash = hashlib.sha256(content_bytes).hexdigest()
    doc_content["attestation_hash"] = attestation_hash

    # Sign the attestation itself using the agent
    sig_result = agent.sign(
        "attestation:generate",
        {
            "attestation_hash": attestation_hash,
            "agent_id": agent_id,
            "session_id": session_id,
            "format": format,
        },
    )

    doc_content["attestation_signature"] = {
        "signature": sig_result.signature,
        "signature_id": sig_result.signature_id,
        "verification_url": sig_result.verification_url,
    }

    return doc_content


def verify_attestation(attestation: dict[str, Any]) -> dict[str, Any]:
    """Verify a previously generated attestation document.

    Checks the attestation signature and verifies each signature in
    the document's signature list.

    Args:
        attestation: The attestation dict returned by generate_attestation().

    Returns:
        Dict with verification results:
            - valid: Whether the attestation signature itself is valid.
            - signatures_checked: Number of signatures verified.
            - all_valid: Whether every signature in the list passed verification.

    Example:
        attestation = asqav.generate_attestation("agt_abc123", session_id="sess_xyz")
        result = asqav.verify_attestation(attestation)
        if result["valid"] and result["all_valid"]:
            print("Attestation fully verified")
    """
    import hashlib
    import json

    # Verify the attestation hash matches the document content
    stored_hash = attestation.get("attestation_hash", "")
    doc_content = {
        k: v
        for k, v in attestation.items()
        if k not in ("attestation_hash", "attestation_signature")
    }
    content_bytes = json.dumps(doc_content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    computed_hash = hashlib.sha256(content_bytes).hexdigest()

    hash_valid = computed_hash == stored_hash

    # Verify the attestation signature
    attestation_sig = attestation.get("attestation_signature", {})
    attestation_sig_id = attestation_sig.get("signature_id", "")

    attestation_valid = False
    if attestation_sig_id:
        try:
            result = verify_signature(attestation_sig_id)
            attestation_valid = result.verified
        except (APIError, AsqavError):
            attestation_valid = False

    # Verify each signature in the list
    signatures = attestation.get("signatures", [])
    signatures_checked = 0
    all_valid = True

    for sig in signatures:
        sig_id = sig.get("signature_id", "")
        if not sig_id:
            all_valid = False
            continue
        try:
            result = verify_signature(sig_id)
            signatures_checked += 1
            if not result.verified:
                all_valid = False
        except (APIError, AsqavError):
            signatures_checked += 1
            all_valid = False

    # If no signatures to check, all_valid stays True (vacuously true)

    return {
        "valid": hash_valid and attestation_valid,
        "signatures_checked": signatures_checked,
        "all_valid": all_valid,
    }


@dataclass
class BudgetCheckResult:
    """Result of a budget pre-check for an agent action."""

    allowed: bool
    current_spend: float
    limit: float
    remaining: float
    reason: str | None = None


class BudgetTracker:
    """Client-side spend budget tracking for agent actions.

    Uses signed records to track cumulative spend, making the budget
    trail tamper-evident and independently verifiable. All spend
    entries are persisted as ML-DSA signatures via the existing
    agent.sign() method, so the trail can be replayed and checked
    against the verification endpoint later.

    This is a client-side helper. The asqav API does not enforce
    budgets directly; enforcement happens in the caller before
    executing the action. The tracker fails closed: if an estimated
    cost would exceed the remaining budget, check() returns
    allowed=False with reason="budget_exhausted".

    Example:
        agent = asqav.Agent.create("my-agent")
        budget = asqav.BudgetTracker(agent, limit=10.0, currency="USD")

        decision = budget.check(estimated_cost=0.25)
        if not decision.allowed:
            raise RuntimeError(decision.reason)

        # ... perform the action ...
        budget.record("api:openai", actual_cost=0.23, context={"model": "gpt-4"})
    """

    def __init__(self, agent: Agent, limit: float, currency: str = "USD"):
        if limit < 0:
            raise ValueError("budget limit must be non-negative")
        self.agent = agent
        self.limit = float(limit)
        self.currency = currency
        self._spend: float = 0.0
        self._records: list[str] = []  # signature_ids

    def check(self, estimated_cost: float) -> BudgetCheckResult:
        """Check if an action with the given cost would exceed the budget.

        Fails closed: negative or non-finite costs are rejected, and
        any cost that would push cumulative spend past the limit is
        denied.

        Args:
            estimated_cost: Expected cost of the pending action.

        Returns:
            BudgetCheckResult describing whether the action is allowed
            and how much budget would remain after it.
        """
        remaining = self.limit - self._spend

        # Fail closed on invalid input.
        try:
            cost = float(estimated_cost)
        except (TypeError, ValueError):
            return BudgetCheckResult(
                allowed=False,
                current_spend=self._spend,
                limit=self.limit,
                remaining=remaining,
                reason="invalid_cost",
            )

        if cost != cost or cost in (float("inf"), float("-inf")):  # NaN or inf
            return BudgetCheckResult(
                allowed=False,
                current_spend=self._spend,
                limit=self.limit,
                remaining=remaining,
                reason="invalid_cost",
            )

        if cost < 0:
            return BudgetCheckResult(
                allowed=False,
                current_spend=self._spend,
                limit=self.limit,
                remaining=remaining,
                reason="invalid_cost",
            )

        if cost > remaining:
            return BudgetCheckResult(
                allowed=False,
                current_spend=self._spend,
                limit=self.limit,
                remaining=remaining,
                reason="budget_exhausted",
            )

        return BudgetCheckResult(
            allowed=True,
            current_spend=self._spend,
            limit=self.limit,
            remaining=remaining - cost,
        )

    def record(
        self,
        action_type: str,
        actual_cost: float,
        context: dict[str, Any] | None = None,
    ) -> SignatureResponse:
        """Record an action cost by signing it. Updates cumulative spend.

        The signature payload includes the cost, currency, resulting
        cumulative spend, and the configured limit, so the trail can
        be independently replayed and verified.

        Args:
            action_type: Type of action being recorded (e.g. "api:openai").
            actual_cost: Realized cost of the action.
            context: Extra context to include in the signed payload.

        Returns:
            SignatureResponse from the underlying sign() call.
        """
        cost = float(actual_cost)
        if cost != cost or cost in (float("inf"), float("-inf")) or cost < 0:
            raise ValueError("actual_cost must be a finite non-negative number")

        self._spend += cost

        payload: dict[str, Any] = {
            "cost": cost,
            "currency": self.currency,
            "cumulative_spend": self._spend,
            "limit": self.limit,
        }
        if context:
            payload.update(context)

        sig = self.agent.sign(f"budget:{action_type}", payload)
        self._records.append(sig.signature_id)
        return sig

    def status(self) -> dict[str, Any]:
        """Get current budget status."""
        return {
            "limit": self.limit,
            "currency": self.currency,
            "spend": self._spend,
            "remaining": self.limit - self._spend,
            "records": len(self._records),
            "signature_ids": list(self._records),
        }
