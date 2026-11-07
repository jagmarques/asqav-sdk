"""Signature verification against the API."""

from __future__ import annotations

import hashlib
import json as _json
from typing import Any

from .._useragent import USER_AGENT
from . import _transport
from ._common import _parse_timestamp
from ._errors import APIError, _connection_error, _error_message, require_field
from ._models import (
    BitcoinAnchorStatus,
    ExecutionEvidence,
    VerificationDetail,
    VerificationResponse,
)


    # GET the public /verify document; errors map to APIError.
def _fetch_verification(url: str) -> dict[str, Any]:
    if _transport._HTTPX_AVAILABLE:
        try:
            response = _transport.httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=30.0)
        except _transport.httpx.RequestError as exc:
            raise _connection_error(url, exc) from exc
        if response.status_code == 404:
            raise APIError("Signature not found", 404)
        if response.status_code >= 400:
            status = response.status_code
            raise APIError(_error_message(status, response.text), status)
        return _transport._strict_response_json(response)
    import urllib.error
    import urllib.request

    from .. import strict_json

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise APIError("Signature not found", 404) from e
        raise APIError(_error_message(e.code, _transport._http_error_text(e)), e.code) from e
    except (urllib.error.URLError, OSError) as e:
        raise _connection_error(url, e) from e
    try:
        result: dict[str, Any] = strict_json.loads(raw.decode("utf-8"))
    except (_json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _transport._non_json_error(exc) from exc
    return result


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
    url = _transport._join_url(_transport._api_base, f"/verify/{signature_id}")

    data = _fetch_verification(url)

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
            anchor_valid_ots=detail_raw.get("anchor_valid_ots"),
            anchor_valid_rfc3161=detail_raw.get("anchor_valid_rfc3161"),
            anchor_status_ots=detail_raw.get("anchor_status_ots"),
            anchor_status_rfc3161=detail_raw.get("anchor_status_rfc3161"),
            anchor_independent_rfc3161=detail_raw.get("anchor_independent_rfc3161"),
            missing_fields=detail_raw.get("missing_fields"),
            policy_digest_resolved=detail_raw.get("policy_digest_resolved"),
            duplicate_emission_candidate=detail_raw.get("duplicate_emission_candidate"),
            regimes_satisfied=list(detail_raw.get("regimes_satisfied") or []),
        )
        if isinstance(detail_raw, dict)
        else None
    )
    evidence_raw = data.get("execution_evidence")
    execution_evidence = (
        ExecutionEvidence(
            present=evidence_raw["present"],
            label=evidence_raw["label"],
            result_digest=evidence_raw.get("result_digest"),
            result_bound=evidence_raw.get("result_bound", False),
            none_by_design=evidence_raw.get("none_by_design", False),
        )
        if isinstance(evidence_raw, dict)
        else None
    )
    anchor_raw = data.get("bitcoin_anchor")
    bitcoin_anchor = (
        BitcoinAnchorStatus(
            status=anchor_raw.get("status", "none"),
            anchor_tx_ref=anchor_raw.get("anchor_tx_ref"),
            anchor_block_height=anchor_raw.get("anchor_block_height"),
        )
        if isinstance(anchor_raw, dict)
        else None
    )
    return VerificationResponse(
        signature_id=require_field(data, "signature_id"),
        agent_id=require_field(data, "agent_id"),
        agent_name=require_field(data, "agent_name"),
        action_id=require_field(data, "action_id"),
        action_type=require_field(data, "action_type"),
        payload=data.get("payload"),
        signature=require_field(data, "signature"),
        algorithm=require_field(data, "algorithm"),
        signed_at=_parse_timestamp(require_field(data, "signed_at")),
        verified=require_field(data, "verified"),
        verification_url=require_field(data, "verification_url"),
        verification_detail=detail,
        execution_evidence=execution_evidence,
        type=data.get("type", "signature"),
        bitcoin_anchor=bitcoin_anchor,
        signature_envelope=data.get("signature_envelope"),
        anchors=data.get("anchors"),
        algorithm_registry_version=data.get("algorithm_registry_version"),
    )


#: Labels where a valid signature only waits on its anchor; others reject.
_ANCHOR_WAIT_LABELS = frozenset(
    {"valid", "anchor_pending", "anchor_pending_no_cryptographic_proof"}
)


#: Labels the cloud counts as a pass with verified false (routes/verify.py).
_PASS_LABELS = frozenset({"valid_commitment_not_rederivable"})


    # True when the server accepts, or only the anchor is still pending.
def _signature_valid(resp: Any) -> bool:
    if resp.verified is True:
        return True
    detail = getattr(resp, "verification_detail", None)
    if not isinstance(detail, VerificationDetail):
        return False
    if detail.validation_label in _PASS_LABELS:
        return True
    return bool(detail.signature_valid) and detail.validation_label in _ANCHOR_WAIT_LABELS


    # Anchor state: confirmed, invalid, pending, none or unknown.
def _anchor_state(detail: Any) -> str:
    if not isinstance(detail, VerificationDetail):
        return "unknown"
    statuses = (detail.anchor_status_ots, detail.anchor_status_rfc3161)
    if "invalid" in statuses or detail.validation_label == "anchor_invalid":
        return "invalid"
    if detail.anchor_valid_ots is True or detail.anchor_valid_rfc3161 is True:
        return "confirmed"
    if any(statuses) or detail.validation_label.startswith("anchor_pending"):
        return "pending"
    return "none"


def verify(signature_id: str) -> dict[str, Any]:
    """Publicly verify a receipt by id and return a plain dict. No API key.

    Fetches the public ``/verify/{id}`` verdict, then recomputes the receipt's
    ``chain_hash`` locally: the SHA-256 over the RFC 8785 canonical payload,
    the exact value a successor receipt carries as its ``previousReceiptHash``,
    so the chain linkage is reproducible offline.

    Args:
        signature_id: The signature id to verify.

    Returns:
        dict with ``verified``, ``signature_valid``, ``anchor`` (``confirmed``,
        ``pending``, ``invalid``, ``none`` or ``unknown``), ``agent_id``,
        ``agent_name``, ``signature_id``,
        ``algorithm``, ``verification_url``, and ``chain_hash`` (None when the
        server does not expose the payload).

    Example:
        result = asqav.verify("sig_abc123")
        assert result["verified"]
        print(result["agent_id"], result["chain_hash"])
    """
    from .._jcs import canonical_json

    resp = verify_signature(signature_id)
    payload = resp.payload if isinstance(resp.payload, dict) else None
    chain_hash = (
        hashlib.sha256(canonical_json(payload)).hexdigest() if payload else None
    )
    return {
        "verified": resp.verified,
        "signature_valid": _signature_valid(resp),
        "anchor": _anchor_state(resp.verification_detail),
        "agent_id": resp.agent_id,
        "agent_name": resp.agent_name,
        "signature_id": resp.signature_id,
        "algorithm": resp.algorithm,
        "verification_url": resp.verification_url,
        "chain_hash": chain_hash,
    }
