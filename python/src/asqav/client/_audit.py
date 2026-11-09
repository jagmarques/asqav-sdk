"""Session listing, audit exports and attestations."""

from __future__ import annotations

from typing import Any

from .._useragent import USER_AGENT
from . import _transport
from ._actions import _parse_signing_session
from ._agent import Agent
from ._agent_session import get_session_signatures
from ._errors import APIError, AsqavError, AuthenticationError, RateLimitError
from ._models import SigningSessionResponse
from ._verify import verify_signature

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
    data = _transport._get(path)
    return [_parse_signing_session(s) for s in data]


def export_audit_json(
    start_date: str | None = None,
    end_date: str | None = None,
    agent_id: str | None = None,
) -> dict[str, Any]:
    """Export signed actions as JSON for audit trail export.

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

    return _transport._get(path)


def export_audit_csv(
    start_date: str | None = None,
    end_date: str | None = None,
    agent_id: str | None = None,
) -> str:
    """Export signed actions as CSV for audit trail export.

    Args:
        start_date: Filter by start date (ISO format).
        end_date: Filter by end date (ISO format).
        agent_id: Filter by agent ID.

    Returns:
        CSV string with signed actions.
    """
    _transport._ensure_initialized()

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

    if _transport._HTTPX_AVAILABLE and _transport._client:
        response = _transport._client.get(path)
        _transport._handle_response(response)
        return response.text
    else:
        import urllib.error
        import urllib.request

        url = _transport._join_url(_transport._api_base, path)
        headers = {
            "Authorization": f"Bearer {_transport._api_key}",
            "User-Agent": USER_AGENT,
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
        format: Output format. Only "json" is supported.

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
        result=dict(doc_content),
    )

    doc_content["attestation_signature"] = {
        "signature": sig_result.signature,
        "signature_id": sig_result.signature_id,
        "verification_url": sig_result.verification_url,
    }

    return doc_content


def verify_attestation(attestation: dict[str, Any]) -> dict[str, Any]:
    """Verify a generated attestation document.

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
