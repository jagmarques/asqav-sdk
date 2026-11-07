"""Applied attestations, output checks and approval requests."""

from __future__ import annotations

from typing import Any

from . import _transport
from ._common import _hash_value
from ._models import ApprovalResponse, SignatureDetail, SigningSessionResponse
from ._verify import verify_signature


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
    return _transport._post(f"/signatures/{signature_id}/applied-attestation", body)


def list_rejected_attempts(
    *,
    failure_reason: str | None = None,
    agent_id: str | None = None,
    hours: int = 24,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """List rejected sign / verify / replay attempts for the org.

    Org-scoped via the API key. Returns the most recent rejections first.
    """
    params = [f"hours={hours}", f"limit={limit}", f"offset={offset}"]
    if failure_reason:
        params.append(f"failure_reason={failure_reason}")
    if agent_id:
        params.append(f"agent_id={agent_id}")
    return _transport._get("/observability/rejected-attempts?" + "&".join(params))


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

    data = _transport._post("/signing-groups/sessions", body)
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
    data = _transport._post(
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
    data = _transport._get(f"/signing-groups/sessions/{session_id}")
    return _parse_signing_session(data)


    # Parse API response into a SigningSessionResponse.
def _parse_signing_session(data: dict[str, Any]) -> SigningSessionResponse:
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
