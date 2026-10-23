"""Async API client; :class:`AsyncAgent` mirrors :class:`Agent` over ``httpx.AsyncClient``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._sql_match import matches_pattern as _matches_pattern
from ._useragent import USER_AGENT
from .client import (
    APIError,
    AsqavError,
    AuthenticationError,
    PreflightResult,
    RateLimitError,
    SessionResponse,
    SignatureResponse,
    VerificationResponse,
    _connection_error,
    _error_message,
    _join_url,
    _parse_bitcoin_anchor,
    _parse_timestamp,
    _resolve_action_ref,
    _resolve_previous_receipt_hash,
    _strict_response_json,
    require_field,
)
from .patterns import resolve_pattern
from .retry import with_async_retry

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    httpx = None  # type: ignore


    # Ensure httpx is available for async operations.
def _ensure_httpx() -> None:
    if not _HTTPX_AVAILABLE:
        raise AsqavError(
            "httpx is required for async operations. Install with: pip install asqav[httpx]"
        )


    # Ensure the SDK is initialized.
def _ensure_initialized() -> None:
    _get_config()


    # Read ``(api_base, api_key)`` live from the client module (set by init()).
def _get_config() -> tuple[str, str]:
    from . import client as _client_mod

    if not _client_mod._api_key:
        raise AuthenticationError("Call asqav.init() first. Get your API key at asqav.com")
    return _client_mod._api_base, _client_mod._api_key


    # Handle API response errors (async version).
async def _handle_response(response: Any) -> None:
    if response.status_code == 401:
        raise AuthenticationError("Invalid API key")
    elif response.status_code == 429:
        raise RateLimitError("Rate limit exceeded. Upgrade at asqav.com/pricing")
    elif response.status_code >= 400:
        try:
            body = response.json()
        except Exception:
            body = None
        text = response.text if isinstance(response.text, str) else ""
        raise APIError(_error_message(response.status_code, text, body), response.status_code)


    # Send one async request; transport failures become AsqavConnectionError.
async def _send(client: Any, verb: str, api_base: str, path: str, **kwargs: Any) -> Any:
    try:
        return await getattr(client, verb)(path, **kwargs)
    except httpx.RequestError as exc:
        target = path if "://" in path else _join_url(api_base, path)
        raise _connection_error(target, exc) from exc


    # Make an async GET request to the API.
@with_async_retry()
async def _async_get(path: str) -> dict[str, Any]:
    _ensure_httpx()
    _ensure_initialized()
    api_base, api_key = _get_config()

    async with httpx.AsyncClient(
        base_url=api_base,
        headers={"X-API-Key": api_key, "User-Agent": USER_AGENT},
        timeout=30.0,
    ) as client:
        response = await _send(client, "get", api_base, path)
        await _handle_response(response)
        result: dict[str, Any] = _strict_response_json(response)
        return result


    # Make an async POST request to the API.
@with_async_retry()
async def _async_post(path: str, data: dict[str, Any]) -> dict[str, Any]:
    _ensure_httpx()
    _ensure_initialized()
    api_base, api_key = _get_config()

    async with httpx.AsyncClient(
        base_url=api_base,
        headers={"X-API-Key": api_key, "User-Agent": USER_AGENT},
        timeout=30.0,
    ) as client:
        response = await _send(client, "post", api_base, path, json=data)
        await _handle_response(response)
        result: dict[str, Any] = _strict_response_json(response)
        return result


    # Make an async PATCH request to the API.
@with_async_retry()
async def _async_patch(path: str, data: dict[str, Any]) -> dict[str, Any]:
    _ensure_httpx()
    _ensure_initialized()
    api_base, api_key = _get_config()

    async with httpx.AsyncClient(
        base_url=api_base,
        headers={"X-API-Key": api_key, "User-Agent": USER_AGENT},
        timeout=30.0,
    ) as client:
        response = await _send(client, "patch", api_base, path, json=data)
        await _handle_response(response)
        result: dict[str, Any] = _strict_response_json(response)
        return result


    # Async agent representation; mirrors the sync :class:`Agent`.
@dataclass
class AsyncAgent:

    agent_id: str
    name: str
    public_key: str
    key_id: str
    algorithm: str
    capabilities: list[str]
    created_at: float
    _session_id: str | None = field(default=None, repr=False)

        # Create a new agent via Asqav Cloud; ``algorithm`` is ml-dsa-44/65/87.
    @classmethod
    async def create(
        cls,
        name: str,
        algorithm: str = "ml-dsa-65",
        capabilities: list[str] | None = None,
    ) -> AsyncAgent:
        data = await _async_post(
            "/agents/create",
            {
                "name": name,
                "algorithm": algorithm,
                "capabilities": capabilities or [],
            },
        )

        return cls(
            agent_id=require_field(data, "agent_id"),
            name=require_field(data, "name"),
            public_key=require_field(data, "public_key"),
            key_id=require_field(data, "key_id"),
            algorithm=require_field(data, "algorithm"),
            capabilities=require_field(data, "capabilities"),
            created_at=_parse_timestamp(require_field(data, "created_at")),
        )

        # Get an existing agent by ID.
    @classmethod
    async def get(cls, agent_id: str) -> AsyncAgent:
        data = await _async_get(f"/agents/{agent_id}")

        return cls(
            agent_id=require_field(data, "agent_id"),
            name=require_field(data, "name"),
            public_key=require_field(data, "public_key"),
            key_id=require_field(data, "key_id"),
            algorithm=require_field(data, "algorithm"),
            capabilities=require_field(data, "capabilities"),
            created_at=_parse_timestamp(require_field(data, "created_at")),
        )

        # Sign an action (async); ``co_signers`` lists countersigner agents.
    async def sign(
        self,
        action_type: str,
        context: dict[str, Any] | None = None,
        tool_name: str | None = None,
        model_name: str | None = None,
        parent_id: str | None = None,
        *,
        co_signers: list[str] | None = None,
        user_intent: dict[str, Any] | None = None,
        result_digest: str | None = None,
        result: bytes | str | dict | list | None = None,
        receipt_type: str | None = None,
        action_ref: str | None = None,
        compliance_mode: bool | None = None,
    ) -> SignatureResponse:
        from .client import _build_sign_body, _digest_result_value

        if tool_name or model_name or parent_id:
            context = dict(context) if context else {}
            if tool_name:
                context["_tool_name"] = tool_name
            if model_name:
                context["_model_name"] = model_name
            if parent_id:
                context["_parent_id"] = parent_id

        # Default-send the outcome (criterion 787): explicit digest wins.
        if result_digest is None and result is not None:
            result_digest = _digest_result_value(result)

        body = _build_sign_body(
            action_type=action_type,
            context=context,
            session_id=self._session_id,
            agent_id=self.agent_id,
            user_intent=user_intent,
        )
        if co_signers:
            body["co_signers"] = list(co_signers)
        if result_digest is not None:
            body["result_digest"] = result_digest
        if receipt_type is not None:
            body["receipt_type"] = receipt_type
        if action_ref is not None:
            body["action_ref"] = action_ref
        if compliance_mode is not None:
            body["compliance_mode"] = compliance_mode
        data = await _async_post(f"/agents/{self.agent_id}/sign", body)

        return SignatureResponse(
            signature=require_field(data, "signature"),
            signature_id=require_field(data, "signature_id"),
            action_id=require_field(data, "action_id"),
            timestamp=require_field(data, "timestamp"),
            verification_url=require_field(data, "verification_url"),
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
            action_ref=_resolve_action_ref(data),
            previous_receipt_hash=_resolve_previous_receipt_hash(data),
        )

    async def attach_result(
        self,
        action_ref: str,
        result: bytes | str | dict | list,
        *,
        action_type: str = "protectmcp:observation",
        result_digest: str | None = None,
    ) -> SignatureResponse:
        """Sign a follow-up result receipt linked to an earlier receipt (787).

        Mirrors :meth:`asqav.client.Agent.attach_result`: the follow-up
        carries ``receipt_type='protectmcp:observation:result_bound'`` and
        ``action_ref`` naming the originating receipt/action.
        """
        return await self.sign(
            action_type,
            result=result,
            result_digest=result_digest,
            receipt_type="protectmcp:observation:result_bound",
            action_ref=action_ref,
            compliance_mode=True,
        )

        # Countersign a signature record (async); appends co_signatures.
    async def countersign(self, signature_id: str) -> SignatureResponse:
        data = await _async_post(
            f"/agents/{self.agent_id}/countersign/{signature_id}",
            {},
        )
        return SignatureResponse(
            signature=require_field(data, "signature"),
            signature_id=require_field(data, "signature_id"),
            action_id=require_field(data, "action_id"),
            timestamp=require_field(data, "timestamp"),
            verification_url=require_field(data, "verification_url"),
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
        )

        # Start a new session (async).
    async def start_session(self) -> SessionResponse:
        data = await _async_post("/sessions/", {"agent_id": self.agent_id})

        self._session_id = data["session_id"]

        return SessionResponse(
            session_id=data["session_id"],
            agent_id=data["agent_id"],
            status=data["status"],
            started_at=data["started_at"],
        )

        # End the current session (async); ``status``: completed/error/timeout.
    async def end_session(self, status: str = "completed") -> SessionResponse:
        if not self._session_id:
            raise AsqavError("No active session")

        data = await _async_patch(
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

    async def preflight(self, action_type: str) -> PreflightResult:
        """Pre-flight check (async): revocation, suspension and policy status.
        Fail-closed like the sync Agent.preflight; check ``checks_complete``."""
        action_type = resolve_pattern(action_type)
        agent_active = True
        policy_allowed = True
        checks_complete = True
        reasons: list[str] = []

        # Check agent status (revoked / suspended).
        try:
            status_data = await _async_get(f"/agents/{self.agent_id}/status")
            if status_data.get("revoked", False):
                agent_active = False
                reasons.append("agent is revoked")
            if status_data.get("suspended", False):
                agent_active = False
                reasons.append("agent is suspended")
        except Exception as exc:
            checks_complete = False
            reasons.append(f"status check failed ({exc}) - could not verify")

        # Check organization policies.
        try:
            policies = await _async_get("/policies")
            if not isinstance(policies, list):
                # A non-list response is anomalous, so fail closed instead of
                # silently clearing on an empty iteration.
                checks_complete = False
                reasons.append("policy check failed (unexpected response) - could not verify")
            else:
                for p in policies:
                    if not p.get("is_active"):
                        continue
                    pattern = p.get("action_pattern", "")
                    if _matches_pattern(pattern, action_type):
                        if p.get("action") in ("block", "block_and_alert"):
                            policy_allowed = False
                            reasons.append(f"blocked by policy: {p.get('name', 'unknown')}")
        except Exception as exc:
            checks_complete = False
            reasons.append(f"policy check failed ({exc}) - could not verify")

        # A failed fetch leaves the verdict unknown, so do not clear on it.
        cleared = agent_active and policy_allowed and checks_complete

        # Build human-readable explanation from check results.
        if cleared:
            explanation = "Allowed: agent is active and action is permitted by policy"
        elif not checks_complete:
            explanation = "Blocked: could not verify (" + "; ".join(reasons) + ")"
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
            checks_complete=checks_complete,
        )

        # Publicly verify a signature by ID (async).
    async def verify(self, signature_id: str) -> VerificationResponse:
        _ensure_httpx()
        api_base, _ = _get_config()
        url = f"{api_base}/verify/{signature_id}"

        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
            response = await _send(client, "get", api_base, url)
            if response.status_code == 404:
                raise APIError("Signature not found", 404)
            if response.status_code >= 400:
                status = response.status_code
                raise APIError(_error_message(status, response.text), status)
            data: dict[str, Any] = _strict_response_json(response)

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
        )
