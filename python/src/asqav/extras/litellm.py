"""LiteLLM integration for asqav.

Install: pip install asqav[litellm]

Two adapters live here:

``AsqavGuardrail``
    Proxy-guardrail seam (pre/post-call hooks). Signs request lifecycle
    events when wired as a LiteLLM guardrail.

``AsqavSigningLogger``
    CustomLogger logging seam. Signs each successful LLM call via the
    Asqav cloud (ML-DSA receipt) and writes a local JSONL index line so
    the cloud signature and the local record are stitched together.

Usage (signing logger)::

    import asqav
    from asqav.extras.litellm import AsqavSigningLogger
    import litellm

    asqav.init("sk_live_...")
    litellm.callbacks = [AsqavSigningLogger()]
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any

try:
    import litellm  # noqa: F401
except ImportError as err:
    raise ImportError(
        "LiteLLM integration requires litellm. "
        "Install with: pip install asqav[litellm]"
    ) from err

try:
    # Optional at import time: the CustomLogger base may not be present in
    # every litellm build. AsqavSigningLogger is duck-typed by litellm's
    # callback dispatch, so falling back to object keeps the module importable
    # (and AsqavGuardrail usable) when the base is unavailable.
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:
    CustomLogger = object  # type: ignore[assignment,misc]

from ._base import AsqavAdapter

logger = logging.getLogger("asqav")

__all__ = ["AsqavGuardrail", "AsqavSigningLogger"]

# Default index path, overridable via ASQAV_LOG_PATH. Aligns with the merged
# litellm AsqavLogger default so local and cloud records share a home.
_DEFAULT_LOG_PATH = os.path.join(
    os.path.expanduser("~"), ".litellm_asqav_audit.jsonl"
)


class AsqavGuardrail(AsqavAdapter):
    """Guardrail that signs LiteLLM calls via Asqav governance.

    Registers as a LiteLLM callback to sign pre-call, post-call,
    error, and moderation events. Signing failures are fail-open -
    they never block LLM requests.

    Args:
        api_key: Optional API key override (uses asqav.init() default).
        agent_name: Name for a new agent (calls Agent.create).
        agent_id: ID of an existing agent (calls Agent.get).
    """

    # -- LiteLLM guardrail callback interface --

    async def async_pre_call_hook(
        self,
        user_api_key_dict: dict,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> None:
        """Sign before an LLM call is made."""
        messages = data.get("messages", [])
        await self._sign_action_async(
            "llm:pre_call",
            {
                "call_type": call_type,
                "model": data.get("model"),
                "message_count": len(messages),
            },
        )

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: dict,
        response: Any,
    ) -> None:
        """Sign after a successful LLM call."""
        context: dict[str, Any] = {
            "model": data.get("model"),
            "response_type": type(response).__name__,
        }
        # Extract token usage if available on the response
        usage = getattr(response, "usage", None)
        if usage is not None:
            context["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        await self._sign_action_async("llm:post_call", context)

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: dict,
    ) -> None:
        """Sign after a failed LLM call.

        Surfaces error events under `risk_class="medium"` so the IETF
        Compliance Receipts profile sees a controlled-vocabulary risk
        signal even when callers do not configure compliance_mode. The
        cloud only emits risk_class on the signed envelope under
        compliance_mode; outside it the value is dropped.
        """
        await self._sign_action_async(
            "llm:error",
            {
                "error_type": type(original_exception).__name__,
                "error_message": str(original_exception),
            },
            risk_class="medium",
        )

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: dict,
        call_type: str,
    ) -> None:
        """Sign moderation events."""
        await self._sign_action_async(
            "llm:moderation",
            {
                "call_type": call_type,
            },
        )


def _content_digest(value: Any) -> str | None:
    """SHA-256 hex digest of a content value, or None when empty.

    Canonical JSON (sorted keys, tight separators) so the digest is
    reproducible from the same value across runs and platforms.
    """
    if value is None:
        return None
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _redact_content_default() -> bool:
    """Read the ASQAV_REDACT_CONTENT env default (digest-only unless 'false')."""
    return os.environ.get("ASQAV_REDACT_CONTENT", "true").lower() != "false"


def _extract_call_fields(
    kwargs: dict[str, Any],
    response_obj: Any,
    start_time: Any,
    end_time: Any,
    redact: bool,
) -> dict[str, Any]:
    """Pull model, provider, usage, and content digests out of a call.

    Raw prompt/response stay out of the signed context unless ``redact``
    is False. Aligns field names with the merged litellm AsqavLogger.
    """
    model = kwargs.get("model", "")
    messages = kwargs.get("messages")

    provider: str | None = None
    prompt_tokens = completion_tokens = total_tokens = None
    finish_reason: str | None = None
    response_content: Any = None
    try:
        provider = (kwargs.get("litellm_params") or {}).get("custom_llm_provider")
    except Exception:
        provider = None
    try:
        usage = getattr(response_obj, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
        choices = getattr(response_obj, "choices", None)
        if choices:
            finish_reason = getattr(choices[0], "finish_reason", None)
            response_content = getattr(
                getattr(choices[0], "message", None), "content", None
            )
    except Exception:
        pass

    latency_ms: int | None = None
    try:
        if start_time is not None and end_time is not None:
            latency_ms = int((end_time - start_time).total_seconds() * 1000)
    except Exception:
        latency_ms = None

    fields: dict[str, Any] = {
        "model": model,
        "provider": provider,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "finish_reason": finish_reason,
        "latency_ms": latency_ms,
        "messages_digest": _content_digest(messages),
        "response_content_digest": _content_digest(response_content),
    }
    if not redact:
        fields["messages"] = messages
        fields["response_content"] = response_content
    return fields


class AsqavSigningLogger(CustomLogger):
    """LiteLLM CustomLogger that cloud-signs each call (ML-DSA receipt).

    On every successful LLM call this signs the call's mapped fields and
    content digests via the Asqav cloud, producing a third-party-verifiable
    ML-DSA receipt. It also appends one local JSONL index line carrying the
    ``signature_id`` + ``verification_url`` (plus the call's digests) so the
    local record and the cloud signature are stitched and verifiable later
    with ``asqav verify-litellm-log``.

    A receipt proves the agent key signed the exact recorded fields and
    digests at sign time, third-party verifiable. It does not reconstruct
    redacted prompt/response content.

    Honesty: this is signing, not enforcement. Signing or network errors
    never raise into the LLM call (fail-soft, reusing the base adapter's
    fail-open signing path).

    Env:
        ASQAV_API_KEY: required. When unset the logger warns once and no-ops.
        ASQAV_LOG_PATH: JSONL index path. Defaults to
            ~/.litellm_asqav_audit.jsonl.
        ASQAV_REDACT_CONTENT: "false" stores raw prompt/response in the
            index. Defaults to digest-only.

    Usage::

        pip install "asqav[litellm]"

        import asqav, litellm
        from asqav.extras.litellm import AsqavSigningLogger

        asqav.init("sk_live_...")
        litellm.callbacks = [AsqavSigningLogger()]

    Args:
        api_key: Optional API key override (else ASQAV_API_KEY / asqav.init).
        agent_name: Name for a new agent (calls Agent.create).
        agent_id: ID of an existing agent (calls Agent.get).
        log_path: JSONL index path override (else ASQAV_LOG_PATH).
        redact_content: Override the ASQAV_REDACT_CONTENT default.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
        log_path: str | None = None,
        redact_content: bool | None = None,
    ) -> None:
        super().__init__()

        self._log_path = log_path or os.environ.get(
            "ASQAV_LOG_PATH", _DEFAULT_LOG_PATH
        )
        self._redact = (
            redact_content
            if redact_content is not None
            else _redact_content_default()
        )
        self._lock = threading.Lock()
        self._warned = False

        # Env-gated: with no key the logger announces no-op (not enforcement)
        # once and signs nothing, rather than half-initialising a client.
        resolved_key = api_key or os.environ.get("ASQAV_API_KEY")
        self._adapter: AsqavAdapter | None = None
        if not resolved_key:
            self._warn_no_key()
            return
        try:
            self._adapter = AsqavAdapter(
                api_key=resolved_key,
                agent_name=agent_name,
                agent_id=agent_id,
            )
        except Exception as exc:
            # Setup failure must never break the host. No-op + warn.
            logger.warning(
                "AsqavSigningLogger setup failed, signing disabled (fail-soft): %s",
                exc,
            )
            self._adapter = None

    def _warn_no_key(self) -> None:
        """Warn once that signing is off so the logger never looks like enforcement."""
        if not self._warned:
            logger.warning(
                "AsqavSigningLogger: ASQAV_API_KEY not set. No receipts are "
                "signed and no enforcement happens. Set ASQAV_API_KEY (or pass "
                "api_key=) to enable cloud-signed receipts."
            )
            self._warned = True

    def _action_type(self, model: str) -> str:
        """Map a call to an action_type like ``llm:gpt-4``."""
        return f"llm:{model}" if model else "llm:call"

    def _sign_and_index(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Sign the call via the cloud then append the local index line.

        Fully fail-soft: any error is swallowed so the LLM call completes.
        """
        if self._adapter is None:
            self._warn_no_key()
            return
        try:
            fields = _extract_call_fields(
                kwargs, response_obj, start_time, end_time, self._redact
            )
            sig = self._adapter._sign_action(
                self._action_type(fields.get("model", "")), fields
            )
            self._append_index(fields, sig)
        except Exception as exc:
            # Last-resort guard: nothing from signing or indexing escapes.
            logger.warning(
                "AsqavSigningLogger sign/index failed (fail-soft): %s", exc
            )

    def _append_index(self, fields: dict[str, Any], sig: Any) -> None:
        """Append one JSONL index line stitching local digests to the cloud receipt."""
        record = {
            "ts": time.time(),
            "model": fields.get("model"),
            "provider": fields.get("provider"),
            "prompt_tokens": fields.get("prompt_tokens"),
            "completion_tokens": fields.get("completion_tokens"),
            "total_tokens": fields.get("total_tokens"),
            "messages_digest": fields.get("messages_digest"),
            "response_content_digest": fields.get("response_content_digest"),
            "signature_id": getattr(sig, "signature_id", None),
            "verification_url": getattr(sig, "verification_url", None),
        }
        with self._lock:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                )

    # -- LiteLLM CustomLogger interface --

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Sync success hook: sign + index the completed call."""
        self._sign_and_index(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        """Async success hook: sign off the event loop so calls keep flowing."""
        import asyncio

        await asyncio.to_thread(
            self._sign_and_index, kwargs, response_obj, start_time, end_time
        )
