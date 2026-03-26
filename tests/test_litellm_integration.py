"""Tests for LiteLLM guardrail integration."""

from __future__ import annotations

import asyncio
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from asqav.client import AsqavError, SignatureResponse

# ---------------------------------------------------------------------------
# Mock litellm before importing the integration module
# ---------------------------------------------------------------------------

_mock_litellm = ModuleType("litellm")
sys.modules.setdefault("litellm", _mock_litellm)

from asqav.extras.litellm import AsqavGuardrail  # noqa: E402

MOCK_SIGN_RESPONSE: dict = {
    "signature": "sig_abc123",
    "signature_id": "sid_abc123",
    "action_id": "act_abc123",
    "timestamp": 1700000001.0,
    "verification_url": "https://api.asqav.com/verify/sid_abc123",
}


def _make_guardrail() -> AsqavGuardrail:
    """Create a guardrail with mocked Agent."""
    with patch("asqav.client._api_key", "sk_test"), \
         patch("asqav.extras._base.Agent") as mock_agent_cls:
        mock_agent_cls.create.return_value = MagicMock()
        return AsqavGuardrail(agent_name="test-litellm")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreCallHook:
    def test_pre_call_hook_signs_action(self):
        guardrail = _make_guardrail()
        mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
        guardrail._sign_action = MagicMock(return_value=mock_sig)

        asyncio.run(guardrail.async_pre_call_hook(
            user_api_key_dict={},
            cache=None,
            data={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
            call_type="completion",
        ))

        guardrail._sign_action.assert_called_once_with(
            "llm:pre_call",
            {
                "call_type": "completion",
                "model": "gpt-4",
                "message_count": 1,
            },
        )

    def test_pre_call_hook_empty_messages(self):
        guardrail = _make_guardrail()
        guardrail._sign_action = MagicMock(return_value=None)

        asyncio.run(guardrail.async_pre_call_hook(
            user_api_key_dict={},
            cache=None,
            data={"model": "gpt-4"},
            call_type="embedding",
        ))

        guardrail._sign_action.assert_called_once_with(
            "llm:pre_call",
            {
                "call_type": "embedding",
                "model": "gpt-4",
                "message_count": 0,
            },
        )


class TestPostCallSuccessHook:
    def test_post_call_success_signs_action(self):
        guardrail = _make_guardrail()
        mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
        guardrail._sign_action = MagicMock(return_value=mock_sig)

        response = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 20
        response.usage.total_tokens = 30

        asyncio.run(guardrail.async_post_call_success_hook(
            data={"model": "gpt-4"},
            user_api_key_dict={},
            response=response,
        ))

        guardrail._sign_action.assert_called_once_with(
            "llm:post_call",
            {
                "model": "gpt-4",
                "response_type": "MagicMock",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            },
        )

    def test_post_call_success_no_usage(self):
        guardrail = _make_guardrail()
        guardrail._sign_action = MagicMock(return_value=None)

        response = MagicMock(spec=[])  # no usage attribute

        asyncio.run(guardrail.async_post_call_success_hook(
            data={"model": "gpt-4"},
            user_api_key_dict={},
            response=response,
        ))

        call_args = guardrail._sign_action.call_args
        context = call_args[0][1]
        assert "usage" not in context


class TestPostCallFailureHook:
    def test_post_call_failure_signs_action(self):
        guardrail = _make_guardrail()
        mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
        guardrail._sign_action = MagicMock(return_value=mock_sig)

        exc = ValueError("rate limit exceeded")

        asyncio.run(guardrail.async_post_call_failure_hook(
            request_data={"model": "gpt-4"},
            original_exception=exc,
            user_api_key_dict={},
        ))

        guardrail._sign_action.assert_called_once_with(
            "llm:error",
            {
                "error_type": "ValueError",
                "error_message": "rate limit exceeded",
            },
        )


class TestModerationHook:
    def test_moderation_hook_signs_action(self):
        guardrail = _make_guardrail()
        mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
        guardrail._sign_action = MagicMock(return_value=mock_sig)

        asyncio.run(guardrail.async_moderation_hook(
            data={"model": "gpt-4"},
            user_api_key_dict={},
            call_type="completion",
        ))

        guardrail._sign_action.assert_called_once_with(
            "llm:moderation",
            {
                "call_type": "completion",
            },
        )


class TestFailOpen:
    def test_fail_open_on_sign_error(self):
        """Async methods do not raise when signing fails (fail-open via base)."""
        guardrail = _make_guardrail()
        # Mock the agent's sign method to raise, exercising the real
        # _sign_action fail-open path in AsqavAdapter.
        guardrail._agent.sign.side_effect = AsqavError("network")

        # None of these should raise
        asyncio.run(guardrail.async_pre_call_hook(
            user_api_key_dict={}, cache=None,
            data={"model": "gpt-4"}, call_type="completion",
        ))
        asyncio.run(guardrail.async_post_call_success_hook(
            data={}, user_api_key_dict={}, response=MagicMock(),
        ))
        asyncio.run(guardrail.async_post_call_failure_hook(
            request_data={}, original_exception=RuntimeError("err"),
            user_api_key_dict={},
        ))
        asyncio.run(guardrail.async_moderation_hook(
            data={}, user_api_key_dict={}, call_type="completion",
        ))
