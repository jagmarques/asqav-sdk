"""Tests for the litellm cloud-signing CustomLogger + verify-litellm-log CLI.

No network: litellm is mocked via ModuleType (including the CustomLogger
base class the logger subclasses), and Agent/sign are patched.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import AsqavError, SignatureResponse  # noqa: E402

# === Mock litellm + its CustomLogger base before importing the integration ===

_mock_litellm = ModuleType("litellm")
_mock_integrations = ModuleType("litellm.integrations")
_mock_custom_logger = ModuleType("litellm.integrations.custom_logger")


class _StubCustomLogger:
    """Stand-in for litellm.integrations.custom_logger.CustomLogger."""

    def __init__(self, *args, **kwargs) -> None:
        pass


_mock_custom_logger.CustomLogger = _StubCustomLogger
_mock_integrations.custom_logger = _mock_custom_logger
_mock_litellm.integrations = _mock_integrations

sys.modules.setdefault("litellm", _mock_litellm)
sys.modules.setdefault("litellm.integrations", _mock_integrations)
sys.modules.setdefault(
    "litellm.integrations.custom_logger", _mock_custom_logger
)

from asqav.extras.litellm import AsqavSigningLogger, _content_digest  # noqa: E402

MOCK_SIGN_RESPONSE: dict = {
    "signature": "sig_abc123",
    "signature_id": "sid_abc123",
    "action_id": "act_abc123",
    "timestamp": 1700000001.0,
    "verification_url": "https://api.asqav.com/verify/sid_abc123",
}


def _make_logger(tmp_path, **kwargs) -> AsqavSigningLogger:
    """Build a logger with a mocked Agent and a temp index path."""
    log_path = str(tmp_path / "index.jsonl")
    with patch("asqav.client._api_key", "sk_test"), patch(
        "asqav.extras._base.Agent"
    ) as mock_agent_cls:
        mock_agent_cls.create.return_value = MagicMock()
        return AsqavSigningLogger(
            api_key="sk_test", agent_name="test-litellm", log_path=log_path, **kwargs
        )


def _response_obj(content: str = "hi there") -> MagicMock:
    resp = MagicMock()
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 20
    resp.usage.total_tokens = 30
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = content
    resp.choices = [choice]
    return resp


# === Signing on success ===


class TestSignOnSuccess:
    def test_success_signs_mapped_action_with_digest_context(self, tmp_path):
        logger = _make_logger(tmp_path)
        mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
        logger._adapter._sign_action = MagicMock(return_value=mock_sig)

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "secret prompt"}],
            "litellm_params": {"custom_llm_provider": "openai"},
        }
        logger.log_success_event(kwargs, _response_obj(), 0, 1)

        logger._adapter._sign_action.assert_called_once()
        action_type, context = logger._adapter._sign_action.call_args[0]
        assert action_type == "llm:gpt-4"
        assert context["model"] == "gpt-4"
        assert context["provider"] == "openai"
        assert context["total_tokens"] == 30
        # Content redacted by default: digests present, raw content absent.
        assert context["messages_digest"] == _content_digest(kwargs["messages"])
        assert context["response_content_digest"] == _content_digest("hi there")
        assert "messages" not in context
        assert "response_content" not in context

    def test_success_writes_index_line_with_signature_id(self, tmp_path):
        logger = _make_logger(tmp_path)
        mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
        logger._adapter._sign_action = MagicMock(return_value=mock_sig)

        logger.log_success_event(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "x"}]},
            _response_obj(),
            0,
            1,
        )

        with open(logger._log_path) as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["signature_id"] == "sid_abc123"
        assert rec["verification_url"].endswith("sid_abc123")
        assert rec["model"] == "gpt-4"
        assert rec["messages_digest"]

    def test_redact_false_keeps_raw_content_in_context(self, tmp_path):
        logger = _make_logger(tmp_path, redact_content=False)
        mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
        logger._adapter._sign_action = MagicMock(return_value=mock_sig)

        msgs = [{"role": "user", "content": "raw prompt"}]
        logger.log_success_event(
            {"model": "gpt-4", "messages": msgs}, _response_obj("raw reply"), 0, 1
        )
        _, context = logger._adapter._sign_action.call_args[0]
        assert context["messages"] == msgs
        assert context["response_content"] == "raw reply"

    def test_async_success_signs(self, tmp_path):
        logger = _make_logger(tmp_path)
        mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
        logger._adapter._sign_action = MagicMock(return_value=mock_sig)

        asyncio.run(
            logger.async_log_success_event(
                {"model": "gpt-4", "messages": []}, _response_obj(), 0, 1
            )
        )
        logger._adapter._sign_action.assert_called_once()


# === Fail-soft ===


class TestFailSoft:
    def test_sign_error_does_not_propagate(self, tmp_path):
        logger = _make_logger(tmp_path)
        logger._adapter._agent.sign.side_effect = AsqavError("network down")

        # Real _sign_action fail-open returns None; the call must still complete
        # and no exception escapes the logger.
        logger.log_success_event(
            {"model": "gpt-4", "messages": []}, _response_obj(), 0, 1
        )

    def test_index_write_error_does_not_propagate(self, tmp_path):
        logger = _make_logger(tmp_path)
        mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
        logger._adapter._sign_action = MagicMock(return_value=mock_sig)
        logger._log_path = "/nonexistent-dir/cannot/write/index.jsonl"

        # Write failure is swallowed (fail-soft), no raise into the call.
        logger.log_success_event(
            {"model": "gpt-4", "messages": []}, _response_obj(), 0, 1
        )


# === init() key resolution (documented quickstart) ===


class TestInitKeyResolution:
    def test_init_key_resolves_adapter_and_signs(self, tmp_path, monkeypatch):
        """The documented quickstart: key only via asqav.init(), no env var.

        Pins that AsqavSigningLogger() honors the client key set by
        asqav.init() rather than no-opping. Fails against code that resolves
        the key from api_key/ASQAV_API_KEY only.
        """
        monkeypatch.delenv("ASQAV_API_KEY", raising=False)
        log_path = str(tmp_path / "index.jsonl")

        # Key set ONLY on the client (as asqav.init does), no api_key=, no env.
        with patch("asqav.client._api_key", "sk_test_init"), patch(
            "asqav.extras._base.Agent"
        ) as mock_agent_cls:
            mock_agent_cls.create.return_value = MagicMock()
            logger = AsqavSigningLogger(agent_name="test-init", log_path=log_path)

        assert logger._adapter is not None

        mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
        logger._adapter._sign_action = MagicMock(return_value=mock_sig)
        logger.log_success_event(
            {"model": "gpt-4", "messages": []}, _response_obj(), 0, 1
        )
        logger._adapter._sign_action.assert_called_once()


# === Missing key -> warn + no-op ===


class TestMissingKey:
    def test_no_key_warns_and_noops(self, tmp_path, monkeypatch, caplog):
        monkeypatch.delenv("ASQAV_API_KEY", raising=False)
        log_path = str(tmp_path / "index.jsonl")
        with patch("asqav.client._api_key", None):
            logger = AsqavSigningLogger(log_path=log_path)
        assert logger._adapter is None

        import logging

        with caplog.at_level(logging.WARNING, logger="asqav"):
            # No crash, no index line written.
            logger.log_success_event(
                {"model": "gpt-4", "messages": []}, _response_obj(), 0, 1
            )
        assert not os.path.exists(log_path)
        assert any("ASQAV_API_KEY not set" in r.message for r in caplog.records)
