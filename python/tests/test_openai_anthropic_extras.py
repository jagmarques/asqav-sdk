"""Tests for the asqav[openai] and asqav[anthropic] extras.

Each extra wraps the vendor SDK so every chat call produces a signed asqav
receipt attached as ``response._asqav_receipt``. The vendor SDKs are mocked
via ``sys.modules`` so these tests run without openai or anthropic installed
and without any network access. Signing is fail-soft by design (matching the
rest of the extras: signing records governance, it does not enforce it), so a
sign failure must never block the underlying LLM call.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav  # noqa: E402
from asqav.extras.anthropic import AsqavAnthropic  # noqa: E402
from asqav.extras.openai import AsqavOpenAI  # noqa: E402


    # Stand-in for a vendor response that allows attribute assignment.
class _FakeResponse:

    def __init__(self, model: str, response_id: str, usage) -> None:
        self.model = model
        self.id = response_id
        self.usage = usage


def _receipt() -> SimpleNamespace:
    return SimpleNamespace(
        signature_id="sig_123",
        verification_url="https://asqav.com/verify/sig_123",
    )


    # Stub asqav.init and asqav.Agent.create so no network call happens.
def _patch_asqav(monkeypatch, agent: MagicMock) -> MagicMock:
    init_mock = MagicMock()
    monkeypatch.setattr(asqav, "init", init_mock)
    create_mock = MagicMock(return_value=agent)
    monkeypatch.setattr(asqav.Agent, "create", create_mock)
    return init_mock


    # Inject a fake openai module whose client returns ``response``.
def _install_fake_openai(monkeypatch, response: _FakeResponse):
    completions = MagicMock()
    completions.create.return_value = response
    client_instance = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    fake_module = ModuleType("openai")
    openai_cls = MagicMock(return_value=client_instance)
    fake_module.OpenAI = openai_cls
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return openai_cls, completions


    # Inject a fake anthropic module whose client returns ``response``.
def _install_fake_anthropic(monkeypatch, response: _FakeResponse):
    messages = MagicMock()
    messages.create.return_value = response
    client_instance = SimpleNamespace(messages=messages)
    fake_module = ModuleType("anthropic")
    anthropic_cls = MagicMock(return_value=client_instance)
    fake_module.Anthropic = anthropic_cls
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return anthropic_cls, messages


# -- OpenAI extra --


def test_openai_signs_receipt_and_attaches_it(monkeypatch):
    response = _FakeResponse(
        "gpt-4o",
        "chatcmpl-1",
        SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )
    _, completions = _install_fake_openai(monkeypatch, response)
    agent = MagicMock()
    receipt = _receipt()
    agent.sign.return_value = receipt
    init_mock = _patch_asqav(monkeypatch, agent)

    client = AsqavOpenAI(
        openai_api_key="sk-openai-secret",
        asqav_api_key="sk_live_asqav",
        agent_name="my-openai-agent",
    )
    out = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert out is response
    completions.create.assert_called_once()
    init_mock.assert_called_once_with("sk_live_asqav")

    agent.sign.assert_called_once()
    kwargs = agent.sign.call_args.kwargs
    assert kwargs["action_type"] == "openai:chat:gpt-4o"
    context = kwargs["context"]
    assert context["model"] == "gpt-4o"
    assert context["openai_id"] == "chatcmpl-1"
    assert context["prompt_tokens"] == 3
    assert context["completion_tokens"] == 5
    assert context["total_tokens"] == 8

    assert response._asqav_receipt is receipt


def test_openai_fail_soft_does_not_block_call(monkeypatch):
    response = _FakeResponse("gpt-4o", "chatcmpl-2", None)
    _, completions = _install_fake_openai(monkeypatch, response)
    agent = MagicMock()
    agent.sign.side_effect = RuntimeError("signing backend down")
    _patch_asqav(monkeypatch, agent)

    client = AsqavOpenAI(openai_api_key="sk-x", asqav_api_key="sk_live_y")
    out = client.chat.completions.create(model="gpt-4o", messages=[])

    assert out is response
    completions.create.assert_called_once()
    agent.sign.assert_called_once()
    assert not hasattr(response, "_asqav_receipt")


def test_openai_real_llm_error_propagates(monkeypatch):
    response = _FakeResponse("gpt-4o", "chatcmpl-3", None)
    _, completions = _install_fake_openai(monkeypatch, response)
    completions.create.side_effect = ValueError("upstream api down")
    agent = MagicMock()
    _patch_asqav(monkeypatch, agent)

    client = AsqavOpenAI(openai_api_key="sk-x", asqav_api_key="sk_live_y")
    with pytest.raises(ValueError, match="upstream api down"):
        client.chat.completions.create(model="gpt-4o", messages=[])

    agent.sign.assert_not_called()


def test_openai_context_leaks_no_secrets_or_content(monkeypatch):
    response = _FakeResponse("gpt-4o", "chatcmpl-4", None)
    _install_fake_openai(monkeypatch, response)
    agent = MagicMock()
    agent.sign.return_value = _receipt()
    _patch_asqav(monkeypatch, agent)

    client = AsqavOpenAI(
        openai_api_key="sk-openai-secret",
        asqav_api_key="sk_live_asqav",
    )
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "super secret prompt text"}],
    )

    context = agent.sign.call_args.kwargs["context"]
    rendered = repr(context)
    assert "sk-openai-secret" not in rendered
    assert "sk_live_asqav" not in rendered
    assert "super secret prompt text" not in rendered
    assert "messages" not in context


# -- Anthropic extra --


def test_anthropic_signs_receipt_and_attaches_it(monkeypatch):
    response = _FakeResponse(
        "claude-sonnet-4-20250514",
        "msg_1",
        SimpleNamespace(input_tokens=4, output_tokens=9),
    )
    _, messages = _install_fake_anthropic(monkeypatch, response)
    agent = MagicMock()
    receipt = _receipt()
    agent.sign.return_value = receipt
    init_mock = _patch_asqav(monkeypatch, agent)

    client = AsqavAnthropic(
        anthropic_api_key="sk-ant-secret",
        asqav_api_key="sk_live_asqav",
        agent_name="my-claude-agent",
    )
    out = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert out is response
    messages.create.assert_called_once()
    init_mock.assert_called_once_with("sk_live_asqav")

    agent.sign.assert_called_once()
    kwargs = agent.sign.call_args.kwargs
    assert kwargs["action_type"] == "anthropic:chat:claude-sonnet-4-20250514"
    context = kwargs["context"]
    assert context["model"] == "claude-sonnet-4-20250514"
    assert context["anthropic_id"] == "msg_1"
    assert context["input_tokens"] == 4
    assert context["output_tokens"] == 9

    assert response._asqav_receipt is receipt


def test_anthropic_fail_soft_does_not_block_call(monkeypatch):
    response = _FakeResponse("claude-sonnet-4-20250514", "msg_2", None)
    _, messages = _install_fake_anthropic(monkeypatch, response)
    agent = MagicMock()
    agent.sign.side_effect = RuntimeError("signing backend down")
    _patch_asqav(monkeypatch, agent)

    client = AsqavAnthropic(anthropic_api_key="sk-ant-x", asqav_api_key="sk_live_y")
    out = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=16)

    assert out is response
    messages.create.assert_called_once()
    agent.sign.assert_called_once()
    assert not hasattr(response, "_asqav_receipt")


def test_anthropic_real_llm_error_propagates(monkeypatch):
    response = _FakeResponse("claude-sonnet-4-20250514", "msg_3", None)
    _, messages = _install_fake_anthropic(monkeypatch, response)
    messages.create.side_effect = ValueError("upstream api down")
    agent = MagicMock()
    _patch_asqav(monkeypatch, agent)

    client = AsqavAnthropic(anthropic_api_key="sk-ant-x", asqav_api_key="sk_live_y")
    with pytest.raises(ValueError, match="upstream api down"):
        client.messages.create(model="claude-sonnet-4-20250514", max_tokens=16)

    agent.sign.assert_not_called()


def test_anthropic_context_leaks_no_secrets_or_content(monkeypatch):
    response = _FakeResponse("claude-sonnet-4-20250514", "msg_4", None)
    _install_fake_anthropic(monkeypatch, response)
    agent = MagicMock()
    agent.sign.return_value = _receipt()
    _patch_asqav(monkeypatch, agent)

    client = AsqavAnthropic(
        anthropic_api_key="sk-ant-secret",
        asqav_api_key="sk_live_asqav",
    )
    client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16,
        messages=[{"role": "user", "content": "super secret prompt text"}],
    )

    context = agent.sign.call_args.kwargs["context"]
    rendered = repr(context)
    assert "sk-ant-secret" not in rendered
    assert "sk_live_asqav" not in rendered
    assert "super secret prompt text" not in rendered
    assert "messages" not in context


# -- Packaging --


def test_pyproject_defines_openai_and_anthropic_extras():
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(pyproject_path, "rb") as fh:
        data = tomllib.load(fh)

    opt_deps = data["project"]["optional-dependencies"]
    assert "openai" in opt_deps
    assert "anthropic" in opt_deps
    assert any(dep.startswith("openai") for dep in opt_deps["openai"])
    assert any(dep.startswith("anthropic") for dep in opt_deps["anthropic"])
