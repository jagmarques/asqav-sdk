"""Tests for Strands AsqavStrandsHooks integration.

Uses mock strands modules injected into sys.modules so tests run without
strands-agents installed. Locks the 787 contract: the model-call outcome
travels as the receipt result by default.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_MISSING = object()
_original_modules: dict[str, ModuleType | object] = {}


def _install_strands_mocks() -> type:
    for mod_name in ("strands", "strands.hooks"):
        _original_modules[mod_name] = sys.modules.get(mod_name, _MISSING)

    class HookProvider:
        def __init__(self) -> None:
            pass

    class HookRegistry:
        def add_callback(self, *args: object, **kwargs: object) -> None:
            pass

    class BeforeModelCallEvent:
        pass

    class AfterModelCallEvent:
        pass

    strands = ModuleType("strands")
    strands_hooks = ModuleType("strands.hooks")
    strands_hooks.HookProvider = HookProvider  # type: ignore[attr-defined]
    strands_hooks.HookRegistry = HookRegistry  # type: ignore[attr-defined]
    strands_hooks.BeforeModelCallEvent = BeforeModelCallEvent  # type: ignore
    strands_hooks.AfterModelCallEvent = AfterModelCallEvent  # type: ignore
    strands.hooks = strands_hooks  # type: ignore[attr-defined]
    sys.modules["strands"] = strands
    sys.modules["strands.hooks"] = strands_hooks
    return HookProvider


def _remove_strands_mocks() -> None:
    for mod_name, original in _original_modules.items():
        if original is _MISSING:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = original  # type: ignore[assignment]
    sys.modules.pop("asqav.extras.strands", None)


HookProvider = _install_strands_mocks()

from asqav.extras.strands import AsqavStrandsHooks  # noqa: E402


@pytest.fixture()
def hook() -> AsqavStrandsHooks:
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        h = AsqavStrandsHooks(agent_name="test-strands")
    h._sign_action = MagicMock(return_value=None)
    return h


def test_after_model_call_binds_message_as_result(hook: AsqavStrandsHooks) -> None:
    message = {"role": "assistant", "text": "hi"}
    event = SimpleNamespace(
        stop_response=SimpleNamespace(message=message, stop_reason="end_turn"),
        exception=None,
    )
    hook._on_after_model_call(event)
    hook._sign_action.assert_called_once()
    args = hook._sign_action.call_args
    assert args[0][0] == "strands:model_call"
    assert args[1]["result"] is message


def test_after_model_call_binds_error_as_result(hook: AsqavStrandsHooks) -> None:
    event = SimpleNamespace(stop_response=None, exception=ValueError("down"))
    hook._on_after_model_call(event)
    hook._sign_action.assert_called_once()
    args = hook._sign_action.call_args
    assert args[0][1]["error"] == "ValueError: down"
    assert args[1]["result"] == "ValueError: down"
