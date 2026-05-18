"""Tests for instructor AsqavInstructorHook integration.

Uses a mock instructor.core.hooks module injected into sys.modules so tests
run without instructor installed.
"""

from __future__ import annotations

import os
import sys
from enum import Enum
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# === Inject fake instructor.core.hooks module before importing the integration ===

_MISSING = object()
_original_modules: dict[str, ModuleType | object] = {}


def _install_instructor_mocks() -> type:
    for mod_name in ("instructor", "instructor.core", "instructor.core.hooks"):
        _original_modules[mod_name] = sys.modules.get(mod_name, _MISSING)

    class HookName(Enum):
        COMPLETION_KWARGS = "completion:kwargs"
        COMPLETION_RESPONSE = "completion:response"
        COMPLETION_ERROR = "completion:error"
        COMPLETION_LAST_ATTEMPT = "completion:last_attempt"
        PARSE_ERROR = "parse:error"

    instructor = ModuleType("instructor")
    instructor_core = ModuleType("instructor.core")
    instructor_core_hooks = ModuleType("instructor.core.hooks")

    instructor_core_hooks.HookName = HookName  # type: ignore[attr-defined]
    instructor_core.hooks = instructor_core_hooks  # type: ignore[attr-defined]
    instructor.core = instructor_core  # type: ignore[attr-defined]

    sys.modules["instructor"] = instructor
    sys.modules["instructor.core"] = instructor_core
    sys.modules["instructor.core.hooks"] = instructor_core_hooks

    return HookName


def _remove_instructor_mocks() -> None:
    for mod_name, original in _original_modules.items():
        if original is _MISSING:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = original  # type: ignore[assignment]
    sys.modules.pop("asqav.extras.instructor", None)


HookName = _install_instructor_mocks()

from asqav.extras.instructor import AsqavInstructorHook


@pytest.fixture
def hook() -> AsqavInstructorHook:
    """Build a hook with _sign_action mocked so we can assert calls."""
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        h = AsqavInstructorHook(agent_name="extractor")
    h._sign_action = MagicMock()  # type: ignore[method-assign]
    return h


# === Initialization ===


def test_no_init_raises_when_no_api_key() -> None:
    from asqav.client import AsqavError

    with patch("asqav.client._api_key", None):
        with pytest.raises(AsqavError):
            AsqavInstructorHook(agent_name="extractor")


# === attach() ===


def test_attach_registers_all_five_hooks(hook: AsqavInstructorHook) -> None:
    client = MagicMock()
    hook.attach(client)

    registered = {call.args[0] for call in client.on.call_args_list}
    assert registered == {
        HookName.COMPLETION_KWARGS,
        HookName.COMPLETION_RESPONSE,
        HookName.COMPLETION_ERROR,
        HookName.COMPLETION_LAST_ATTEMPT,
        HookName.PARSE_ERROR,
    }


def test_attach_rejects_non_instructor_client(hook: AsqavInstructorHook) -> None:
    not_a_client = object()
    with pytest.raises(TypeError):
        hook.attach(not_a_client)


# === completion:kwargs ===


def test_on_kwargs_signs_start(hook: AsqavInstructorHook) -> None:
    class User:
        pass

    hook._on_kwargs(model="gpt-4o-mini", response_model=User, stream=False)
    hook._sign_action.assert_called_once_with(
        "instructor.completion.start",
        {"model": "gpt-4o-mini", "response_model": "User", "stream": False},
    )


def test_on_kwargs_handles_missing_fields(hook: AsqavInstructorHook) -> None:
    """Missing model and response_model still produce a clean signed payload."""
    hook._on_kwargs()
    args, _ = hook._sign_action.call_args
    assert args[0] == "instructor.completion.start"
    assert args[1] == {"model": "unknown", "response_model": None, "stream": False}


# === completion:response ===


def test_on_response_signs_complete_with_usage(hook: AsqavInstructorHook) -> None:
    response = MagicMock(
        id="cmpl_a1",
        usage=MagicMock(prompt_tokens=12, completion_tokens=34, total_tokens=46),
    )
    hook._on_response(response)
    hook._sign_action.assert_called_once_with(
        "instructor.completion.complete",
        {
            "id": "cmpl_a1",
            "prompt_tokens": 12,
            "completion_tokens": 34,
            "total_tokens": 46,
        },
    )


def test_on_response_handles_missing_usage(hook: AsqavInstructorHook) -> None:
    response = MagicMock(spec=["id"])
    response.id = "cmpl_a2"
    hook._on_response(response)
    args, _ = hook._sign_action.call_args
    assert args[1] == {"id": "cmpl_a2"}


# === error events ===


def test_on_error_signs_completion_error(hook: AsqavInstructorHook) -> None:
    hook._on_error(RuntimeError("provider down"), model="gpt-4o-mini")
    args, _ = hook._sign_action.call_args
    assert args[0] == "instructor.completion.error"
    assert args[1]["error_type"] == "RuntimeError"
    assert "provider down" in args[1]["message"]


def test_on_last_attempt_signs_separate_action(hook: AsqavInstructorHook) -> None:
    hook._on_last_attempt(TimeoutError("retry exhausted"))
    args, _ = hook._sign_action.call_args
    assert args[0] == "instructor.completion.last_attempt"
    assert args[1]["error_type"] == "TimeoutError"


def test_on_parse_error_signs_parse_error(hook: AsqavInstructorHook) -> None:
    hook._on_parse_error(ValueError("schema mismatch"))
    args, _ = hook._sign_action.call_args
    assert args[0] == "instructor.parse.error"
    assert args[1]["error_type"] == "ValueError"


# === Fail-open: signing must never block instructor execution ===


def test_fail_open_on_asqav_error() -> None:
    from asqav.client import AsqavError

    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent = MagicMock()
        mock_agent.sign.side_effect = AsqavError("network down")
        mock_agent_cls.create.return_value = mock_agent
        h = AsqavInstructorHook(agent_name="fail-open")

    # None of these should raise even though sign() always errors.
    h._on_kwargs(model="gpt-4o-mini")
    h._on_response(MagicMock())
    h._on_error(RuntimeError("x"))
    h._on_last_attempt(RuntimeError("y"))
    h._on_parse_error(ValueError("z"))

    assert mock_agent.sign.call_count == 5
    assert len(h._signatures) == 0


# === Cleanup ===


def teardown_module() -> None:
    _remove_instructor_mocks()
