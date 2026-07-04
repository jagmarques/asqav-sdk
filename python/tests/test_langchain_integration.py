"""Tests for LangChain AsqavCallbackHandler integration.

Uses mock langchain_core modules injected into sys.modules so tests
run without langchain-core installed.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from asqav.client import AsqavError

# === Mock langchain_core before importing the handler ===

_original_modules: dict[str, ModuleType | None] = {}

# Sentinel for "was not in sys.modules"
_MISSING = object()


# Fake registry mirroring langchain_core's private _configure_hooks list.
# Each entry is the (context_var, inheritable, handle_class, env_var) tuple
# register_configure_hook appends and the _configure loop iterates.
_FAKE_CONFIGURE_HOOKS: list = []


def _install_langchain_mocks() -> type:
    """Inject fake langchain_core modules and return the BaseCallbackHandler class."""
    # Save originals so we can restore later
    for mod_name in (
        "langchain_core",
        "langchain_core.callbacks",
        "langchain_core.outputs",
        "langchain_core.tracers",
        "langchain_core.tracers.context",
    ):
        _original_modules[mod_name] = sys.modules.get(mod_name, _MISSING)  # type: ignore[arg-type]

    # Create BaseCallbackHandler as a real class (not MagicMock) so MRO works
    class BaseCallbackHandler:
        """Mock LangChain BaseCallbackHandler."""

        def __init__(self) -> None:
            pass

    # Create LLMResult as a simple data class
    class LLMResult:
        """Mock LangChain LLMResult."""

        def __init__(
            self,
            generations: list | None = None,
            llm_output: dict | None = None,
        ) -> None:
            self.generations = generations or []
            self.llm_output = llm_output

    def register_configure_hook(
        context_var, inheritable, handle_class=None, env_var=None
    ) -> None:
        """Mock of langchain_core.tracers.context.register_configure_hook."""
        _FAKE_CONFIGURE_HOOKS.append(
            (context_var, inheritable, handle_class, env_var)
        )

    # Wire up the module tree
    langchain_core = ModuleType("langchain_core")
    langchain_core_callbacks = ModuleType("langchain_core.callbacks")
    langchain_core_outputs = ModuleType("langchain_core.outputs")
    langchain_core_tracers = ModuleType("langchain_core.tracers")
    langchain_core_tracers_context = ModuleType("langchain_core.tracers.context")

    langchain_core_callbacks.BaseCallbackHandler = BaseCallbackHandler  # type: ignore[attr-defined]
    langchain_core_outputs.LLMResult = LLMResult  # type: ignore[attr-defined]
    langchain_core_tracers_context.register_configure_hook = register_configure_hook  # type: ignore[attr-defined]
    langchain_core.callbacks = langchain_core_callbacks  # type: ignore[attr-defined]
    langchain_core.outputs = langchain_core_outputs  # type: ignore[attr-defined]
    langchain_core.tracers = langchain_core_tracers  # type: ignore[attr-defined]
    langchain_core_tracers.context = langchain_core_tracers_context  # type: ignore[attr-defined]

    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.callbacks"] = langchain_core_callbacks
    sys.modules["langchain_core.outputs"] = langchain_core_outputs
    sys.modules["langchain_core.tracers"] = langchain_core_tracers
    sys.modules["langchain_core.tracers.context"] = langchain_core_tracers_context

    return LLMResult


def _remove_langchain_mocks() -> None:
    """Restore sys.modules to pre-mock state."""
    for mod_name, original in _original_modules.items():
        if original is _MISSING:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = original  # type: ignore[assignment]

    # Also clear the cached asqav.extras.langchain module so future imports
    # re-evaluate the try/except ImportError guard.
    sys.modules.pop("asqav.extras.langchain", None)


# Install mocks before importing the handler
LLMResult = _install_langchain_mocks()

from asqav.extras.langchain import AsqavCallbackHandler  # noqa: E402, I001


# === Fixtures ===


@pytest.fixture()
def handler():
    """Create an AsqavCallbackHandler with mocked Agent internals."""
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        h = AsqavCallbackHandler(agent_name="test-langchain")
    # Patch _sign_action for call tracking
    h._sign_action = MagicMock(return_value=None)
    return h


# === Chain callback tests ===


def test_on_chain_start_signs_action(handler):
    """on_chain_start signs chain:start with chain name and input keys."""
    handler.on_chain_start(
        serialized={"name": "RetrievalQA", "id": ["langchain", "chains", "RetrievalQA"]},
        inputs={"query": "what is asqav?"},
    )
    handler._sign_action.assert_called_once_with(
        "chain:start",
        {"chain": "RetrievalQA", "input_keys": ["query"]},
    )


def test_on_chain_start_falls_back_to_id(handler):
    """on_chain_start uses last element of id when name is missing."""
    handler.on_chain_start(
        serialized={"id": ["langchain", "chains", "LLMChain"]},
        inputs={"prompt": "hello"},
    )
    handler._sign_action.assert_called_once_with(
        "chain:start",
        {"chain": "LLMChain", "input_keys": ["prompt"]},
    )


def test_on_chain_end_signs_action(handler):
    """on_chain_end signs chain:end with output keys."""
    handler.on_chain_end(outputs={"result": "answer", "source_documents": []})
    handler._sign_action.assert_called_once_with(
        "chain:end",
        {"output_keys": ["result", "source_documents"]},
    )


def test_on_chain_error_signs_action(handler):
    """on_chain_error signs chain:error with error type and message."""
    handler.on_chain_error(error=ValueError("bad input"))
    handler._sign_action.assert_called_once_with(
        "chain:error",
        {"error_type": "ValueError", "error": "bad input"},
    )


# === Tool callback tests ===


def test_on_tool_start_signs_action(handler):
    """on_tool_start signs tool:start with tool name and input."""
    handler.on_tool_start(
        serialized={"name": "Calculator"},
        input_str="2 + 2",
    )
    handler._sign_action.assert_called_once_with(
        "tool:start",
        {"tool": "Calculator", "input": "2 + 2"},
    )


def test_on_tool_end_signs_action(handler):
    """on_tool_end signs tool:end with output type and length."""
    handler.on_tool_end(output="4")
    handler._sign_action.assert_called_once_with(
        "tool:end",
        {"output_type": "str", "output_length": 1},
    )


def test_on_tool_error_signs_action(handler):
    """on_tool_error signs tool:error with error details."""
    handler.on_tool_error(error=RuntimeError("tool broke"))
    handler._sign_action.assert_called_once_with(
        "tool:error",
        {"error_type": "RuntimeError", "error": "tool broke"},
    )


# === LLM callback tests ===


def test_on_llm_start_signs_action(handler):
    """on_llm_start signs llm:start with model name and prompt count."""
    handler.on_llm_start(
        serialized={"name": "gpt-4", "id": ["langchain", "llms", "openai"]},
        prompts=["Hello", "World"],
    )
    handler._sign_action.assert_called_once_with(
        "llm:start",
        {"model": "gpt-4", "prompt_count": 2},
    )


def test_on_llm_end_signs_action(handler):
    """on_llm_end signs llm:end with generation count and token usage."""
    response = LLMResult(
        generations=[["gen1", "gen2"]],
        llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 20}},
    )
    handler.on_llm_end(response=response)
    handler._sign_action.assert_called_once_with(
        "llm:end",
        {
            "generation_count": 2,
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 20},
        },
    )


def test_on_llm_end_no_token_usage(handler):
    """on_llm_end works when llm_output has no token_usage."""
    response = LLMResult(generations=[["gen1"]], llm_output={})
    handler.on_llm_end(response=response)
    handler._sign_action.assert_called_once_with(
        "llm:end",
        {"generation_count": 1},
    )


def test_on_llm_end_no_llm_output(handler):
    """on_llm_end works when llm_output is None."""
    response = LLMResult(generations=[[]], llm_output=None)
    handler.on_llm_end(response=response)
    handler._sign_action.assert_called_once_with(
        "llm:end",
        {"generation_count": 0},
    )


def test_on_llm_error_signs_action(handler):
    """on_llm_error signs llm:error with error details."""
    handler.on_llm_error(error=TimeoutError("model timeout"))
    handler._sign_action.assert_called_once_with(
        "llm:error",
        {"error_type": "TimeoutError", "error": "model timeout"},
    )


# === Edge cases and fail-open ===


def test_tool_input_truncated(handler):
    """Long tool inputs are truncated to 200 chars."""
    long_input = "x" * 500
    handler.on_tool_start(serialized={"name": "Search"}, input_str=long_input)

    call_args = handler._sign_action.call_args
    assert len(call_args[0][1]["input"]) == 200


def test_fail_open_on_sign_error():
    """Callback methods do not raise when agent.sign raises AsqavError.

    Uses the real _sign_action (from AsqavAdapter) with a mocked agent
    whose sign method raises AsqavError, verifying the fail-open path.
    """
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent = MagicMock()
        mock_agent.sign.side_effect = AsqavError("network timeout")
        mock_agent_cls.create.return_value = mock_agent
        h = AsqavCallbackHandler(agent_name="fail-open-test")

    # None of these should raise - AsqavError is caught by _sign_action
    h.on_chain_start(serialized={"name": "test"}, inputs={"q": "hi"})
    h.on_chain_end(outputs={"r": "bye"})
    h.on_chain_error(error=ValueError("err"))
    h.on_tool_start(serialized={"name": "t"}, input_str="in")
    h.on_tool_end(output="out")
    h.on_tool_error(error=RuntimeError("err"))
    h.on_llm_start(serialized={"name": "m"}, prompts=["p"])
    h.on_llm_end(response=LLMResult(generations=[[]]))
    h.on_llm_error(error=TimeoutError("err"))

    # All 9 calls attempted, none succeeded (all returned None)
    assert mock_agent.sign.call_count == 9
    assert len(h._signatures) == 0


# === Default-on entrypoint tests ===

from asqav.extras import langchain as _lc_mod  # noqa: E402


def _collect_default_handlers() -> list:
    """Replicate langchain_core's _configure loop over the configure hooks.

    Reads each registered hook's ContextVar and collects the bound handler,
    exactly as langchain attaches inheritable handlers to a run without the
    caller passing config={"callbacks": [...]}.
    """
    handlers = []
    for context_var, _inheritable, _handle_class, _env_var in _FAKE_CONFIGURE_HOOKS:
        h = context_var.get()
        if h is not None:
            handlers.append(h)
    return handlers


@pytest.fixture()
def clean_hook_state():
    """Reset the configure-hook registry and the one-time registration flag."""
    _FAKE_CONFIGURE_HOOKS.clear()
    _lc_mod._hook_registered = False
    _lc_mod._ASQAV_LC_HANDLER.set(None)
    yield
    _FAKE_CONFIGURE_HOOKS.clear()
    _lc_mod._hook_registered = False
    _lc_mod._ASQAV_LC_HANDLER.set(None)


def test_enable_registers_configure_hook(clean_hook_state):
    """One call registers exactly one langchain configure hook."""
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        _lc_mod.enable_langchain_governance(agent_name="default-on")

    assert len(_FAKE_CONFIGURE_HOOKS) == 1
    context_var, inheritable, _hc, _ev = _FAKE_CONFIGURE_HOOKS[0]
    assert context_var is _lc_mod._ASQAV_LC_HANDLER
    assert inheritable is True


def test_default_on_signs_without_explicit_callbacks(clean_hook_state):
    """A run that passes NO callbacks still signs, via the auto-attached handler.

    Non-vacuous: if enable_langchain_governance did not register the hook and
    bind the ContextVar (the auto-attach), _collect_default_handlers finds no
    handler, no signing happens, and this assertion fails.
    """
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        handler = _lc_mod.enable_langchain_governance(agent_name="default-on")
    handler._sign_action = MagicMock(return_value=None)

    # Simulate LangChain configuring a run with no explicit callbacks.
    default_handlers = _collect_default_handlers()
    assert handler in default_handlers, "handler was not auto-attached"

    for h in default_handlers:
        h.on_tool_start(serialized={"name": "Search"}, input_str="q")

    handler._sign_action.assert_called_once_with(
        "tool:start",
        {"tool": "Search", "input": "q"},
    )


def test_on_chain_start_none_serialized_signs_receipt(handler):
    """on_chain_start with serialized=None still signs a chain:start receipt.

    Non-vacuous: without the isinstance(serialized, dict) guard, calling
    serialized.get(...) on None raises AttributeError; LangChain swallows the
    callback exception so the receipt is silently dropped. Revert the guard
    in on_chain_start and this assertion fails (no sign call).
    """
    handler.on_chain_start(
        serialized=None,
        inputs={"text": "hello"},
    )
    handler._sign_action.assert_called_once_with(
        "chain:start",
        {"chain": "chain", "input_keys": ["text"]},
    )


def test_on_chain_start_non_dict_serialized_signs_receipt(handler):
    """on_chain_start with a non-dict serialized still signs a receipt."""
    handler.on_chain_start(
        serialized="some-string",  # type: ignore[arg-type]
        inputs={"q": "x"},
    )
    handler._sign_action.assert_called_once_with(
        "chain:start",
        {"chain": "chain", "input_keys": ["q"]},
    )


def test_on_chain_start_none_serialized_non_dict_inputs_signs_receipt(handler):
    """on_chain_start(None, 'hello') yields input_keys=[] - never crashes.

    Non-vacuous: without the isinstance(inputs, dict) guard, calling
    list('hello'.keys()) raises AttributeError; LangChain swallows the exception
    so the receipt is silently dropped. Revert the guard and this test fails
    (sign not called, AttributeError swallowed).
    """
    handler.on_chain_start(
        serialized=None,
        inputs="hello",  # type: ignore[arg-type]
    )
    handler._sign_action.assert_called_once_with(
        "chain:start",
        {"chain": "chain", "input_keys": []},
    )


def test_on_chain_start_dict_serialized_non_dict_inputs_signs_receipt(handler):
    """on_chain_start({'name':'X'}, 'hello') still signs with input_keys=[].

    Non-vacuous: without the guard the dict-serialized branch also calls
    list(inputs.keys()) on the raw string, raising AttributeError and causing
    a silent receipt drop. Revert the guard and this test fails.
    """
    handler.on_chain_start(
        serialized={"name": "MyChain"},
        inputs="hello",  # type: ignore[arg-type]
    )
    handler._sign_action.assert_called_once_with(
        "chain:start",
        {"chain": "MyChain", "input_keys": []},
    )


def test_enable_hook_registered_once_across_calls(clean_hook_state):
    """Repeated enable calls register the hook once and rebind the ContextVar."""
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        _lc_mod.enable_langchain_governance(agent_name="first")
        second = _lc_mod.enable_langchain_governance(agent_name="second")

    assert len(_FAKE_CONFIGURE_HOOKS) == 1
    assert _lc_mod._ASQAV_LC_HANDLER.get() is second


# === Cleanup: restore sys.modules ===


def teardown_module() -> None:
    """Remove mock langchain_core from sys.modules."""
    _remove_langchain_mocks()
