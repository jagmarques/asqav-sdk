"""LangChain integration for asqav.

Install: pip install asqav[langchain]

Default-on (one call, no per-invoke config)::

    from asqav.extras.langchain import enable_langchain_governance
    enable_langchain_governance(agent_name="my-langchain-agent")
    chain.invoke(input)  # signs receipts, no config={"callbacks": [...]}

Manual attach (unchanged, still supported)::

    from asqav.extras.langchain import AsqavCallbackHandler
    handler = AsqavCallbackHandler(agent_name="my-langchain-agent")
    chain.invoke(input, config={"callbacks": [handler]})
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError as err:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install asqav[langchain]"
    ) from err

from ._base import AsqavAdapter

__all__ = ["AsqavCallbackHandler", "enable_langchain_governance"]


class AsqavCallbackHandler(BaseCallbackHandler, AsqavAdapter):  # type: ignore[misc]
    """LangChain callback handler that auto-signs chain, tool, and LLM events.

    Every chain run, tool call, and LLM interaction is signed via asqav
    for governance and audit. Signing failures are logged but never raise -
    governance must not break the user's AI pipeline.

    Args:
        api_key: Optional API key override (uses asqav.init() default).
        agent_name: Name for a new agent (calls Agent.create).
        agent_id: ID of an existing agent (calls Agent.get).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
        observe: bool = False,
        **kwargs: Any,
    ) -> None:
        AsqavAdapter.__init__(
            self,
            api_key=api_key,
            agent_name=agent_name,
            agent_id=agent_id,
            observe=observe,
        )
        BaseCallbackHandler.__init__(self)

    # -- Chain callbacks -------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Sign chain:start with chain name and input keys.

        serialized may be None on the LCEL/RunnableLambda path; fall back
        to a stable "chain" label so the receipt is always signed.
        """
        # Guard: LCEL/RunnableLambda passes raw str/list/number as inputs.
        input_keys = list(inputs.keys()) if isinstance(inputs, dict) else []
        if not isinstance(serialized, dict):
            self._sign_action(
                "chain:start",
                {"chain": "chain", "input_keys": input_keys},
            )
            return
        chain_id = serialized.get("id", ["unknown"])
        name = serialized.get("name") or chain_id[-1]
        self._sign_action(
            "chain:start",
            {"chain": str(name), "input_keys": input_keys},
        )

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Sign chain:end with output keys."""
        self._sign_action("chain:end", {"output_keys": list(outputs.keys())})

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Sign chain:error with error type and message."""
        self._sign_action(
            "chain:error",
            {"error_type": type(error).__name__, "error": str(error)[:200]},
        )

    # -- Tool callbacks --------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Sign tool:start with tool name and truncated input."""
        name = serialized.get("name") or serialized.get("id", ["unknown"])[-1]
        self._sign_action(
            "tool:start",
            {"tool": str(name), "input": str(input_str)[:200]},
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Sign tool:end with output type and length."""
        output_str = str(output)
        self._sign_action(
            "tool:end",
            {"output_type": type(output).__name__, "output_length": len(output_str)},
        )

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Sign tool:error with error details."""
        self._sign_action(
            "tool:error",
            {"error_type": type(error).__name__, "error": str(error)[:200]},
        )

    # -- LLM callbacks ---------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Sign llm:start with model name and prompt count."""
        model = serialized.get("name") or serialized.get("id", ["unknown"])[-1]
        self._sign_action(
            "llm:start",
            {"model": str(model), "prompt_count": len(prompts)},
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Sign llm:end with generation count and token usage if available."""
        generation_count = sum(len(g) for g in response.generations)
        context: dict[str, Any] = {"generation_count": generation_count}

        if response.llm_output and isinstance(response.llm_output, dict):
            token_usage = response.llm_output.get("token_usage")
            if token_usage and isinstance(token_usage, dict):
                context["token_usage"] = {
                    k: v for k, v in token_usage.items() if isinstance(v, (int, float))
                }

        self._sign_action("llm:end", context)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Sign llm:error with error details."""
        self._sign_action(
            "llm:error",
            {"error_type": type(error).__name__, "error": str(error)[:200]},
        )


# -- Default-on entrypoint -----------------------------------------------------

# LangChain auto-attaches a handler to every run when its ContextVar is set and
# the paired configure hook is registered (langchain_core _configure loop).
_ASQAV_LC_HANDLER: ContextVar[AsqavCallbackHandler | None] = ContextVar(
    "asqav_langchain_handler", default=None
)
_hook_registered = False


def enable_langchain_governance(
    *,
    api_key: str | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
    observe: bool = False,
) -> AsqavCallbackHandler:
    """Turn on default-on asqav governance for LangChain in one call.

    After this call every chain, tool, and LLM run in the current context
    signs an asqav receipt with no per-invoke ``config={"callbacks": [...]}``.
    It registers a langchain_core configure hook once and binds the handler to
    a ContextVar; LangChain's run configuration then attaches it to every run.
    Signing stays fail-open and the returned handler holds the signatures.
    """
    global _hook_registered
    from langchain_core.tracers.context import register_configure_hook

    handler = AsqavCallbackHandler(
        api_key=api_key,
        agent_name=agent_name,
        agent_id=agent_id,
        observe=observe,
    )
    # Register the configure hook at most once so a second call cannot
    # attach the ContextVar handler twice per run (double-sign).
    if not _hook_registered:
        register_configure_hook(_ASQAV_LC_HANDLER, True)
        _hook_registered = True
    _ASQAV_LC_HANDLER.set(handler)
    return handler
