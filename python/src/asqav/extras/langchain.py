"""LangChain integration for asqav.

Install: pip install asqav[langchain]

Usage::

    from asqav.extras.langchain import AsqavCallbackHandler
    handler = AsqavCallbackHandler(agent_name="my-langchain-agent")
    chain.invoke(input, config={"callbacks": [handler]})
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install asqav[langchain]"
    )

from ._base import AsqavAdapter


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
        **kwargs: Any,
    ) -> None:
        AsqavAdapter.__init__(
            self, api_key=api_key, agent_name=agent_name, agent_id=agent_id
        )
        BaseCallbackHandler.__init__(self)

    # -- Chain callbacks -------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Sign chain:start with chain name and input keys."""
        chain_id = serialized.get("id", ["unknown"])
        name = serialized.get("name") or chain_id[-1]
        self._sign_action(
            "chain:start",
            {"chain": str(name), "input_keys": list(inputs.keys())},
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
