"""DSPy integration for asqav.

Install: pip install asqav[dspy]

Usage::

    import asqav
    import dspy
    from asqav.extras.dspy import AsqavDSPyCallback

    asqav.init(api_key="...")
    dspy.configure(callbacks=[AsqavDSPyCallback(agent_name="my-agent")])

    # All subsequent DSPy module, LM, and tool calls are automatically signed.
"""

from __future__ import annotations

from typing import Any

try:
    from dspy.utils.callback import BaseCallback
except ImportError:
    raise ImportError(
        "DSPy integration requires dspy. "
        "Install with: pip install asqav[dspy]"
    )

from ._base import AsqavAdapter


def _safe_class_name(instance: Any) -> str:
    """Best-effort class name extraction for audit context."""
    try:
        return type(instance).__name__
    except Exception:
        return "unknown"


def _exception_name(exception: BaseException | None) -> str | None:
    if exception is None:
        return None
    try:
        return type(exception).__name__
    except Exception:
        return "Exception"


class AsqavDSPyCallback(AsqavAdapter, BaseCallback):  # type: ignore[misc]
    """DSPy callback that signs module, LM, and tool events via asqav.

    Implements ``dspy.utils.callback.BaseCallback`` and records signed
    governance events for every module execution, LM call, and tool
    invocation. Signing is fail-open: asqav failures are logged but never
    raised, so DSPy pipelines keep running even when asqav is unreachable.

    Args:
        api_key: Optional API key override (uses ``asqav.init()`` default).
        agent_name: Name for a new asqav agent (calls ``Agent.create``).
        agent_id: ID of an existing asqav agent (calls ``Agent.get``).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        AsqavAdapter.__init__(
            self, api_key=api_key, agent_name=agent_name, agent_id=agent_id
        )
        BaseCallback.__init__(self)

    # ------------------------------------------------------------------
    # Module handlers
    # ------------------------------------------------------------------

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """Sign a dspy.module.start event."""
        self._sign_action(
            "dspy.module.start",
            {
                "call_id": call_id,
                "module": _safe_class_name(instance),
                "input_keys": sorted(inputs.keys()) if isinstance(inputs, dict) else [],
            },
        )

    def on_module_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        """Sign a dspy.module.end event."""
        self._sign_action(
            "dspy.module.end",
            {
                "call_id": call_id,
                "ok": exception is None,
                "exception": _exception_name(exception),
            },
        )

    # ------------------------------------------------------------------
    # LM handlers
    # ------------------------------------------------------------------

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """Sign a dspy.lm.start event with model name if available."""
        self._sign_action(
            "dspy.lm.start",
            {
                "call_id": call_id,
                "lm": _safe_class_name(instance),
                "model": getattr(instance, "model", None),
                "input_keys": sorted(inputs.keys()) if isinstance(inputs, dict) else [],
            },
        )

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        """Sign a dspy.lm.end event."""
        self._sign_action(
            "dspy.lm.end",
            {
                "call_id": call_id,
                "ok": exception is None,
                "exception": _exception_name(exception),
            },
        )

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """Sign a dspy.tool.start event."""
        tool_name = getattr(instance, "name", None) or _safe_class_name(instance)
        self._sign_action(
            "dspy.tool.start",
            {
                "call_id": call_id,
                "tool": tool_name,
                "input_keys": sorted(inputs.keys()) if isinstance(inputs, dict) else [],
            },
        )

    def on_tool_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        """Sign a dspy.tool.end event."""
        self._sign_action(
            "dspy.tool.end",
            {
                "call_id": call_id,
                "ok": exception is None,
                "exception": _exception_name(exception),
            },
        )

    # ------------------------------------------------------------------
    # Evaluate handlers
    # ------------------------------------------------------------------

    def on_evaluate_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """Sign a dspy.evaluate.start event."""
        self._sign_action(
            "dspy.evaluate.start",
            {
                "call_id": call_id,
                "evaluator": _safe_class_name(instance),
                "input_keys": sorted(inputs.keys()) if isinstance(inputs, dict) else [],
            },
        )

    def on_evaluate_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        """Sign a dspy.evaluate.end event."""
        self._sign_action(
            "dspy.evaluate.end",
            {
                "call_id": call_id,
                "ok": exception is None,
                "exception": _exception_name(exception),
            },
        )

    # ------------------------------------------------------------------
    # Adapter format handlers
    # ------------------------------------------------------------------

    def on_adapter_format_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """Sign a dspy.adapter.format.start event."""
        self._sign_action(
            "dspy.adapter.format.start",
            {
                "call_id": call_id,
                "adapter": _safe_class_name(instance),
                "input_keys": sorted(inputs.keys()) if isinstance(inputs, dict) else [],
            },
        )

    def on_adapter_format_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        """Sign a dspy.adapter.format.end event."""
        self._sign_action(
            "dspy.adapter.format.end",
            {
                "call_id": call_id,
                "ok": exception is None,
                "exception": _exception_name(exception),
            },
        )

    # ------------------------------------------------------------------
    # Adapter parse handlers
    # ------------------------------------------------------------------

    def on_adapter_parse_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """Sign a dspy.adapter.parse.start event."""
        self._sign_action(
            "dspy.adapter.parse.start",
            {
                "call_id": call_id,
                "adapter": _safe_class_name(instance),
                "input_keys": sorted(inputs.keys()) if isinstance(inputs, dict) else [],
            },
        )

    def on_adapter_parse_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        """Sign a dspy.adapter.parse.end event."""
        self._sign_action(
            "dspy.adapter.parse.end",
            {
                "call_id": call_id,
                "ok": exception is None,
                "exception": _exception_name(exception),
            },
        )
