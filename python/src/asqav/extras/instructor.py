"""instructor integration for asqav.

Install: pip install asqav[instructor]

Usage::

    import asqav
    import instructor
    from asqav.extras.instructor import AsqavInstructorHook

    asqav.init(api_key="...")
    client = instructor.from_provider("openai/gpt-4o-mini")

    hook = AsqavInstructorHook(agent_name="my-extractor")
    hook.attach(client)

    # Every client.chat.completions.create(...) call is now signed via asqav.

instructor's hook system fires four events: ``completion:kwargs`` (request),
``completion:response`` (success), ``completion:error`` (HTTP / provider
error), ``parse:error`` (model output failed Pydantic validation). The
adapter signs each event with a stable ``action_type`` so an auditor can
distinguish requests, successes, and the two failure modes.
"""

from __future__ import annotations

from typing import Any

try:
    from instructor.core.hooks import HookName
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "instructor integration requires instructor. "
        "Install with: pip install asqav[instructor]"
    ) from exc

from ._base import AsqavAdapter


def _model_name(kwargs: dict[str, Any]) -> str:
    """Best-effort extraction of the model id from completion kwargs."""
    model = kwargs.get("model")
    if isinstance(model, str):
        return model
    return "unknown"


def _response_model_name(kwargs: dict[str, Any]) -> str | None:
    """Best-effort name of the Pydantic schema instructor is targeting."""
    rm = kwargs.get("response_model")
    if rm is None:
        return None
    try:
        return rm.__name__  # type: ignore[no-any-return]
    except AttributeError:
        try:
            return type(rm).__name__
        except Exception:
            return None


def _exception_name(exc: BaseException | None) -> str | None:
    if exc is None:
        return None
    try:
        return type(exc).__name__
    except Exception:
        return "Exception"


class AsqavInstructorHook(AsqavAdapter):
    """Asqav adapter for instructor clients.

    Signs ``instructor.completion.start`` on every extraction request,
    ``instructor.completion.complete`` on each successful response, and
    ``instructor.completion.error`` / ``instructor.parse.error`` on the two
    failure paths. Signing is fail-open: if Asqav is unreachable the
    extraction still runs.

    Args:
        api_key: Optional API key override (uses ``asqav.init()`` default).
        agent_name: Name for an Asqav agent (calls ``Agent.create``).
        agent_id: ID of an existing Asqav agent (calls ``Agent.get``).

    Example:
        client = instructor.from_provider("openai/gpt-4o-mini")
        hook = AsqavInstructorHook(agent_name="my-extractor")
        hook.attach(client)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(api_key=api_key, agent_name=agent_name, agent_id=agent_id)

    # === Public attachment surface ===

    def attach(self, client: Any) -> None:
        """Register this hook on an instructor client.

        Accepts both sync (``instructor.from_openai`` style) and async
        (``instructor.from_openai(client, ...)`` with the async OpenAI client)
        instructor clients. Both expose the same ``.on()`` interface.
        """
        if not hasattr(client, "on"):
            raise TypeError(
                "client does not look like an instructor client; "
                "expected a .on(hook_name, handler) method"
            )
        client.on(HookName.COMPLETION_KWARGS, self._on_kwargs)
        client.on(HookName.COMPLETION_RESPONSE, self._on_response)
        client.on(HookName.COMPLETION_ERROR, self._on_error)
        client.on(HookName.COMPLETION_LAST_ATTEMPT, self._on_last_attempt)
        client.on(HookName.PARSE_ERROR, self._on_parse_error)

    # === Event handlers ===

    def _on_kwargs(self, *args: Any, **kwargs: Any) -> None:
        """instructor emits the LLM kwargs right before the call."""
        self._sign_action(
            "instructor.completion.start",
            {
                "model": _model_name(kwargs),
                "response_model": _response_model_name(kwargs),
                "stream": bool(kwargs.get("stream", False)),
            },
        )

    def _on_response(self, response: Any) -> None:
        """Successful completion. We only persist hashable, low-PII fields."""
        usage = getattr(response, "usage", None)
        ctx: dict[str, Any] = {"id": getattr(response, "id", None)}
        if usage is not None:
            ctx["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
            ctx["completion_tokens"] = getattr(usage, "completion_tokens", None)
            ctx["total_tokens"] = getattr(usage, "total_tokens", None)
        self._sign_action("instructor.completion.complete", ctx)

    def _on_error(self, error: Exception, **_kwargs: Any) -> None:
        self._sign_action(
            "instructor.completion.error",
            {"error_type": _exception_name(error), "message": str(error)[:500]},
        )

    def _on_last_attempt(self, error: Exception, **_kwargs: Any) -> None:
        self._sign_action(
            "instructor.completion.last_attempt",
            {"error_type": _exception_name(error), "message": str(error)[:500]},
        )

    def _on_parse_error(self, error: Exception) -> None:
        """Pydantic validation failed against response_model."""
        self._sign_action(
            "instructor.parse.error",
            {"error_type": _exception_name(error), "message": str(error)[:500]},
        )
