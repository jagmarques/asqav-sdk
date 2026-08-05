"""Ergonomic signing: ``@asqav.sign`` decorator and ``asqav.session()`` (sync + async)."""

from __future__ import annotations

import functools
import inspect
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, TypeVar

from .client import Agent, SessionResponse, SignatureResponse, get_agent

F = TypeVar("F")


# === Session context manager ===


    # Groups sign calls under a single session; created by :func:`session`.
class Session:

    def __init__(self, agent: Agent, session_response: SessionResponse) -> None:
        self._agent = agent
        self._session_response = session_response

    @property
    def session_id(self) -> str:
        return self._session_response.session_id

        # Sign an action within this session.
    def sign(self, action_type: str, context: dict[str, Any] | None = None) -> SignatureResponse:
        return self._agent.sign(action_type, context)


    # Context manager that groups sign calls; signs ``session:error`` on exception.
@contextmanager
def session() -> Generator[Session, None, None]:
    agent = get_agent()
    session_resp = agent.start_session()
    sess = Session(agent, session_resp)

    try:
        yield sess
    except Exception as e:
        # Sign the error before ending the session
        agent.sign(
            action_type="session:error",
            context={"error": str(e), "error_type": type(e).__name__},
        )
        agent.end_session(status="error")
        raise
    else:
        agent.end_session(status="completed")


    # Async counterpart of :func:`session`; signs ``session:error`` on exception.
@asynccontextmanager
async def async_session() -> AsyncGenerator[Session, None]:
    agent = get_agent()
    session_resp = agent.start_session()
    sess = Session(agent, session_resp)

    try:
        yield sess
    except Exception as e:
        agent.sign(
            action_type="session:error",
            context={"error": str(e), "error_type": type(e).__name__},
        )
        agent.end_session(status="error")
        raise
    else:
        agent.end_session(status="completed")


# === sign decorator ===


def sign(func: Any = None, *, action_type: str | None = None) -> Any:
    """Decorator that auto-signs function execution (sync + async, with or without parens).

    ``action_type`` overrides the default ``"function:call"``.

    Trust contract (fail-closed): the ``function:call`` sign runs BEFORE the wrapped
    body. If signing is refused (a blocked policy, a non-2xx response, or a network
    error), the raised error propagates and the body never executes. Unsigned actions
    abort by default - the caller does not need to check a return value to stay safe.
    """

    def decorator(fn: Any) -> Any:
        call_type = action_type or "function:call"

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                agent = get_agent()

                agent.sign(
                    action_type=call_type,
                    context={
                        "function": fn.__name__,
                        "module": fn.__module__,
                        "args_count": len(args),
                    },
                )

                try:
                    result = await fn(*args, **kwargs)

                    agent.sign(
                        action_type="function:result",
                        context={
                            "function": fn.__name__,
                            "success": True,
                        },
                    )

                    return result

                except Exception as e:
                    agent.sign(
                        action_type="function:error",
                        context={
                            "function": fn.__name__,
                            "error": str(e),
                        },
                    )
                    raise

            return async_wrapper

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                agent = get_agent()

                agent.sign(
                    action_type=call_type,
                    context={
                        "function": fn.__name__,
                        "module": fn.__module__,
                        "args_count": len(args),
                    },
                )

                try:
                    result = fn(*args, **kwargs)

                    agent.sign(
                        action_type="function:result",
                        context={
                            "function": fn.__name__,
                            "success": True,
                        },
                    )

                    return result

                except Exception as e:
                    agent.sign(
                        action_type="function:error",
                        context={
                            "function": fn.__name__,
                            "error": str(e),
                        },
                    )
                    raise

            return sync_wrapper

    if func is not None:
        # Called without parens: @asqav.sign
        return decorator(func)
    # Called with parens: @asqav.sign(action_type="x")
    return decorator
