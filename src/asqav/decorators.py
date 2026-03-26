"""Ergonomic signing patterns for asqav SDK.

Provides the @asqav.sign decorator and asqav.session() context manager
for developers who want concise, Pythonic signing without manual Agent calls.

Example - decorator:
    import asqav

    asqav.init(api_key="sk_...")

    @asqav.sign
    def process_data(data: dict) -> dict:
        return {"processed": True}

    @asqav.sign(action_type="deploy:prod")
    def deploy(env: str) -> None:
        ...

Example - session context manager:
    with asqav.session() as s:
        s.sign("data:read", {"file": "config.json"})
        s.sign("data:write", {"file": "output.json"})
"""

from __future__ import annotations

import functools
import inspect
from contextlib import asynccontextmanager, contextmanager
from collections.abc import AsyncGenerator, Generator
from typing import Any, TypeVar

from .client import Agent, AsqavError, SignatureResponse, SessionResponse, get_agent, _post

F = TypeVar("F")


# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------


class Session:
    """Groups multiple sign calls under a single session.

    Created by the session() context manager. Do not instantiate directly.
    """

    def __init__(self, agent: Agent, session_response: SessionResponse) -> None:
        self._agent = agent
        self._session_response = session_response

    @property
    def session_id(self) -> str:
        return self._session_response.session_id

    def sign(self, action_type: str, context: dict[str, Any] | None = None) -> SignatureResponse:
        """Sign an action within this session.

        Args:
            action_type: Type of action (e.g., "data:read", "api:call").
            context: Additional context for the action.

        Returns:
            SignatureResponse with the signature.
        """
        return self._agent.sign(action_type, context)


@contextmanager
def session() -> Generator[Session, None, None]:
    """Context manager that groups multiple sign calls under a session.

    Usage:
        with asqav.session() as s:
            s.sign("data:read", {"file": "config.json"})
            s.sign("data:write", {"file": "output.json"})

    On enter, starts a session via the API. On exit, ends the session.
    If an exception occurs, signs an error action before ending with error status.
    """
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


@asynccontextmanager
async def async_session() -> AsyncGenerator[Session, None]:
    """Async context manager that groups multiple sign calls under a session.

    Usage:
        async with asqav.async_session() as s:
            s.sign("data:read", {"file": "config.json"})
            s.sign("data:write", {"file": "output.json"})

    Same lifecycle as the sync version - starts session on enter,
    ends on exit, signs errors on exception.
    """
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


# ---------------------------------------------------------------------------
# sign decorator
# ---------------------------------------------------------------------------


def sign(func: Any = None, *, action_type: str | None = None) -> Any:
    """Decorator to auto-sign function execution.

    Works on both sync and async functions (auto-detects coroutines).
    Can be used with or without parentheses:

        @asqav.sign
        def my_func(): ...

        @asqav.sign(action_type="deploy:prod")
        def my_func(): ...

    Args:
        func: The function to decorate (when used without parens).
        action_type: Override the default "function:call" action type.

    Returns:
        Decorated function that signs calls via asqav.
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
