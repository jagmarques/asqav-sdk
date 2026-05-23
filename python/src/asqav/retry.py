"""Retry decorators with exponential backoff + jitter for sync and async functions.

Retries 429, 5xx, ``ConnectionError``, and ``TimeoutError``; never retries
4xx except 429 or :class:`AuthenticationError`.
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _is_retryable(exc: Exception) -> bool:
    """Return True for retryable exceptions: rate-limit, 5xx, connection, timeout."""
    from .client import APIError, AuthenticationError, RateLimitError

    if isinstance(exc, AuthenticationError):
        return False

    if isinstance(exc, RateLimitError):
        return True

    if isinstance(exc, APIError):
        if exc.status_code is not None and exc.status_code >= 500:
            return True
        return False

    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True

    return False


def _calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    jitter: bool,
) -> float:
    """Exponential backoff delay in seconds for ``attempt`` (zero-based)."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter:
        delay = random.uniform(0, delay)
    return delay


def with_retry(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    jitter: bool = True,
) -> Callable[[F], F]:
    """Decorator that retries sync functions with exponential backoff + optional jitter."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if not _is_retryable(exc) or attempt == max_retries:
                        raise
                    delay = _calculate_delay(attempt, base_delay, max_delay, jitter)
                    time.sleep(delay)
            raise last_exc  # pragma: no cover

        return wrapper  # type: ignore[return-value]

    return decorator


def with_async_retry(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    jitter: bool = True,
) -> Callable[[F], F]:
    """Decorator that retries async functions with exponential backoff + optional jitter."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if not _is_retryable(exc) or attempt == max_retries:
                        raise
                    delay = _calculate_delay(attempt, base_delay, max_delay, jitter)
                    await asyncio.sleep(delay)
            raise last_exc  # pragma: no cover

        return wrapper  # type: ignore[return-value]

    return decorator
