"""Retry logic with exponential backoff for asqav SDK.

Provides decorators for automatic retry of transient API errors
with exponential backoff and jitter. Works with both sync and async functions.

Retries on:
    - RateLimitError (429)
    - APIError with 5xx status codes
    - ConnectionError
    - TimeoutError

Does NOT retry:
    - AuthenticationError (401/403)
    - APIError with 4xx status codes (except 429)
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
    """Determine if an exception is retryable.

    Uses lazy imports to avoid circular dependency with client.py.

    Returns True for:
        - RateLimitError
        - APIError with 5xx status code
        - ConnectionError
        - TimeoutError

    Returns False for:
        - AuthenticationError
        - APIError with 4xx status code (except 429, which is RateLimitError)
        - Any other exception
    """
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
    """Calculate delay for a given retry attempt using exponential backoff.

    Args:
        attempt: Zero-based attempt number.
        base_delay: Base delay in seconds.
        max_delay: Maximum delay cap in seconds.
        jitter: Whether to add random jitter.

    Returns:
        Delay in seconds.
    """
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
    """Decorator for retrying sync functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default 3).
        base_delay: Initial delay in seconds (default 0.5).
        max_delay: Maximum delay cap in seconds (default 30.0).
        jitter: Add random jitter to delay (default True).

    Returns:
        Decorated function with retry logic.

    Example:
        @with_retry(max_retries=5, base_delay=1.0)
        def call_api():
            return _post("/endpoint", data)
    """

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
    """Decorator for retrying async functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default 3).
        base_delay: Initial delay in seconds (default 0.5).
        max_delay: Maximum delay cap in seconds (default 30.0).
        jitter: Add random jitter to delay (default True).

    Returns:
        Decorated async function with retry logic.

    Example:
        @with_async_retry(max_retries=5, base_delay=1.0)
        async def call_api():
            return await _async_post("/endpoint", data)
    """

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
