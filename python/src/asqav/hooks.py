"""Generic hooks/middleware system for asqav signing.

Register before/after callbacks that fire around every ``Agent.sign(...)`` call.
Before-hooks may mutate the context dict; after-hooks observe the response.
Both are fail-open: if a registered hook raises, it is logged and signing
continues unaffected.

Pattern matching:
    - ``"*"`` matches every action_type
    - ``"strands:*"`` glob-matches any action_type starting with ``strands:``
    - A compiled ``re.Pattern`` is matched with ``pattern.search(action_type)``
    - Otherwise the pattern must equal the action_type exactly

Usage::

    import asqav

    def add_request_id(action_type, context):
        context["request_id"] = "req_abc"
        return context

    asqav.register_before("*", add_request_id)
    asqav.register_after("strands:*", lambda resp: print(resp.signature_id))
"""

from __future__ import annotations

import fnmatch
import logging
import re
import threading
from typing import Any, Callable, Pattern, Union

logger = logging.getLogger("asqav")

PatternType = Union[str, Pattern[str]]
BeforeHook = Callable[[str, dict], Union[dict, None]]
AfterHook = Callable[[Any], None]

_lock = threading.Lock()
_before_hooks: list[tuple[PatternType, BeforeHook]] = []
_after_hooks: list[tuple[PatternType, AfterHook]] = []


def _matches(pattern: PatternType, action_type: str) -> bool:
    """Return True if ``pattern`` matches ``action_type``."""
    if isinstance(pattern, re.Pattern):
        return pattern.search(action_type) is not None
    if pattern == "*":
        return True
    if "*" in pattern or "?" in pattern or "[" in pattern:
        return fnmatch.fnmatchcase(action_type, pattern)
    return pattern == action_type


def register_before(pattern: PatternType, fn: BeforeHook) -> None:
    """Register ``fn`` to run before signing actions matching ``pattern``.

    The hook is called as ``fn(action_type, context)`` and may return a
    modified context dict; if it returns ``None`` the existing context is
    kept. Hooks are invoked in registration order.
    """
    with _lock:
        _before_hooks.append((pattern, fn))


def register_after(pattern: PatternType, fn: AfterHook) -> None:
    """Register ``fn`` to run after signing actions matching ``pattern``.

    The hook is called as ``fn(response)`` where ``response`` is a
    ``SignatureResponse``. After-hooks are observers; their return value is
    ignored.
    """
    with _lock:
        _after_hooks.append((pattern, fn))


def clear_hooks() -> None:
    """Remove every registered before/after hook. Intended for tests."""
    with _lock:
        _before_hooks.clear()
        _after_hooks.clear()


def _dispatch_before(action_type: str, context: dict) -> dict:
    """Invoke matching before-hooks. Returns the (possibly mutated) context."""
    with _lock:
        snapshot = list(_before_hooks)
    current = context if context is not None else {}
    for pattern, fn in snapshot:
        try:
            if not _matches(pattern, action_type):
                continue
            result = fn(action_type, current)
            if isinstance(result, dict):
                current = result
        except Exception as exc:
            logger.warning(
                "asqav before-hook %r failed (fail-open): %s", fn, exc
            )
    return current


def _dispatch_after(action_type: str, response: Any) -> None:
    """Invoke matching after-hooks. Failures are logged and swallowed."""
    with _lock:
        snapshot = list(_after_hooks)
    for pattern, fn in snapshot:
        try:
            if not _matches(pattern, action_type):
                continue
            fn(response)
        except Exception as exc:
            logger.warning(
                "asqav after-hook %r failed (fail-open): %s", fn, exc
            )
