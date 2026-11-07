"""Shared type variable, logger and value helpers for the client."""

from __future__ import annotations

import hashlib
import json as _json
import logging
import uuid
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


logger = logging.getLogger("asqav")


_OWASP_AGENTIC_TOP10_IDS = frozenset(f"ASI{number:02d}" for number in range(1, 11))


    # Hash an arbitrary value with SHA-256 for output verification.
def _hash_value(value: Any) -> str:
    if isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, bytes):
        raw = value
    else:
        raw = _json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


    # Parse a timestamp from API response (ISO string or float).
def _parse_timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        from datetime import datetime

        # Parse ISO format datetime string
        try:
            # Handle both with and without timezone
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            dt = datetime.fromisoformat(value)
            return dt.timestamp()
        except ValueError:
            return 0.0
    return 0.0


    # Generate a unique trace ID for correlating actions across a workflow.
def generate_trace_id() -> str:
    return uuid.uuid4().hex
