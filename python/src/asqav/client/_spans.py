"""Tracing spans and their OpenTelemetry export."""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .._useragent import USER_AGENT
from ._agents import get_agent


@dataclass
class Span:
    """A single traced operation.

    Spans track the duration and context of operations. When ended,
    they are signed server-side with ML-DSA for cryptographic proof.
    """

    span_id: str
    name: str
    start_time: float
    attributes: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    end_time: float | None = None
    status: str = "ok"
    signature: str | None = None

        # Add an attribute to the span.
    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

        # Set span status (ok, error).
    def set_status(self, status: str) -> None:
        self.status = status


# Global tracer state
_current_span: Span | None = None


_span_stack: list[Span] = []


_completed_spans: list[Span] = []


_otel_endpoint: str | None = None


@contextmanager
def span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Span, None, None]:
    """Create a traced span with automatic signing.
    Example: ``with asqav.span("api:openai", {"model": "gpt-4"}) as s:``"""
    global _current_span, _span_stack

    span_obj = Span(
        span_id=str(uuid.uuid4()),
        name=name,
        start_time=time.time(),
        attributes=attributes or {},
        parent_id=_current_span.span_id if _current_span else None,
    )

    _span_stack.append(span_obj)
    _current_span = span_obj

    try:
        yield span_obj
        span_obj.status = "ok"
    except Exception as e:
        span_obj.status = "error"
        span_obj.set_attribute("error.message", str(e))
        span_obj.set_attribute("error.type", type(e).__name__)
        raise
    finally:
        span_obj.end_time = time.time()
        _span_stack.pop()
        _current_span = _span_stack[-1] if _span_stack else None

        # Sign the span via API
        try:
            agent = get_agent()
            result = agent.sign(
                action_type=f"span:{name}",
                context={
                    "span_id": span_obj.span_id,
                    "parent_id": span_obj.parent_id,
                    "start_time": span_obj.start_time,
                    "end_time": span_obj.end_time,
                    "duration_ms": (span_obj.end_time - span_obj.start_time) * 1000,
                    "status": span_obj.status,
                    "attributes": span_obj.attributes,
                },
                result=dict(span_obj.attributes),
            )
            span_obj.signature = result.signature
        except Exception:
            pass  # Don't fail user code if signing fails

        # Add to completed spans for OTEL export
        _completed_spans.append(span_obj)


    # Get the active span, if any.
def get_current_span() -> Span | None:
    return _current_span


def configure_otel(endpoint: str | None = None) -> None:
    """Configure OpenTelemetry export.

    Args:
        endpoint: OTEL collector endpoint (e.g., "http://localhost:4318/v1/traces").
                  Set to None to disable export.

    Example:
        asqav.configure_otel("http://localhost:4318/v1/traces")
    """
    global _otel_endpoint
    _otel_endpoint = endpoint


    # Convert a Span to OTEL format.
def span_to_otel(s: Span) -> dict[str, Any]:
    return {
        "traceId": s.span_id.replace("-", "")[:32].ljust(32, "0"),
        "spanId": s.span_id.replace("-", "")[:16],
        "parentSpanId": s.parent_id.replace("-", "")[:16] if s.parent_id else None,
        "name": s.name,
        "kind": 1,  # INTERNAL
        "startTimeUnixNano": int(s.start_time * 1_000_000_000),
        "endTimeUnixNano": int((s.end_time or s.start_time) * 1_000_000_000),
        "attributes": [
            {"key": k, "value": {"stringValue": str(v)}} for k, v in s.attributes.items()
        ]
        + (
            [{"key": "asqav.signature", "value": {"stringValue": s.signature}}]
            if s.signature
            else []
        ),
        "status": {"code": 1 if s.status == "ok" else 2},
    }


def export_spans() -> list[dict[str, Any]]:
    """Export completed spans in OTEL format.

    Returns:
        List of spans in OTEL format.
    """
    global _completed_spans
    spans = [span_to_otel(s) for s in _completed_spans]
    _completed_spans = []
    return spans


    # Flush spans to configured OTEL endpoint.
def flush_spans() -> None:
    global _otel_endpoint, _completed_spans

    if not _otel_endpoint or not _completed_spans:
        return

    from .. import __version__

    spans = export_spans()
    payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "asqav"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "asqav", "version": __version__},
                        "spans": spans,
                    }
                ],
            }
        ]
    }

    try:
        import json
        import urllib.request

        req = urllib.request.Request(
            _otel_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # Don't fail on export errors
