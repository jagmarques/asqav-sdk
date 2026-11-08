"""Tracing spans and their OpenTelemetry export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._useragent import USER_AGENT


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


_completed_spans: list[Span] = []


def export_spans() -> list[dict[str, Any]]:
    """Export completed spans in OTEL format.

    Returns:
        List of spans in OTEL format.
    """
    spans = [span_to_otel(s) for s in _completed_spans]
    _completed_spans.clear()
    return spans


_otel_endpoint: str | None = None


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
