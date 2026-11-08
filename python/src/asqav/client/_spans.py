"""Tracing spans and their OpenTelemetry export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
