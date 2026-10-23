"""Pluggable detector framework, run in Agent.sign() before the HTTP call.
A denying detector raises DetectorBlockedError; an erroring one blocks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    # Python 3.7 fallback (SDK requires 3.8+, so this is belt-and-suspenders).
    from typing_extensions import (  # type: ignore[assignment]
        Protocol,
        runtime_checkable,
    )

logger = logging.getLogger("asqav")

_DETECTOR_DOCS_URL = "https://asqav.com/docs/detector-plugins"


@dataclass
class DetectorResult:
    """Result returned by a DetectorPlugin.inspect() call.
    Fields: allow, confidence in [0, 1], labels, detector (auto), reason."""

    allow: bool
    confidence: float = 1.0
    labels: list[str] = field(default_factory=list)
    detector: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "allow": self.allow,
            "confidence": self.confidence,
            "labels": self.labels,
            "reason": self.reason,
        }


@runtime_checkable
class DetectorPlugin(Protocol):
    """Protocol every detector must satisfy.

    Implement name (str) and inspect(); register via asqav.register_detector().
    """

    name: str

    def inspect(self, action_type: str, context: dict[str, Any]) -> DetectorResult:
        """Inspect a pending sign action; return a DetectorResult.
        ``action_type`` and normalized ``context`` come from Agent.sign()."""
        ...


class DetectorBlockedError(Exception):
    """Raised when a detector denies a sign() call.
    Carries ``detector_name``, the blocking ``result`` and ``docs_url``."""

    def __init__(self, message: str, result: DetectorResult) -> None:
        super().__init__(message)
        self.detector_name = result.detector
        self.result = result
        self.docs_url = _DETECTOR_DOCS_URL


# Module-level registry: list of (detector, fail_open) pairs.
_registry: list[tuple[DetectorPlugin, bool]] = []


def register_detector(detector: DetectorPlugin, *, fail_open: bool = False) -> None:
    """Register a detector to run on every Agent.sign() call.
    ``fail_open=True`` ignores its errors; the default fails closed."""
    _registry.append((detector, fail_open))


    # Remove all registered detectors (useful in tests).
def clear_detectors() -> None:
    _registry.clear()


def run_detectors(
    action_type: str,
    context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Run all registered detectors; raise DetectorBlockedError on first deny.
    Returns one result dict per detector (none if empty)."""
    if not _registry:
        return []

    ctx = context or {}
    records: list[dict[str, Any]] = []

    for detector, fail_open in _registry:
        try:
            result = detector.inspect(action_type, ctx)
        except Exception as exc:
            if fail_open:
                logger.warning(
                    "asqav detector %r raised (fail-open): %s",
                    getattr(detector, "name", repr(detector)),
                    exc,
                    exc_info=True,
                )
                # Record a synthetic error result so the receipt shows it.
                records.append(
                    DetectorResult(
                        allow=True,
                        confidence=0.0,
                        labels=["error"],
                        detector=getattr(detector, "name", "unknown"),
                        reason=f"inspector raised (fail-open): {exc}",
                    ).as_dict()
                )
                continue
            # Fail-closed: block sign().
            block_result = DetectorResult(
                allow=False,
                confidence=0.0,
                labels=["error"],
                detector=getattr(detector, "name", "unknown"),
                reason=f"inspector raised (fail-closed): {exc}",
            )
            raise DetectorBlockedError(
                f"detector_error_blocked: detector '{block_result.detector}' raised "
                f"during inspect() and fail_open=False: sign() blocked. "
                f"Error: {exc}",
                block_result,
            ) from exc

        # Back-fill detector name from the plugin if the result left it blank.
        if not result.detector:
            result.detector = getattr(detector, "name", "unknown")

        records.append(result.as_dict())

        if not result.allow:
            raise DetectorBlockedError(
                f"detector_blocked: detector '{result.detector}' denied the sign() "
                f"call. reason='{result.reason}' labels={result.labels}",
                result,
            )

    return records
