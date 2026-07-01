"""Pluggable detector framework for asqav (criterion 331).

Detectors run inside Agent.sign() after context normalization and before
the HTTP call. If any detector denies the action, sign() raises
DetectorBlockedError and no POST is made.

Usage::

    import asqav
    from asqav._detectors import DetectorPlugin, DetectorResult

    class MyDetector:
        name = "my-detector"

        def inspect(self, action_type: str, context: dict) -> DetectorResult:
            if "ssn" in str(context):
                return DetectorResult(allow=False, labels=["SSN"], reason="PII found")
            return DetectorResult(allow=True)

    asqav.register_detector(MyDetector())

    # Now agent.sign() will call MyDetector.inspect() on every sign call.

Fail-closed semantics: if a detector's inspect() raises an exception, sign()
blocks by default. Pass fail_open=True to register_detector() to allow sign()
to proceed even when that detector errors.

The signed receipt context carries the verdict under the ``_detectors`` key
so the signed bytes prove which detectors fired and why.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    # Python 3.7 fallback (SDK requires 3.8+, so this is belt-and-suspenders).
    from typing_extensions import Protocol, runtime_checkable  # type: ignore[assignment]

logger = logging.getLogger("asqav")

_DETECTOR_DOCS_URL = "https://asqav.com/docs/detector-plugins"


@dataclass
class DetectorResult:
    """Result returned by a DetectorPlugin.inspect() call.

    Fields:
        allow:      True = sign proceeds; False = sign is blocked.
        confidence: Confidence score in [0.0, 1.0]. Informational only.
        labels:     Semantic tags fired (e.g. entity types, policy IDs).
        detector:   Name of the detector that produced this result.
                    Auto-filled by the registry from DetectorPlugin.name.
        reason:     Human-readable explanation for the decision.
    """

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
        """Inspect a pending sign action.

        Args:
            action_type: The action type string passed to Agent.sign().
            context:     The normalized context dict (after schema step).

        Returns:
            DetectorResult with allow=True to permit or allow=False to deny.
        """
        ...


class DetectorBlockedError(Exception):
    """Raised when a detector denies a sign() call.

    Attributes:
        detector_name: Name of the detector that blocked.
        result:        The full DetectorResult from the blocking detector.
        docs_url:      Docs pointer for the feature.
    """

    def __init__(self, message: str, result: DetectorResult) -> None:
        super().__init__(message)
        self.detector_name = result.detector
        self.result = result
        self.docs_url = _DETECTOR_DOCS_URL


# Module-level registry: list of (detector, fail_open) pairs.
_registry: list[tuple[DetectorPlugin, bool]] = []


def register_detector(detector: DetectorPlugin, *, fail_open: bool = False) -> None:
    """Register a detector to run on every Agent.sign() call.

    Args:
        detector:  Any object satisfying the DetectorPlugin protocol.
        fail_open: When True, errors from this detector's inspect() are logged
                   and ignored (sign proceeds). Default is fail-closed: an
                   inspect() exception blocks sign().
    """
    _registry.append((detector, fail_open))


def clear_detectors() -> None:
    """Remove all registered detectors (useful in tests)."""
    _registry.clear()


def run_detectors(
    action_type: str,
    context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Run all registered detectors; raise DetectorBlockedError on first deny.

    Args:
        action_type: Action type forwarded from Agent.sign().
        context:     Normalized context dict (may be None).

    Returns:
        List of result dicts (one per registered detector). Empty when no
        detectors are registered; callers add this to context only when
        non-empty to keep the signed body identical to today's baseline.

    Raises:
        DetectorBlockedError: When any detector returns allow=False or raises
            (and fail_open=False for that detector).
    """
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
                # Record a synthetic error result so the receipt reflects what happened.
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
