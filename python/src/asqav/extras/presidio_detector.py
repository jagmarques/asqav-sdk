"""PresidioDetector: wraps Microsoft Presidio Analyzer for PII detection.

Install: ``pip install asqav[presidio]``

Usage::

    from asqav.extras.presidio_detector import PresidioDetector
    import asqav

    asqav.register_detector(PresidioDetector())

The detector stringifies the context dict and scans it with Presidio's
AnalyzerEngine. Any high-confidence PII entities (>= threshold) trigger
an allow=False result. Entity types become labels on the result.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .._detectors import DetectorResult

logger = logging.getLogger("asqav")

_PRESIDIO_DOCS = "https://microsoft.github.io/presidio/"


def _load_analyzer() -> Any:
    """Lazy-load presidio_analyzer.AnalyzerEngine; raise ImportError if missing."""
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import]

        return AnalyzerEngine
    except ImportError as exc:
        raise ImportError(
            "PresidioDetector requires presidio-analyzer. "
            "Install with: pip install asqav[presidio]"
        ) from exc


class PresidioDetector:
    """DetectorPlugin that blocks sign() when Presidio finds high-confidence PII.

    Args:
        language:   Language code for Presidio analysis. Default ``"en"``.
        threshold:  Confidence threshold [0.0, 1.0]. Entities at or above this
                    score cause allow=False. Default ``0.5``.
        entities:   Optional list of entity types to scan for. ``None`` scans
                    all entities Presidio supports.
    """

    name = "presidio"

    def __init__(
        self,
        *,
        language: str = "en",
        threshold: float = 0.5,
        entities: list[str] | None = None,
    ) -> None:
        self._language = language
        self._threshold = threshold
        self._entities = entities
        # Lazy; constructed on first use so import errors surface at call time.
        self._engine: Any = None

    def _get_engine(self) -> Any:
        if self._engine is None:
            AnalyzerEngine = _load_analyzer()
            self._engine = AnalyzerEngine()
        return self._engine

    def inspect(self, action_type: str, context: dict[str, Any]) -> DetectorResult:
        """Scan the stringified context for PII.

        Returns allow=True when no entities exceed the threshold, allow=False
        otherwise. Labels carry the entity type strings.
        """
        text = json.dumps(context, default=str)
        engine = self._get_engine()

        kwargs: dict[str, Any] = {"text": text, "language": self._language}
        if self._entities:
            kwargs["entities"] = self._entities

        results = engine.analyze(**kwargs)

        hits = [r for r in results if r.score >= self._threshold]
        if not hits:
            return DetectorResult(allow=True, detector=self.name)

        labels = sorted({r.entity_type for r in hits})
        max_score = max(r.score for r in hits)
        return DetectorResult(
            allow=False,
            confidence=max_score,
            labels=labels,
            detector=self.name,
            reason=(
                f"PII detected: {', '.join(labels)} "
                f"(max confidence {max_score:.2f}, threshold {self._threshold})"
            ),
        )
