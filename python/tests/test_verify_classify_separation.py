"""Verification never classifies: layering invariant (488 C38).

Classification (`classify_and_attribute` and its helpers) lives at the
review boundary, where a retained original exists (`record`/`review` in
`__main__.py`). The verification layer — format adapters, dispatch,
corpus runner and retention storage — stays classification-free: it
holds no retained original to classify against, and wiring one in
would cycle the imports (`live_compare` already imports `adapter` and
the package root that re-exports `core.verify`). Tests read the
sources; they change no product bytes.
"""
from __future__ import annotations

from pathlib import Path

_ORACLE = Path(__file__).resolve().parent.parent / "src" / "asqav" / "verifier" / "oracle"

#: Files that verify or retain but must never classify (488 C38).
_PURE_FILES = [
    "adapter.py",
    "core.py",
    "runner.py",
    "retain.py",
    "assessment_store.py",
    *(f"adapters/{p.name}" for p in sorted((_ORACLE / "adapters").glob("*.py"))),
]

#: Classification entry points that must not appear above.
_CLASSIFY_NAMES = (
    "classify_and_attribute",
    "attribute_live_split",
    "compare_live",
    "live_compare",
    "spec_case",
    "conflict_kind",
    "attributed_to_trust_time",
)


def test_verification_layer_has_no_classification_caller() -> None:
    for rel in _PURE_FILES:
        text = (_ORACLE / rel).read_text()
        hits = [name for name in _CLASSIFY_NAMES if name in text]
        assert hits == [], f"{rel} references classification: {hits}"


def test_separation_probe_fires_on_the_review_boundary() -> None:
    text = (_ORACLE / "__main__.py").read_text()
    assert "classify_and_attribute" in text
    assert "live_compare" in text
