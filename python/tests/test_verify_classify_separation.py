"""Classification lives at the review boundary, not in verify() (488 C38)."""
from __future__ import annotations

import re
from pathlib import Path

_ORACLE = Path(__file__).resolve().parent.parent / "src" / "asqav" / "verifier" / "oracle"

#: Files that verify or store but must never classify (488 C38).
_PURE_FILES = [
    "adapter.py",
    "core.py",
    "runner.py",
    "assessment_store.py",
    *(f"adapters/{p.name}" for p in sorted((_ORACLE / "adapters").glob("*.py"))),
]

#: Classification references: entry names plus the import and call shapes a
#: bare `from .disagreement import classify` wiring takes (CRITIC-C38).
_CLASSIFY_PATTERNS = (
    "classify_and_attribute",
    "attribute_live_split",
    "compare_live",
    "live_compare",
    "spec_case",
    "conflict_kind",
    "attributed_to_trust_time",
    r"from\s+[\.\w]*disagreement\b",
    r"from\s+[\.\w]*live_compare\b",
    r"import\s+[\.\w]*disagreement\b",
    r"import\s+[\.\w]*live_compare\b",
    r"disagreement\.",
    r"live_compare\.",
    r"\bclassify\(",
    r"\battribute_split\(",
)


def _hits(text: str) -> list[str]:
    found = [p for p in _CLASSIFY_PATTERNS if re.search(p, text)]
    return found


def test_verification_layer_has_no_classification_caller() -> None:
    for rel in _PURE_FILES:
        text = (_ORACLE / rel).read_text()
        assert _hits(text) == [], f"{rel} references classification"


def test_separation_probe_fires_on_the_classification_callers() -> None:
    assert _hits((_ORACLE / "__main__.py").read_text()) != []
    assert _hits((_ORACLE / "retain.py").read_text()) != []
