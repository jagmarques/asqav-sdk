"""Classification lives at the review boundary, not in verify() (488 C38). `record`/`review` in `__main__.py`
classify live input against a retained original; `retain.reassess_and_classify` is the retention caller:
it raises unless an original exists, so both comparanda are real, and it imports only `disagreement`. The
layer's other parts - adapters, dispatch, corpus runner, the store - hold no retained original and stay
classification-free. TS has no oracle CLI, so the guard covers the Python oracle only (DECISION-C2-TS).
"""
from __future__ import annotations

import re
from pathlib import Path

_ORACLE = Path(__file__).resolve().parent.parent / "src" / "asqav" / "verifier" / "oracle"

# These tests read the sources; they change no product bytes.

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
