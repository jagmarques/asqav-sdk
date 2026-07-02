"""Consistency checks for the Verified by Asqav badge section across READMEs."""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATHS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "python" / "README.md",
    REPO_ROOT / "typescript" / "README.md",
]

SECTION_HEADING = "## Verified by Asqav"
PASTE_INSTRUCTION = re.compile(r"Paste (?:your|any) `?\w+_id`? into", re.IGNORECASE)
PLACEHOLDER_TERM = re.compile(r"\b(record_id|signature_id)\b")


def _badge_section(text: str) -> str:
    start = text.index(SECTION_HEADING)
    rest = text[start + len(SECTION_HEADING):]
    next_heading = re.search(r"\n## ", rest)
    end = next_heading.start() if next_heading else len(rest)
    return rest[:end]


@pytest.mark.parametrize("path", README_PATHS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_paste_instruction_appears_at_most_once(path):
    section = _badge_section(path.read_text())
    matches = PASTE_INSTRUCTION.findall(section)
    assert len(matches) <= 1, (
        f"{path.relative_to(REPO_ROOT)}: the paste-into-badge.html instruction "
        f"repeats {len(matches)} times in the badge section, expected at most 1"
    )


def test_placeholder_spelling_is_uniform_across_readmes():
    terms_by_file = {}
    for path in README_PATHS:
        section = _badge_section(path.read_text())
        terms_by_file[path.relative_to(REPO_ROOT)] = set(PLACEHOLDER_TERM.findall(section))

    all_terms = set().union(*terms_by_file.values())
    assert len(all_terms) <= 1, (
        "badge sections use inconsistent placeholder spellings for the verify "
        f"path value: {terms_by_file}, expected a single uniform spelling"
    )
