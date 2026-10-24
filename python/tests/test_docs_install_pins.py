"""Integration docs name the current release, not a frozen version (646)."""

from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"


def test_github_actions_guide_names_the_release_history_not_a_frozen_pin() -> None:
    text = (DOCS / "github-actions.md").read_text(encoding="utf-8")
    assert "asqav==0.4.4" not in text, "the guide pins a release that predates the current SDK"
    assert "asqav==<version>" in text
    assert "pypi.org/project/asqav/#history" in text


def test_no_integration_guide_pins_an_asqav_release() -> None:
    """A literal release pin in prose ages into a stale claim about the SDK."""
    import re

    frozen = re.compile(r"asqav==\d+\.\d+\.\d+")
    for path in DOCS.rglob("*.md"):
        assert not frozen.search(path.read_text(encoding="utf-8")), path
