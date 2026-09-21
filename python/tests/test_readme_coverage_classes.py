"""Pin the README capture-coverage table (criterion 581).

The table classifies every shipped adapter, so a new adapter without a
row, a reclassified adapter, or a dropped rule sentence fails here
instead of rotting unseen. CI runs the python job on README edits.
"""
from __future__ import annotations

from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"

CLASSES = {
    "harness_enforced",
    "framework_callback",
    "in_path_proxy",
    "model_invoked",
    "passive",
}

EXPECTED_ROWS = {
    ("openai (python)", "in_path_proxy"),
    ("anthropic (python)", "in_path_proxy"),
    ("letta (python)", "in_path_proxy"),
    ("haystack (python)", "in_path_proxy"),
    ("smolagents (python)", "model_invoked"),
    ("langchain (python)", "framework_callback"),
    ("litellm (python)", "framework_callback"),
    ("llamaindex (python)", "framework_callback"),
    ("dspy (python)", "framework_callback"),
    ("instructor (python)", "framework_callback"),
    ("crewai (python)", "framework_callback"),
    ("openai_agents (python)", "framework_callback"),
    ("strands (python)", "framework_callback"),
    ("opa_detector (python)", "passive"),
    ("presidio_detector (python)", "passive"),
    ("cli_hook (harness hook)", "harness_enforced"),
    ("langchain (typescript)", "framework_callback"),
    ("openai-agents (typescript)", "framework_callback"),
    ("mastra (typescript)", "framework_callback"),
    ("vercel-ai (typescript)", "framework_callback"),
    ("opa (typescript)", "passive"),
    ("presidio (typescript)", "passive"),
}


def _table_rows(text: str) -> set[tuple[str, str]]:
    rows: set[tuple[str, str]] = set()
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4 or cells[0] in ("adapter", "") or set(cells[0]) <= {"-"}:
            continue
        rows.add((cells[0], cells[1]))
    return rows


def test_capture_coverage_table_matches_shipped_adapters() -> None:
    text = README.read_text()
    assert "## Capture coverage" in text
    rows = _table_rows(text)
    assert rows == EXPECTED_ROWS, f"table drift: {rows ^ EXPECTED_ROWS}"
    assert {cls for _, cls in rows} <= CLASSES
    assert {cls for _, cls in rows} == CLASSES


def test_capture_coverage_rules_present() -> None:
    text = README.read_text()
    assert "No `model_invoked` surface is ever built" in text
    assert "an MCP-server request is answered" in text
    assert "with the harness hook plus the SDK" in text
    assert "600 seconds for command, HTTP and MCP-tool hooks" in text
