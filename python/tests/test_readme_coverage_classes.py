"""Pin the README capture-coverage table (criterion 581).

The table classifies every shipped adapter, so a new adapter without a
row, a reclassified adapter, or a dropped rule sentence fails here
instead of rotting unseen. CI runs the python job on README edits.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"
PY_EXTRAS = REPO / "python" / "src" / "asqav" / "extras"
TS_EXTRAS = REPO / "typescript" / "src" / "extras"
HARNESS_HOOK = REPO / "python" / "src" / "asqav" / "cli_hook.py"

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
    ("smolagents (python)", "in_path_proxy"),
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


def _shipped_adapters() -> set[str]:
    """Row names for each adapter module on disk, so the table keeps up."""
    names = {
        f"{p.stem} (python)"
        for p in PY_EXTRAS.glob("*.py")
        if not p.stem.startswith("_")
    }
    names |= {
        f"{p.stem} (typescript)"
        for p in TS_EXTRAS.glob("*.ts")
        if not p.stem.startswith("_") and p.stem != "index"
    }
    if HARNESS_HOOK.is_file():
        names.add("cli_hook (harness hook)")
    return names


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
    shipped = _shipped_adapters()
    assert shipped, "no adapter modules found on disk"
    names = {name for name, _ in rows}
    assert names == shipped, (
        f"adapters without a row: {sorted(shipped - names)}, "
        f"rows without an adapter: {sorted(names - shipped)}"
    )
    assert rows == EXPECTED_ROWS, f"table drift: {rows ^ EXPECTED_ROWS}"
    used = {cls for _, cls in rows}
    assert used <= CLASSES
    # model_invoked is the never-built class, so no adapter may use it
    assert used == CLASSES - {"model_invoked"}


def test_capture_coverage_rules_present() -> None:
    text = README.read_text()
    assert "No `model_invoked` surface is ever built" in text
    assert "an MCP-server request is answered" in text
    assert "with the harness hook plus the SDK" in text
    assert "600 seconds for command, HTTP and MCP-tool hooks" in text


_VERSION_RE = re.compile(r"^v?\d+(\.\d+)*([.-][0-9A-Za-z.]+)?$")


def test_capture_coverage_rows_carry_versions() -> None:
    """Every row names the checked framework release (criterion 581 C1)."""
    text = README.read_text()
    rows = _table_rows(text)
    assert len(rows) == 22, f"expected 22 adapter rows, got {len(rows)}"
    header = next(
        line for line in text.splitlines() if line.startswith("| adapter |")
    )
    header_cells = [c.strip() for c in header.strip("|").split("|")]
    expected_header = ["adapter", "coverage class", "citation (URL)", "accessed", "version"]
    assert header_cells == expected_header, header_cells
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 5 or cells[0] in ("adapter", "") or set(cells[0]) <= {"-"}:
            continue
        assert (cells[0], cells[1]) in rows, cells[0]
        assert _VERSION_RE.match(cells[4]), f"{cells[0]}: {cells[4]!r} is no release identifier"
