# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""The generated vector index in the corpus README (criterion 821).

`verifier/render_vector_list.py` rewrites the list between the marker comments in
`verifier/conformance-vectors/README.md` from `manifest.json`. The README is the
only rendered copy of that index, so a hand edit, a stale render or a separator
that drifts from the renderer's own output must fail here rather than reach the
corpus lock as an unexplained re-cut.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[2] / "verifier"
CORPUS = VERIFIER / "conformance-vectors"
README = CORPUS / "README.md"

#: The em dash, written as an escape so the guard does not carry the character it forbids.
EM_DASH = "\u2014"


def _renderer():
    spec = importlib.util.spec_from_file_location(
        "render_vector_list", VERIFIER / "render_vector_list.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest() -> list[dict]:
    return json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))


def _readme_list() -> list[str]:
    lines = README.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("<!-- vectors-list:start"))
    end = next(i for i, line in enumerate(lines) if line.startswith("<!-- vectors-list:end"))
    assert start < end, "the vectors-list markers are in the wrong order"
    return lines[start + 1 : end]


def _expected_lines() -> list[str]:
    """The index the README carries, built from manifest.json alone.

    The separator is written out here on purpose: the corpus README's index
    format is a published contract, so a test that only re-ran the renderer
    would agree with a renderer that drifts.
    """
    lines = []
    for entry in _manifest():
        outcome = entry["outcome"]
        if entry.get("reason_code"):
            outcome = f"{outcome} ({entry['reason_code']})"
        lines.append(f"- `{entry['dir']}`: {outcome}")
    return lines


def test_manifest_dirs_keep_the_code_span_well_formed() -> None:
    for entry in _manifest():
        directory = entry["dir"]
        assert "`" not in directory, directory
        assert ":" not in directory, directory


def test_renderer_emits_the_published_index_format() -> None:
    assert _renderer()._render(_manifest()) == _expected_lines()


def test_readme_index_is_the_current_render_of_the_manifest() -> None:
    assert _readme_list() == _expected_lines()


def test_index_lines_carry_one_well_formed_code_span() -> None:
    directories = [entry["dir"] for entry in _manifest()]
    for line, directory in zip(_readme_list(), directories, strict=True):
        assert line.count("`") == 2, line
        assert line.startswith(f"- `{directory}`: "), line


def test_readme_carries_no_em_dash() -> None:
    text = README.read_text(encoding="utf-8")
    assert EM_DASH not in text


def test_render_twice_leaves_the_readme_unchanged(tmp_path: Path) -> None:
    renderer = _renderer()
    target = tmp_path / "README.md"
    target.write_text(README.read_text(encoding="utf-8"), encoding="utf-8")
    renderer.README = target
    assert renderer.main() == 0
    first = target.read_bytes()
    assert renderer.main() == 0
    assert target.read_bytes() == first
