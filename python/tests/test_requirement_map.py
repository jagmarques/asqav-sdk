"""The vector-to-requirement mapping is current, derived, and publishes its gaps.

A mapping that drifts from the corpus is worse than none, because it reads as a
coverage claim while describing a corpus that no longer exists. And a mapping
without its unmapped half can be made to look complete by growing the corpus,
which is the failure mode the second half exists to prevent.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAP = ROOT / "verifier" / "conformance-vectors" / "requirement-map.json"
GENERATOR = ROOT / "verifier" / "build_requirement_map.py"


@pytest.fixture(scope="module")
def document():
    assert MAP.exists(), f"{MAP} is not published"
    return json.loads(MAP.read_text())


def test_regenerating_reproduces_the_published_map(document, tmp_path):
    """The committed map is what the generator produces from today's corpus."""
    before = MAP.read_text()
    result = subprocess.run(
        [sys.executable, str(GENERATOR)], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, result.stderr
    after = MAP.read_text()
    if after != before:
        MAP.write_text(before)
        pytest.fail(
            "requirement-map.json is stale: the corpus moved without the map being "
            "regenerated. Run python3 verifier/build_requirement_map.py and commit."
        )


def test_the_unmapped_list_is_published_even_when_it_is_empty(document):
    """The key must exist; an absent key would read as 'no gaps' by omission."""
    assert "unmapped_requirements" in document
    assert isinstance(document["unmapped_requirements"], list)
    assert "unmapped_note" in document


def test_every_unmapped_requirement_says_why(document):
    """A gap without a reason is not actionable, so it does not count as declared."""
    for req in document["unmapped_requirements"]:
        detail = document["unmapped_detail"][req]
        assert detail["axis"]
        assert detail["results_seen_across_vectors"], req
        # Whatever the reason, no vector may have exercised it, or it is not unmapped.
        assert "PASS" not in detail["results_seen_across_vectors"], req
        assert "FAIL" not in detail["results_seen_across_vectors"], req


def test_coverage_and_unmapped_partition_the_requirement_set(document):
    """No requirement may be silently absent from both halves."""
    requirements = set(document["requirements"])
    covered = {r for r, hits in document["coverage"].items() if hits}
    unmapped = set(document["unmapped_requirements"])
    assert covered | unmapped == requirements
    assert not (covered & unmapped)


def test_interop_fixtures_are_not_counted_as_profile_coverage(document):
    """Another specification's fixture is not evidence about this profile."""
    for name, entry in document["interop_fixtures"].items():
        assert entry["exercises_profile_requirements"] is False, name
    for req, hits in document["coverage"].items():
        for vector in hits:
            assert vector in document["vectors"], f"{req} credits a non-asqav vector {vector}"


def test_seq_chain_vector_chain_axis_is_pass_in_the_committed_map(document):
    """asqav-17's chain link rederives, and the published axis evidence says so.

    The builder hands run_structured the predecessor's payload, not its
    envelope; a chain FAIL in this entry would mean the map derived REQ-CHAIN
    coverage from a false chain break over the envelope bytes.
    """
    axes = document["axis_results"]["asqav-17-seq-contiguous"]
    assert axes["chain"] == "PASS"
