# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""The corpus requirement map must name the corpus it was built from.

`verifier/build_requirement_map.py` writes `requirement-map.json` from the
corpus, and the corpus lock pins the map's bytes, so the map cannot drift
silently. A vector added to the corpus without regenerating the map leaves the
map short, and the lock check stays green because the map's own bytes did not
change. This module compares the map with the corpus and names what is
missing, so the corpus job can say which vector the map forgot.

The generator's own check, in `test_requirement_map.py`, rebuilds the map and
compares it byte for byte. This one reads the corpus and the map only, so it
is cheap enough for the job that runs on every corpus change.
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = _REPO_ROOT / "verifier" / "conformance-vectors"
REQUIREMENT_MAP = CORPUS / "requirement-map.json"

#: The format the builder maps against the profile; every other format is an
#: interoperability fixture, the rule build_requirement_map.py applies.
NATIVE_FORMAT = "asqav-native"


def _corpus_dirs() -> tuple[set[str], set[str]]:
    """The native vectors and the interoperability fixtures, read from the corpus."""
    native: set[str] = set()
    interop: set[str] = set()
    for directory in sorted(p for p in CORPUS.iterdir() if p.is_dir()):
        expected = directory / "expected.json"
        if not expected.exists():
            continue
        fmt = json.loads(expected.read_text(encoding="utf-8")).get("format", "unknown")
        (native if fmt == NATIVE_FORMAT else interop).add(directory.name)
    return native, interop


def test_the_map_names_every_native_vector() -> None:
    native, interop = _corpus_dirs()
    document = json.loads(REQUIREMENT_MAP.read_text(encoding="utf-8"))
    mapped = set(document["vectors"])
    assert mapped == native, (
        f"requirement-map.json does not name {sorted(native - mapped)} and names "
        f"{sorted(mapped - native)}; regenerate with verifier/build_requirement_map.py"
    )


def test_the_map_names_every_interoperability_fixture() -> None:
    native, interop = _corpus_dirs()
    document = json.loads(REQUIREMENT_MAP.read_text(encoding="utf-8"))
    fixtures = set(document["interop_fixtures"])
    assert fixtures == interop, (
        f"requirement-map.json does not name the fixture {sorted(interop - fixtures)} and "
        f"names {sorted(fixtures - interop)}; regenerate with "
        "verifier/build_requirement_map.py"
    )


def test_a_short_requirement_map_is_caught() -> None:
    """The negative control: a map that names one vector fewer must be named."""
    native, _interop = _corpus_dirs()
    document = json.loads(REQUIREMENT_MAP.read_text(encoding="utf-8"))
    dropped = sorted(document["vectors"])[0]
    del document["vectors"][dropped]
    assert sorted(native - set(document["vectors"])) == [dropped]
