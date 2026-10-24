# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""The independent-runs registry is honest: every entry pins real history.

``verifier/independent-runs.json`` records outside recomputations of the corpus.
An entry that names a commit this repository never had, or a vector the corpus
did not carry at that commit, is not evidence - it is marketing. These tests
refuse both, so the file stays an append-only log of runs that actually
happened against bytes that actually existed.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "verifier" / "independent-runs.json"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

#: Every run entry carries these members with these types.
_REQUIRED_MEMBERS = {
    "id": str,
    "date": str,
    "runner": dict,
    "asqav_sdk_commit": str,
    "corpus": str,
    "vectors": list,
    "algorithms": list,
    "uses_asqav_code": bool,
    "independent_checks": list,
    "result": str,
    "not_proved": list,
    "non_claims_text": str,
}
_RUNNER_MEMBERS = {
    "repository": str,
    "evidence": str,
    "merge_commit": str,
    "head_commit": str,
}


def _document() -> dict:
    return json.loads(REGISTRY.read_text())


def _runs() -> list[dict]:
    return _document()["runs"]


def _entry_ids() -> list[str]:
    return [entry["id"] for entry in _runs()]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
    )


def test_document_shape_is_the_published_one() -> None:
    doc = _document()
    assert doc["version"] == 1
    assert isinstance(doc["purpose"], str) and doc["purpose"]
    assert isinstance(doc["rules"], list) and all(isinstance(r, str) for r in doc["rules"])
    assert isinstance(doc["runs"], list) and doc["runs"], "the registry has no entries"


@pytest.mark.parametrize("entry", _runs(), ids=_entry_ids())
def test_entry_members_have_the_required_types(entry: dict) -> None:
    missing = sorted(set(_REQUIRED_MEMBERS) - set(entry))
    assert not missing, f"{entry.get('id')}: missing members {missing}"
    for member, kind in _REQUIRED_MEMBERS.items():
        assert isinstance(entry[member], kind), (
            f"{entry['id']}: {member} must be {kind.__name__}, "
            f"got {type(entry[member]).__name__}"
        )
    for member, kind in _RUNNER_MEMBERS.items():
        assert isinstance(entry["runner"].get(member), kind), (
            f"{entry['id']}: runner.{member} must be {kind.__name__}"
        )
    assert all(isinstance(v, str) for v in entry["vectors"])
    assert all(isinstance(c, str) for c in entry["independent_checks"])
    assert all(isinstance(n, str) for n in entry["not_proved"])


@pytest.mark.parametrize("entry", _runs(), ids=_entry_ids())
def test_pinned_commit_is_40_hex_and_exists_in_this_repository(entry: dict) -> None:
    sha = entry["asqav_sdk_commit"]
    assert _COMMIT_RE.match(sha), (
        f"{entry['id']}: asqav_sdk_commit {sha!r} is not 40 lowercase hex"
    )
    probe = _git("cat-file", "-e", f"{sha}^{{commit}}")
    assert probe.returncode == 0, (
        f"{entry['id']}: pinned commit {sha} does not exist in this repository"
    )


@pytest.mark.parametrize("entry", _runs(), ids=_entry_ids())
def test_vectors_exist_in_the_corpus_at_the_pinned_commit(entry: dict) -> None:
    sha = entry["asqav_sdk_commit"]
    corpus = entry["corpus"]
    listing = _git("ls-tree", "--name-only", "-d", sha, f"{corpus}/")
    assert listing.returncode == 0, (
        f"{entry['id']}: git ls-tree failed for {corpus} at {sha}: {listing.stderr}"
    )
    present = {line.rsplit("/", 1)[-1] for line in listing.stdout.splitlines()}
    for vector in entry["vectors"]:
        assert vector in present, (
            f"{entry['id']}: vector {vector!r} is not a directory under {corpus} "
            f"at pinned commit {sha}"
        )


@pytest.mark.parametrize("entry", _runs(), ids=_entry_ids())
def test_date_is_iso_and_the_code_flag_is_a_bool(entry: dict) -> None:
    try:
        date.fromisoformat(entry["date"])
    except ValueError:
        pytest.fail(f"{entry['id']}: date {entry['date']!r} is not an ISO date")
    # bool must not slip in as 0/1: isinstance(True, int) is True, so check exact.
    assert type(entry["uses_asqav_code"]) is bool


def test_entry_ids_are_unique() -> None:
    ids = _entry_ids()
    assert len(ids) == len(set(ids)), f"duplicate entry ids: {sorted(ids)}"
