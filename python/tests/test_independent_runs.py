# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Each independent-runs entry pins real history, or a tagged copy of it.
A rewritten commit needs its tag, an equal corpus and a map line."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "verifier" / "independent-runs.json"
REWRITE_MAP = REPO_ROOT / "verifier" / "commit-rewrite-map.json"

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


PRESERVED_TAG = "refs/tags/pre-rewrite/{}"


def _mapping_problems(old: str, new: str, entries: list[dict]) -> list[str]:
    """Why a rewrite-map line cannot stand for a commit; empty when it can."""
    if not (_COMMIT_RE.match(old) and _COMMIT_RE.match(new)) or old == new:
        return [f"{old} -> {new} is not two different 40-hex commits"]
    named = [e for e in entries if e["asqav_sdk_commit"] == old]
    if not named:
        return [f"no registry entry names {old}"]
    if _git("cat-file", "-e", f"{new}^{{commit}}").returncode != 0:
        return [f"replacement {new} is not in this repository"]
    kept = _git("rev-parse", "--verify", PRESERVED_TAG.format(old) + "^{commit}")
    if kept.stdout.strip() != old:
        return [f"{old} is not preserved under {PRESERVED_TAG.format(old)}"]
    out = []
    for e in named:
        trees = [_git("rev-parse", f"{c}:{e['corpus']}").stdout.strip() for c in (old, new)]
        if not trees[0] or trees[0] != trees[1]:
            out.append(f"{e['id']}: corpus differs between {old} and {new}")
    return out


def _resolve(sha: str, maps: dict[str, str], entries: list[dict]) -> str:
    """The commit in today's history that stands for a pinned commit."""
    new = maps.get(sha)
    if new is not None and not _mapping_problems(sha, new, entries):
        return new
    return sha


def _rewrite_maps() -> dict[str, str]:
    return json.loads(REWRITE_MAP.read_text())["maps"]


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
def test_pinned_commit_is_40_hex_and_resolves_to_one_in_history(entry: dict) -> None:
    sha = entry["asqav_sdk_commit"]
    assert _COMMIT_RE.match(sha), (
        f"{entry['id']}: asqav_sdk_commit {sha!r} is not 40 lowercase hex"
    )
    real = _resolve(sha, _rewrite_maps(), _runs())
    probe = _git("cat-file", "-e", f"{real}^{{commit}}")
    assert probe.returncode == 0, (
        f"{entry['id']}: pinned commit {sha} resolves to {real}, "
        "which is not in this repository"
    )


@pytest.mark.parametrize("entry", _runs(), ids=_entry_ids())
def test_vectors_exist_in_the_corpus_at_the_resolved_commit(entry: dict) -> None:
    sha = _resolve(entry["asqav_sdk_commit"], _rewrite_maps(), _runs())
    corpus = entry["corpus"]
    listing = _git("ls-tree", "--name-only", "-d", sha, f"{corpus}/")
    assert listing.returncode == 0, (
        f"{entry['id']}: git ls-tree failed for {corpus} at {sha}: {listing.stderr}"
    )
    present = {line.rsplit("/", 1)[-1] for line in listing.stdout.splitlines()}
    for vector in entry["vectors"]:
        assert vector in present, (
            f"{entry['id']}: vector {vector!r} is not a directory under {corpus} "
            f"at resolved commit {sha}"
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


def test_every_mapped_commit_is_a_preserved_equal_replacement() -> None:
    for old, new in _rewrite_maps().items():
        assert _mapping_problems(old, new, _runs()) == []


def test_a_fabricated_mapping_is_refused() -> None:
    # NEGCTL: each planted line runs the same check as the real map
    old, new = next(iter(_rewrite_maps().items()))
    ghost = "0" * 40
    assert _mapping_problems(old, ghost, _runs())
    assert _mapping_problems("1" * 40, new, _runs())
    entry = dict(_runs()[0], asqav_sdk_commit="2" * 40)
    assert _mapping_problems("2" * 40, new, [entry])
    assert _mapping_problems(old, old, _runs())
    root = _git("rev-list", "--max-parents=0", "HEAD").stdout.split()[0]
    assert any(
        "corpus differs" in p for p in _mapping_problems(old, root, _runs())
    )
    assert _resolve("1" * 40, {"1" * 40: new}, _runs()) == "1" * 40
    # and the real line is followed
    assert _resolve(old, {old: new}, _runs()) == new
