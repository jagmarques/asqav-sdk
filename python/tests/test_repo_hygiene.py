"""Agent and session leftovers stay untracked; a new dot entry needs review."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Each has a consumer: Actions, git, gitleaks, and npm pack for the TS package
DOT_OK = {".github", ".gitignore", ".gitleaks.toml", ".npmignore"}
JUNK = [
    re.compile(r"(^|/)(results|scratch|scratchpad|research)/"),
    re.compile(r"(^|/)CLAUDE\.local\.md$"),
    re.compile(r"(^|/)(PENDING|HANDOVER|HANDOFF)(\.md)?$", re.I),
    re.compile(r"\.(bak|orig|rej|swp)$|\.bak\.|~$"),
]
# The repo's own .gitignore must hold each rule, not a developer's global one
GIT_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def junk(paths: list[str]) -> list[str]:
    return [
        p for p in paths
        if any(r.search(p) for r in JUNK)
        or any(part.startswith(".") and part not in DOT_OK for part in p.split("/"))
    ]


def ignored(path: str) -> bool:
    r = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT, env=GIT_ENV)
    return r.returncode == 0


def test_no_leftover_is_tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    files = [p for p in out.stdout.decode().split("\0") if p]
    assert len(files) > 100, len(files)
    assert junk(files) == []


def test_negctl_planted_leftovers_are_caught():
    planted = ["results/c1/OBSERVE.md", ".cursor/rules.mdc", "a/CLAUDE.local.md", "x.py.bak",
               "docs/PENDING.md", "tests/.aider/x", ".DS_Store", "scratchpad/n.txt", ".relay/r.md"]
    assert junk(planted) == planted
    kept = [".github/workflows/ci.yml", "typescript/.npmignore", ".gitleaks.toml", "CHANGELOG.md"]
    assert junk(kept) == []


def test_local_output_is_ignored():
    local = ["CLAUDE.local.md", "results/x.md", "scratchpad/x", ".relay/x", ".playwright-mcp/x",
             ".claude/x", ".asondy/x", ".cursor/x", ".aider.conf.yml", ".continue/x", ".DS_Store"]
    assert [p for p in local if not ignored(p)] == []
    assert not ignored("python/tests/test_repo_hygiene.py"), "NEGCTL: source reads as ignored"
