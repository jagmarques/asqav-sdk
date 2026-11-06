"""Only tag, release or manual workflows may use the hz-publish runner."""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
PUBLISH_LABEL = "hz-publish"
PR_TRIGGERS = (
    "pull_request",
    "pull_request_target",
    "schedule",
    "issue_comment",
    "workflow_run",
)
_RUNS_ON = re.compile(r"^\s*runs-on:.*\b" + PUBLISH_LABEL + r"\b", re.M)


def _on_block(text: str) -> str:
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if re.match(r"^on:\s*$", ln))
    block = []
    for ln in lines[start + 1 :]:
        if ln and not ln.startswith((" ", "#")):
            break
        block.append(ln)
    return "\n".join(block)


def _problems(name: str, text: str) -> list[str]:
    if not _RUNS_ON.search(text):
        return []
    on = _on_block(text)
    out = [
        f"{name}: {t} trigger can reach the {PUBLISH_LABEL} runner"
        for t in PR_TRIGGERS
        if re.search(r"^\s{2}" + t + r"\s*:", on, re.M)
    ]
    if re.search(r"^\s{4}branches\s*:", on, re.M):
        out.append(f"{name}: a branch push can reach the {PUBLISH_LABEL} runner")
    return out


def test_publish_runner_is_reached_only_by_tag_or_manual_runs() -> None:
    files = sorted(WORKFLOWS.glob("*.yml"))
    used = [f for f in files if _RUNS_ON.search(f.read_text())]
    assert used, f"no workflow uses {PUBLISH_LABEL}"
    for f in files:
        assert _problems(f.name, f.read_text()) == []


def test_a_pr_reachable_workflow_on_the_publish_runner_is_refused() -> None:
    # NEGCTL: planted workflows run through the same _problems as the real ones
    job = f"jobs:\n  j:\n    runs-on: [self-hosted, {PUBLISH_LABEL}]\n"
    for trigger in PR_TRIGGERS:
        text = f"on:\n  {trigger}:\n  workflow_dispatch:\n{job}"
        assert _problems("planted.yml", text), trigger
    pushed = f"on:\n  push:\n    branches: [main]\n{job}"
    assert _problems("planted.yml", pushed)
    clean = f"on:\n  push:\n    tags: ['v*']\n  workflow_dispatch:\n{job}"
    assert _problems("planted.yml", clean) == []
    elsewhere = "on:\n  pull_request:\njobs:\n  j:\n    runs-on: [self-hosted]\n"
    assert _problems("planted.yml", elsewhere) == []
