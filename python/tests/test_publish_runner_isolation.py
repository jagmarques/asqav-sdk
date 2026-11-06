"""Only tag-pushed or manual workflows may use the hz-publish runner."""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
PUBLISH_LABEL = "hz-publish"
SUFFIXES = (".yml", ".yaml")


def _strings(node: object) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [s for item in node for s in _strings(item)]
    if isinstance(node, dict):
        return [s for item in node.values() for s in _strings(item)]
    return []


def _uses_publish_runner(doc: dict) -> bool:
    jobs = doc.get("jobs") or {}
    for job in jobs.values():
        labels = _strings(job.get("runs-on"))
        if any(PUBLISH_LABEL in s or "${{" in s for s in labels):
            return True
    return False


def _triggers(doc: dict) -> dict:
    on = doc.get("on", doc.get(True))
    if isinstance(on, str):
        return {on: None}
    if isinstance(on, list):
        return {name: None for name in on}
    return on or {}


def _tag_only_push(push: object) -> bool:
    if not isinstance(push, dict) or not push.get("tags"):
        return False
    return set(push) == {"tags"}


def _problems(name: str, text: str) -> list[str]:
    doc = yaml.safe_load(text) or {}
    if not _uses_publish_runner(doc):
        return []
    out = []
    for trigger, body in _triggers(doc).items():
        if trigger == "workflow_dispatch":
            continue
        if trigger == "push" and _tag_only_push(body):
            continue
        out.append(f"{name}: {trigger} can reach the {PUBLISH_LABEL} runner")
    return out


def _workflow_files(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.suffix in SUFFIXES)


def _all_problems(root: Path) -> dict[str, list[str]]:
    found = {}
    for f in _workflow_files(root):
        found[f.name] = _problems(f.name, f.read_text())
    return found


def test_publish_runner_is_reached_only_by_tag_or_manual_runs() -> None:
    used = [
        f
        for f in _workflow_files(WORKFLOWS)
        if _uses_publish_runner(yaml.safe_load(f.read_text()))
    ]
    assert used, f"no workflow uses {PUBLISH_LABEL}"
    assert {n: p for n, p in _all_problems(WORKFLOWS).items() if p} == {}


def test_a_workflow_on_the_publish_runner_with_another_trigger_is_refused() -> None:
    # NEGCTL: planted workflows run through the same _problems as the real ones
    forms = {
        "inline": f"runs-on: [self-hosted, {PUBLISH_LABEL}]",
        "block": f"runs-on:\n      - self-hosted\n      - {PUBLISH_LABEL}",
        "labels": f"runs-on:\n      labels: [{PUBLISH_LABEL}]",
        "string": f"runs-on: {PUBLISH_LABEL}",
    }
    triggers = [
        "pull_request",
        "pull_request_target",
        "pull_request_review",
        "pull_request_review_comment",
        "issue_comment",
        "issues",
        "schedule",
        "workflow_run",
        "workflow_call",
        "repository_dispatch",
        "check_run",
        "create",
    ]
    for form, runs_on in forms.items():
        job = f"jobs:\n  j:\n    {runs_on}\n"
        for t in triggers:
            text = f"on:\n  {t}:\n  workflow_dispatch:\n{job}"
            assert _problems("planted.yml", text), (form, t)
        for push in (
            "push:",
            "push:\n    branches: [main]",
            "push:\n    branches-ignore: [x]",
            "push:\n    paths: ['a']",
            "push:\n    tags: []",
            "push:\n    tags: ['v*']\n    branches: [main]",
        ):
            text = f"on:\n  {push}\n{job}"
            assert _problems("planted.yml", text), (form, push)
    job = f"jobs:\n  j:\n    runs-on: [{PUBLISH_LABEL}]\n"
    assert _problems("planted.yml", f"on: pull_request\n{job}")
    assert _problems("planted.yml", f"on: [push, workflow_dispatch]\n{job}")
    expr = "jobs:\n  j:\n    runs-on: ${{ matrix.os }}\n"
    assert _problems("planted.yml", f"on:\n  pull_request:\n{expr}")


def test_allowed_triggers_and_other_runners_pass() -> None:
    job = f"jobs:\n  j:\n    runs-on: [self-hosted, {PUBLISH_LABEL}]\n"
    clean = f"on:\n  push:\n    tags: ['v*']\n  workflow_dispatch:\n{job}"
    assert _problems("planted.yml", clean) == []
    other = "on:\n  pull_request:\njobs:\n  j:\n    runs-on: [self-hosted]\n"
    assert _problems("planted.yml", other) == []


def test_yaml_extension_workflows_are_checked(tmp_path: Path) -> None:
    # NEGCTL: planted b.yaml on the publish runner with a PR trigger
    job = f"jobs:\n  j:\n    runs-on: [self-hosted, {PUBLISH_LABEL}]\n"
    (tmp_path / "a.yml").write_text(f"on:\n  workflow_dispatch:\n{job}")
    (tmp_path / "b.yaml").write_text(f"on:\n  pull_request:\n{job}")
    (tmp_path / "c.txt").write_text(f"on:\n  pull_request:\n{job}")
    found = _all_problems(tmp_path)
    assert sorted(found) == ["a.yml", "b.yaml"]
    assert found["a.yml"] == []
    assert found["b.yaml"]
