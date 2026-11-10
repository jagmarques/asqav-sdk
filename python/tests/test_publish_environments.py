"""Publish jobs declare the registry environment their step needs."""

from __future__ import annotations

import copy

import yaml

from tests.test_publish_runner_isolation import WORKFLOWS

PUBLISH_WORKFLOW = WORKFLOWS / "publish.yml"
NPM_ENV = "npm"
PYPI_ENV = "pypi"
NPM_PUBLISH = "npm publish"
PYPI_PUBLISH = "pypa/gh-action-pypi-publish"
TAG_PREFIX = {NPM_ENV: "refs/tags/ts-v", PYPI_ENV: "refs/tags/py-v"}


def _load() -> dict:
    return yaml.safe_load(PUBLISH_WORKFLOW.read_text()) or {}


def _environment_name(node: object) -> str | None:
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        return node.get("name")
    return None


def jobs_running(doc: dict, marker: str) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for name, job in (doc.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            if not isinstance(step, dict):
                continue
            run = step.get("run")
            uses = step.get("uses")
            if (isinstance(run, str) and marker in run) or (
                isinstance(uses, str) and uses.startswith(marker)
            ):
                found[name] = job
                break
    return found


def _problems(doc: dict) -> list[str]:
    out: list[str] = []
    for env, marker in ((NPM_ENV, NPM_PUBLISH), (PYPI_ENV, PYPI_PUBLISH)):
        prefix = TAG_PREFIX[env]
        for name, job in jobs_running(doc, marker).items():
            if _environment_name(job.get("environment")) != env:
                out.append(f"{name}: environment is not {env}")
            if f"startsWith(github.ref, '{prefix}')" not in str(job.get("if") or ""):
                out.append(f"{name}: if is not limited to {prefix}")
    return out


def test_npm_publish_job_has_the_npm_environment() -> None:
    doc = _load()
    assert jobs_running(doc, NPM_PUBLISH), f"no job runs {NPM_PUBLISH}"
    assert _problems(doc) == []


def test_pypi_publish_job_has_the_pypi_environment() -> None:
    doc = _load()
    assert jobs_running(doc, PYPI_PUBLISH), f"no job uses {PYPI_PUBLISH}"
    assert _problems(doc) == []


def test_npm_job_if_is_limited_to_ts_v_tags() -> None:
    doc = _load()
    assert jobs_running(doc, NPM_PUBLISH), f"no job runs {NPM_PUBLISH}"
    assert _problems(doc) == []


def test_pypi_job_if_is_limited_to_py_v_tags() -> None:
    doc = _load()
    assert jobs_running(doc, PYPI_PUBLISH), f"no job uses {PYPI_PUBLISH}"
    assert _problems(doc) == []


def test_npm_environment_check_rejects_missing_environment() -> None:
    # NEGCTL: the shipped checker runs on planted copies of the parsed workflow
    doc = _load()
    missing = copy.deepcopy(doc)
    missing["jobs"]["publish-typescript"].pop("environment", None)
    assert _problems(missing), "checker accepted a missing npm environment"

    wrong = copy.deepcopy(doc)
    wrong["jobs"]["publish-python"]["environment"] = {"name": NPM_ENV}
    assert _problems(wrong), "checker accepted a pypi job in the npm environment"

    broad = copy.deepcopy(doc)
    broad["jobs"]["publish-typescript"]["if"] = "startsWith(github.ref, 'refs/tags/v*')"
    assert _problems(broad), "checker accepted a tag if outside ts-v"
