"""The ci-ok job is the required check; it must never pass on a hidden skip."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

CI = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
AREAS = {"python": "PY", "typescript": "TS", "verifier": "VER", "actions": "ACT"}


def _jobs() -> dict:
    return yaml.safe_load(CI.read_text())["jobs"]


def _step() -> dict:
    return next(s for s in _jobs()["ci-ok"]["steps"] if "run" in s)


def _script() -> str:
    return _step()["run"]


def _rows() -> list[tuple[str, str]]:
    template = _step()["env"]["RESULTS"]
    return [(m[0], m[1]) for m in re.findall(r"^(\S+) (\S+) \$\{\{", template, re.M)]


def _run(script: str, changes: str, changed: dict[str, str], results: dict[str, str]) -> int:
    results_text = "".join(
        f"{job} {area} {results.get(job, 'success')}\n" for job, area in _rows()
    )
    env = {"CHANGES": changes, "RESULTS": results_text, "PATH": "/usr/bin:/bin"}
    for area, var in AREAS.items():
        env[var] = changed.get(area, "false")
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True).returncode


def _area_job(area: str) -> str:
    return next(job for job, a in _rows() if a == area)


def violations(script: str) -> list[str]:
    found: list[str] = []

    def expect(label: str, code: int, want_fail: bool) -> None:
        if (code != 0) != want_fail:
            found.append(label)

    expect("all green", _run(script, "success", {}, {}), False)
    expect("changes failed", _run(script, "failure", {}, {}), True)
    for area in AREAS:
        job = _area_job(area)
        hidden = _run(script, "success", {area: "true"}, {job: "skipped"})
        expect(f"{job} skipped, {area} changed", hidden, True)
        expect(f"{job} skipped, {area} unchanged", _run(script, "success", {area: "false"}, {job: "skipped"}), False)
        expect(f"{job} failed", _run(script, "success", {}, {job: "failure"}), True)
    expect("always-run job skipped", _run(script, "success", {}, {"wire-vocabulary": "skipped"}), True)
    return found


def test_ci_ok_needs_every_other_job():
    others = set(_jobs()) - {"ci-ok"}
    assert set(_jobs()["ci-ok"]["needs"]) == others
    assert _jobs()["ci-ok"]["if"] == "always()"


def test_every_needed_job_is_read_by_the_script():
    listed = {job for job, _ in _rows()}
    assert listed == set(_jobs()["ci-ok"]["needs"]) - {"changes"}


def test_ci_ok_fails_on_changes_failure_hidden_skip_and_failure():
    assert violations(_script()) == []


def test_negctl_the_reverted_logic_is_reported_by_the_same_checks():
    # NEGCTL: reverted scripts go through the same violations() as the real test
    script = _script()
    no_changes_gate = script.replace('[[ "$CHANGES" == "success" ]] || exit 1', "true")
    skip_is_ok = script.replace(
        'if [[ "$result" == "skipped" && "$changed" != "true" ]]; then continue; fi',
        'if [[ "$result" == "skipped" ]]; then continue; fi',
    )
    assert no_changes_gate != script
    assert skip_is_ok != script
    assert "changes failed" in violations(no_changes_gate)
    assert any("skipped, " in v and "changed" in v for v in violations(skip_is_ok))
