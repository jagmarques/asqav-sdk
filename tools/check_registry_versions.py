"""Fail when a released SDK version is missing from, or behind on, its registry."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PYPI = "https://pypi.org/pypi/asqav/json"
NPM = "https://registry.npmjs.org/@asqav%2fsdk"
# A tag names a release; its publish job can fail unnoticed (npm 0.10.11-12)
PREFIXES = {"python": "py-v", "typescript": "ts-v"}
REGISTRY = {"python": "PyPI", "typescript": "npm"}


def parse(version):
    """Numeric (major, minor, patch), or None for anything else."""
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version or "")
    return tuple(int(part) for part in match.groups()) if match else None


def fmt(version):
    return ".".join(str(part) for part in version) if version else "none"


def newest_tags(tags):
    """Newest released version per language, from py-v* and ts-v* tag names."""
    newest = {}
    for lang, prefix in PREFIXES.items():
        found = [parse(tag[len(prefix):]) for tag in tags if tag.startswith(prefix)]
        found = [version for version in found if version]
        newest[lang] = max(found) if found else None
    return newest


def repo_versions(root=ROOT):
    pyproject = (root / "python/pyproject.toml").read_text(encoding="utf-8")
    python = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    package = json.loads((root / "typescript/package.json").read_text(encoding="utf-8"))
    return {"python": parse(python), "typescript": parse(package["version"])}


def drift(tags, registry, repo):
    """One line per problem; empty when every tagged release is published."""
    problems = []
    for lang in PREFIXES:
        tag, state, here = tags.get(lang), registry.get(lang) or {}, repo.get(lang)
        latest = parse(state.get("latest"))
        published = {parse(version) for version in state.get("versions", ())}
        name = REGISTRY[lang]
        if tag and tag not in published:
            problems.append(f"{lang}: tag {fmt(tag)} is not on {name} (latest {fmt(latest)})")
        elif tag and (latest is None or latest < tag):
            problems.append(f"{lang}: {name} latest {fmt(latest)} is behind tag {fmt(tag)}")
        if here and latest and here < latest:
            problems.append(f"{lang}: repo version {fmt(here)} is behind {name} {fmt(latest)}")
    return problems


def fetch(url):
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "asqav-sdk-registry-drift"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def registry_state():
    pypi, npm = fetch(PYPI), fetch(NPM)
    return {
        "python": {"latest": pypi["info"]["version"], "versions": list(pypi["releases"])},
        "typescript": {"latest": npm["dist-tags"]["latest"], "versions": list(npm["versions"])},
    }


def git_tags(root=ROOT):
    out = subprocess.run(["git", "tag", "--list", "py-v*", "ts-v*"], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    return out.split()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", type=int, default=0,
                        help="keep checking for this many seconds, for a publish that just ran")
    args = parser.parse_args(argv)
    tags, repo = newest_tags(git_tags()), repo_versions()
    if not any(tags.values()):
        print("no py-v* or ts-v* tags: fetch tags first (fetch-depth: 0)", file=sys.stderr)
        return 2
    deadline = time.monotonic() + args.wait
    while True:
        registry = registry_state()
        problems = drift(tags, registry, repo)
        if not problems or time.monotonic() >= deadline:
            break
        time.sleep(30)
    for lang in PREFIXES:
        print(f"{lang}: repo {fmt(repo[lang])}, tag {fmt(tags[lang])}, "
              f"{REGISTRY[lang]} latest {registry[lang]['latest']}")
    for problem in problems:
        print("DRIFT " + problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
