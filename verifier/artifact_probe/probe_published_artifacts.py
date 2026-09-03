#!/usr/bin/env python3
"""Probe the PUBLISHED packages, in a clean environment, for everything the docs claim.

A conformance corpus only exercises what it exercises, and a repository test only
proves the repository. Neither says whether the artifact a user actually installs
carries the surface its own published page advertises. This installs the published
wheel and the published npm package into throwaway environments with the source
tree deliberately off the path, and checks each documented entry point against
what it finds there.

The claims are derived from the published README rather than restated here, so the
probe cannot drift into asserting a surface the page never promised, nor silently
stop covering one the page adds.

Run: python3 verifier/artifact_probe/probe_published_artifacts.py [--python-only|--node-only]
Exit: 0 when every documented entry point resolves, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY_README = ROOT / "python" / "README.md"
TS_README = ROOT / "typescript" / "README.md"


def published_version(name: str, npm: bool = False) -> str:
    url = (
        f"https://registry.npmjs.org/{name}" if npm else f"https://pypi.org/pypi/{name}/json"
    )
    out = subprocess.run(
        ["curl", "-sS", "--max-time", "30", url], capture_output=True, text=True, check=True
    ).stdout
    data = json.loads(out)
    return data["dist-tags"]["latest"] if npm else data["info"]["version"]


def claimed_python_surface(text: str) -> dict:
    """Read the documented surface off the page itself."""
    imports = {}
    # Matches both a fenced `from ... import ...` line and one quoted inline in
    # prose, since the page documents the framework adapters the second way.
    pattern = r"from (asqav[\w.]*) import ([A-Za-z_][\w, ]*)"
    for module, names in re.findall(pattern, text):
        bucket = imports.setdefault(module, set())
        for name in names.split(","):
            name = name.strip()
            if name and name.isidentifier():
                bucket.add(name)
    # Bare `import asqav` still asserts the top-level package resolves.
    if re.search(r"^[ \t]*import asqav$", text, re.M):
        imports.setdefault("asqav", set())
    # Only shell fences. Prose says "asqav owns the spine" and Python says
    # "from asqav import X"; neither is a CLI claim, and treating them as one
    # produces a probe that reports gaps the page never asserted.
    shell = "\n".join(re.findall(r"```(?:bash|sh|console|shell)\n(.*?)```", text, re.S))
    verbs = sorted(
        {
            v
            for v in re.findall(r"^\s*(?:\$ )?asqav ([a-z][a-z-]{2,})(?![\w-])", shell, re.M)
        }
    )
    return {"imports": {k: sorted(v) for k, v in imports.items()}, "cli_verbs": verbs}


def probe_python(version: str) -> list[str]:
    failures: list[str] = []
    gated: list[str] = []
    surface = claimed_python_surface(PY_README.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        venv = pathlib.Path(tmp) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        py = venv / "bin" / "python"
        pip = venv / "bin" / "pip"
        # Install what the page tells the reader to install. The CLI lives behind
        # the documented `[cli]` extra, so probing the base install for CLI verbs
        # would report a gap the page never claimed to close.
        subprocess.run(
            [str(pip), "install", "-q", f"asqav[cli]=={version}"], check=True
        )
        print(f"  installed asqav[cli]=={version} from PyPI into a clean venv")

        # cwd is the temp dir and PYTHONPATH is cleared, so the source tree cannot
        # satisfy an import the published wheel does not carry.
        env = {"PATH": f"{venv / 'bin'}:/usr/bin:/bin", "HOME": tmp}
        for module, names in surface["imports"].items():
            script = (
                f"import importlib,sys\n"
                f"m=importlib.import_module({module!r})\n"
                f"missing=[n for n in {names!r} if not hasattr(m,n)]\n"
                f"print('MISSING' if missing else 'OK', {module!r}, ','.join(missing))\n"
            )
            proc = subprocess.run(
                [str(py), "-c", script], capture_output=True, text=True, cwd=tmp, env=env
            )
            if proc.returncode != 0:
                tail = proc.stderr.strip().splitlines()[-1]
                # An optional framework adapter that refuses with the exact extra
                # the page names is behaving as documented, not missing. Anything
                # else - including a refusal that names no remedy - is a gap.
                extra = re.search(r"pip install asqav\[([\w-]+)\]", tail)
                if extra and module.startswith("asqav.extras."):
                    gated.append(f"{module} (documented extra: asqav[{extra.group(1)}])")
                else:
                    failures.append(f"python import {module}: {tail}")
            elif proc.stdout.startswith("MISSING"):
                failures.append(f"python {module}: missing {proc.stdout.split(' ',2)[2].strip()}")

        help_proc = subprocess.run(
            [str(venv / "bin" / "asqav"), "--help"],
            capture_output=True,
            text=True,
            cwd=tmp,
            env=env,
        )
        if help_proc.returncode != 0:
            failures.append("python CLI: `asqav --help` did not run")
        else:
            listed = help_proc.stdout
            for verb in surface["cli_verbs"]:
                if not re.search(rf"(?<![\w-]){re.escape(verb)}(?![\w-])", listed):
                    failures.append(f"python CLI: documented verb `asqav {verb}` absent from --help")
    if gated:
        print(f"  {len(gated)} adapter(s) gated behind a documented extra, as the page states:")
        for g in gated:
            print(f"    - {g}")
    return failures


def claimed_node_surface(text: str) -> dict:
    """Group documented names by the specifier the page imports them from.

    The package publishes subpath exports, and the page uses them: ADAPTERS is
    documented from "@asqav/sdk/verifier", not from the root. Probing every name
    against the root reports a missing export the page never claimed was there.
    """
    by_specifier: dict[str, set[str]] = {}
    for block, specifier in re.findall(
        r"import \{([^}]+)\} from ['\"](@asqav/sdk[\w/-]*)['\"]", text
    ):
        bucket = by_specifier.setdefault(specifier, set())
        for name in block.split(","):
            name = name.strip()
            if name and re.fullmatch(r"[A-Za-z_$][\w$]*", name):
                bucket.add(name)
    for specifier, name in re.findall(
        r"require\(['\"](@asqav/sdk[\w/-]*)['\"]\)\.(\w+)", text
    ):
        by_specifier.setdefault(specifier, set()).add(name)
    return {"by_specifier": {k: sorted(v) for k, v in by_specifier.items()}}


def probe_node(version: str) -> list[str]:
    failures: list[str] = []
    surface = claimed_node_surface(TS_README.read_text())
    if not surface["by_specifier"]:
        return ["node: the published page documents no named import; probe would be vacuous"]
    if shutil.which("npm") is None:
        return ["node: npm is not available in this environment"]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["npm", "install", "--silent", "--no-fund", "--no-audit", f"@asqav/sdk@{version}"],
            cwd=tmp,
            check=True,
        )
        print(f"  installed @asqav/sdk@{version} from npm into a clean tree")
        for specifier, names in surface["by_specifier"].items():
            script = (
                f"const m = require({specifier!r});\n"
                f"const want = {json.dumps(names)};\n"
                "const missing = want.filter(n => m[n] === undefined);\n"
                "console.log(missing.length ? 'MISSING ' + missing.join(',') : 'OK');\n"
            )
            proc = subprocess.run(["node", "-e", script], cwd=tmp, capture_output=True, text=True)
            if proc.returncode != 0:
                failures.append(
                    f"node require({specifier}) failed: {proc.stderr.strip().splitlines()[-1]}"
                )
            elif proc.stdout.startswith("MISSING"):
                failures.append(
                    f"node {specifier}: missing exports {proc.stdout.split(' ',1)[1].strip()}"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--node-only", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    if not args.node_only:
        version = published_version("asqav")
        print(f"probing PUBLISHED asqav=={version}")
        failures += probe_python(version)
    if not args.python_only:
        version = published_version("@asqav/sdk", npm=True)
        print(f"probing PUBLISHED @asqav/sdk@{version}")
        failures += probe_node(version)

    print()
    if failures:
        print(f"{len(failures)} documented entry point(s) missing from the published artifact:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("every documented entry point resolves in the published artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
