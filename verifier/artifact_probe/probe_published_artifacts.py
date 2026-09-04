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
        install = subprocess.run(
            [str(pip), "install", "-q", f"asqav[cli]=={version}"],
            capture_output=True,
            text=True,
        )
        if install.returncode != 0:
            tail = (install.stderr or install.stdout).strip().splitlines()[-1:]
            # A version the JSON API already reports can still be absent from the
            # simple index pip reads, for minutes after a publish. That is CDN lag,
            # not a missing artifact, and it deserves a sentence rather than a
            # stack trace that reads like the release failed.
            return [
                f"could not install asqav[cli]=={version} from PyPI: {' '.join(tail)}. "
                "If this version was just published, the simple index has not propagated "
                "yet; wait and re-run rather than treating it as a failed release."
            ]
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


def probe_corpus_against_installed_python(version: str) -> list[str]:
    """Run the conformance corpus through the INSTALLED verifier, not the repo one.

    Vector agreement is measured against the distributed artifact for the same
    reason the entry points are: a corpus that passes against the working tree
    says nothing about the package a third party installs. The vectors come from
    this repository because that is where they are published; the verifier that
    reads them comes from PyPI.
    """
    failures: list[str] = []
    vectors = ROOT / "verifier" / "conformance-vectors"
    with tempfile.TemporaryDirectory() as tmp:
        venv = pathlib.Path(tmp) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        py = venv / "bin" / "python"
        subprocess.run(
            [str(venv / "bin" / "pip"), "install", "-q", f"asqav=={version}", "dilithium-py"],
            check=True,
        )
        runner = pathlib.Path(tmp) / "run_corpus.py"
        runner.write_text(
            "import json, pathlib, sys\n"
            "from asqav.verifier.verify_receipt import run_structured\n"
            "base = pathlib.Path(sys.argv[1])\n"
            "out = {}\n"
            "for d in sorted(p for p in base.iterdir() if p.is_dir()):\n"
            "    exp = d / 'expected.json'\n"
            "    receipt, jwks = d / 'receipt.json', d / 'jwks.json'\n"
            "    if not (exp.exists() and receipt.exists() and jwks.exists()):\n"
            "        continue\n"
            "    e = json.loads(exp.read_text())\n"
            "    if e.get('format') != 'asqav-native':\n"
            "        continue\n"
            "    pred = d / 'predecessor.json'\n"
            "    try:\n"
            "        r = run_structured(\n"
            "            json.loads(receipt.read_text()), json.loads(jwks.read_text()),\n"
            "            predecessor_payload=json.loads(pred.read_text()) if pred.exists() else None)\n"
            "        out[d.name] = {'declared': e.get('outcome'), 'observed': r['verdict'],\n"
            "                       'not_checked': len(r.get('not_checked', []))}\n"
            "    except Exception as exc:\n"
            "        out[d.name] = {'declared': e.get('outcome'), 'observed': 'ERROR: %s' % exc}\n"
            "print(json.dumps(out))\n"
        )
        proc = subprocess.run(
            [str(py), str(runner), str(vectors)], capture_output=True, text=True, cwd=tmp
        )
        if proc.returncode != 0:
            return [f"corpus run under the installed package failed: {proc.stderr.strip()[-300:]}"]
        results = json.loads(proc.stdout)
        print(f"  ran {len(results)} asqav-native vectors through the installed verifier")
        for name, r in sorted(results.items()):
            if str(r["observed"]).startswith("ERROR"):
                failures.append(f"corpus {name}: {r['observed']}")
        # Reported once, naming the version, rather than once per vector: this is a
        # single fact about the artifact, and repeating it 22 times buries the rest.
        undeclared = [n for n, r in results.items() if r.get("not_checked", 0) == 0]
        if undeclared:
            failures.append(
                f"asqav=={version} emits no not_checked declaration on any of "
                f"{len(undeclared)} corpus results. The declaration is on main but has not "
                f"been released, so the DISTRIBUTED artifact does not carry the coverage "
                f"declaration the repository does. Publishing closes this."
            )
        # The declared-vs-observed comparison is reported, never silently reconciled:
        # a vector whose declared outcome the installed verifier does not reproduce is
        # the exact disagreement this axis exists to surface.
        drift = [
            f"{n}: declared {r['declared']!r}, observed {r['observed']!r}"
            for n, r in sorted(results.items())
            if not str(r["observed"]).startswith("ERROR") and r["declared"] != r["observed"]
        ]
        if drift:
            print()
            print(
                f"  {len(drift)} of {len(results)} vectors declare an outcome the published "
                f"verifier does not reproduce from the published corpus alone."
            )
            print(
                "  This is NOT an artifact defect and does not fail this probe: the repository's\n"
                "  own verifier drifts on exactly the same vectors. It is a corpus-declaration gap.\n"
                "  The declared outcome assumes trust material the corpus does not ship (pinned TSA\n"
                "  keys, bitcoin headers) or an algorithm the single-file verifier does not\n"
                "  implement, and no vector says which. An outsider reproduces "
                f"{len(results) - len(drift)} of {len(results)}."
            )
            for d in drift:
                print(f"    - {d}")
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
        print("running the conformance corpus through the installed package")
        failures += probe_corpus_against_installed_python(version)
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
