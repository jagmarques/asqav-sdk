#!/usr/bin/env python3
"""Record an Asqav code-authorship receipt from a GitHub Actions run.

Honest scope: the action computes an ADVISORY change digest (sha256 of
``git diff base..head``) and posts ``{repo, commit_sha, base_sha,
change_digest}`` to ``POST /v1/code-authorship``. The Asqav server re-fetches
the commit, recomputes the canonical diff, and signs the authoritative in-toto
Statement. ``subject[0].digest.sha256`` is the SERVER-recomputed diff hash, not
the action's advisory value. ``digest_match`` only reports whether the advisory
digest agreed with the server's.

The action writes the server's authoritative in-toto Statement to a file so it
interoperates with attestation tooling that consumes in-toto.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

INTOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"


@dataclass
class GitContext:
    """The git facts the request binds to, read from the GitHub Actions env."""

    repo_ref: str
    commit_sha: str
    base_sha: str | None
    change_ref: str | None
    change_digest: str


def compute_change_digest(base_sha: str | None, head_sha: str, diff_text: str | None = None) -> str:
    """Compute the ADVISORY ``change_digest`` = ``sha256:`` + sha256(diff).

    Delegates to the SDK so the action and the SDK agree byte-for-byte. The
    digest is advisory: the server recomputes its own and signs that.
    """
    from asqav.code_authorship import compute_advisory_digest

    return compute_advisory_digest(base_sha, head_sha, diff_text=diff_text)


def _read_git_context() -> GitContext:
    """Derive the git context from the GitHub Actions environment.

    repo_ref  <- GITHUB_REPOSITORY (owner/repo)
    commit_sha<- GITHUB_SHA
    base_sha  <- the PR base sha from the event payload (pull_request.base.sha)
    change_ref<- the PR html_url from the event payload, else the run URL
    """
    repo_ref = os.environ.get("GITHUB_REPOSITORY", "")
    commit_sha = os.environ.get("GITHUB_SHA", "")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")

    base_sha: str | None = None
    change_ref: str | None = None
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        try:
            with open(event_path, encoding="utf-8") as fh:
                event = json.load(fh)
            pr = event.get("pull_request") or {}
            base_sha = (pr.get("base") or {}).get("sha") or base_sha
            change_ref = pr.get("html_url") or change_ref
        except (OSError, ValueError):
            pass

    if change_ref is None and repo_ref:
        run_id = os.environ.get("GITHUB_RUN_ID", "")
        change_ref = f"{server}/{repo_ref}/actions/runs/{run_id}"

    change_digest = compute_change_digest(base_sha, commit_sha)
    return GitContext(
        repo_ref=repo_ref,
        commit_sha=commit_sha,
        base_sha=base_sha,
        change_ref=change_ref,
        change_digest=change_digest,
    )


def _write_github_output(values: dict[str, str]) -> None:
    """Append key=value lines to $GITHUB_OUTPUT if it is set."""
    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        return
    with open(out_path, "a", encoding="utf-8") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")


def _signature_id(result: Any) -> str:
    """Read the signature id from the server receipt, best-effort."""
    receipt = getattr(result, "receipt", None) or {}
    if isinstance(receipt, dict):
        value = receipt.get("signature_id")
        if isinstance(value, str):
            return value
    return ""


def sign_and_export(
    *,
    api_key: str,
    change_class: str,
    author: str | None,
    intoto_path: str,
    git_ctx: GitContext | None = None,
    anchor: str | None = None,
) -> dict[str, str]:
    """Post the advisory digest and emit the server's authoritative Statement.

    Returns the GitHub outputs: signature-id, digest-match, server-digest,
    intoto-statement-path.
    """
    import asqav
    from asqav.code_authorship import submit_code_authorship

    ctx = git_ctx or _read_git_context()

    asqav.init(api_key=api_key)
    result = submit_code_authorship(
        repo=ctx.repo_ref,
        commit_sha=ctx.commit_sha,
        base_sha=ctx.base_sha,
        change_digest=ctx.change_digest,
        change_class=change_class,
        author=author,
        anchor=anchor or ctx.change_ref,
    )

    # The server's envelope IS the authoritative in-toto Statement. Write it
    # verbatim so the bound subject digest is the server-recomputed value.
    with open(intoto_path, "w", encoding="utf-8") as fh:
        json.dump(result.envelope, fh, indent=2, sort_keys=True)

    signature_id = _signature_id(result)
    print(f"Recorded code-authorship receipt: {signature_id or '(no signature id)'}")
    print(f"capture_layer={result.capture_layer} digest_match={result.digest_match}")
    if result.subject_digest:
        print(f"server subject digest: {result.subject_digest}")
    print(f"in-toto Statement written to: {intoto_path}")

    outputs = {
        "signature-id": signature_id,
        "digest-match": "true" if result.digest_match else "false",
        "server-digest": result.server_digest or "",
        "capture-layer": result.capture_layer or "",
        "intoto-statement-path": intoto_path,
    }
    _write_github_output(outputs)
    return outputs


def main() -> int:
    api_key = os.environ.get("ASQAV_API_KEY", "")
    if not api_key:
        print(
            "ERROR: ASQAV_API_KEY must be set, with the code_authorship:write scope.",
            file=sys.stderr,
        )
        return 2

    change_class = os.environ.get("ASQAV_CHANGE_CLASS", "write") or "write"
    author = os.environ.get("ASQAV_AUTHOR") or None
    anchor = os.environ.get("ASQAV_ANCHOR") or None

    workspace = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
    intoto_path = os.environ.get(
        "ASQAV_INTOTO_PATH",
        os.path.join(workspace, "asqav-code-authorship.intoto.jsonl"),
    )

    try:
        sign_and_export(
            api_key=api_key,
            change_class=change_class,
            author=author,
            intoto_path=intoto_path,
            anchor=anchor,
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the runner
        print(f"ERROR recording code-authorship receipt: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
