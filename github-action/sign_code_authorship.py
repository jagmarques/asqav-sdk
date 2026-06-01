#!/usr/bin/env python3
"""Sign an Asqav code-authorship receipt from a GitHub Actions run.

Honest scope: this script signs what git reports (the commit, its base, and a
digest of the diff between them) plus the Asqav agent key. Everything else on
the receipt is producer-asserted and recorded, never verified by Asqav. Model
authorship fields (`model_id` / `model_version`) are ALWAYS sent under
`authored_by.attestation_source="github-actions-input"` so they can never be
read as an Asqav-verified attestation.

The signed receipt is also wrapped as an in-toto Statement v1 and written to a
file so it interoperates with attestation tooling that consumes in-toto.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

RECEIPT_TYPE = "protectmcp:lifecycle:code_authorship"
PREDICATE_TYPE = "https://asqav.com/CodeAuthorship/v1"
INTOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
#: Fixed label so model authorship is always recorded as producer-asserted.
ATTESTATION_SOURCE = "github-actions-input"


@dataclass
class GitContext:
    """The git facts the receipt binds to, read from the GitHub Actions env."""

    repo_ref: str
    commit_sha: str
    base_sha: str | None
    change_ref: str | None
    change_digest: str


def _run_git_diff(base_sha: str, head_sha: str) -> str:
    """Return the textual `git diff <base>..<head>`.

    Raises CalledProcessError if git fails so the action surfaces the problem
    rather than silently signing an empty digest.
    """
    out = subprocess.run(
        ["git", "diff", f"{base_sha}..{head_sha}"],
        check=True,
        capture_output=True,
    )
    return out.stdout.decode("utf-8", errors="replace")


def compute_change_digest(base_sha: str | None, head_sha: str, diff_text: str | None = None) -> str:
    """Compute `change_digest` = "sha256:" + sha256(`git diff base..head`).

    When `diff_text` is supplied it is hashed directly (used by tests and to
    avoid re-shelling). When it is None and a base is known, the diff is read
    from git. When no base is known, the bare head sha is hashed so the field
    is always a well-formed sha256:<64 hex> existence proof.

    Always returns the cloud wire form `sha256:<64 lowercase hex>`.
    """
    if diff_text is None:
        if base_sha:
            diff_text = _run_git_diff(base_sha, head_sha)
        else:
            diff_text = head_sha
    digest = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_authored_by(
    *,
    agent_id: str | None,
    model_id: str | None,
    model_version: str | None,
    tool: str | None,
) -> dict[str, Any]:
    """Build the producer-asserted `authored_by` object.

    `attestation_source` is ALWAYS set to ``github-actions-input`` so any model
    authorship claim is recorded as producer-asserted and is never read as an
    Asqav-verified attestation. The SDK enforces that a populated model_id /
    model_version requires attestation_source; we satisfy that unconditionally.
    """
    authored_by: dict[str, Any] = {"attestation_source": ATTESTATION_SOURCE}
    if agent_id:
        authored_by["agent_id"] = agent_id
    if model_id:
        authored_by["model_id"] = model_id
    if model_version:
        authored_by["model_version"] = model_version
    if tool:
        authored_by["tool"] = tool
    return authored_by


def _bare_change_digest_hex(change_digest: str) -> str:
    """Strip the `sha256:` prefix from a change_digest for the in-toto subject."""
    return change_digest.split(":", 1)[-1]


def build_intoto_statement(
    *,
    repo_ref: str,
    commit_sha: str,
    change_digest: str,
    signed_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Wrap the signed receipt payload as an in-toto Statement v1.

    Subject names the repo at the signed commit and carries the bare commit
    sha as its sha256 digest (the resource the attestation is about). The
    predicate is the returned signed receipt payload, including the ML-DSA-65
    signature envelope, so a consumer can re-verify it offline.
    """
    return {
        "_type": INTOTO_STATEMENT_TYPE,
        "subject": [
            {
                "name": f"{repo_ref}@{commit_sha}",
                "digest": {"sha256": commit_sha},
            }
        ],
        "predicateType": PREDICATE_TYPE,
        "predicate": signed_receipt,
    }


def _signed_receipt_payload(response: Any) -> dict[str, Any]:
    """Project the SDK SignatureResponse into a JSON-able signed-receipt payload.

    Prefers the cloud three-key compliance payload when present; always carries
    the signature envelope (alg/kid/sig, alg=ML-DSA-65) plus the identifiers a
    verifier needs.
    """
    envelope = None
    try:
        envelope = response.signature_envelope()
    except Exception:
        envelope = None
    payload: dict[str, Any] = {
        "signature_id": response.signature_id,
        "action_id": response.action_id,
        "receipt_type": response.receipt_type or RECEIPT_TYPE,
        "algorithm": response.algorithm,
        "signature": envelope if envelope is not None else response.signature,
        "verification_url": response.verification_url,
        "timestamp": response.timestamp,
    }
    if getattr(response, "payload", None) is not None:
        payload["payload"] = response.payload
    if getattr(response, "anchors", None) is not None:
        payload["anchors"] = response.anchors
    if getattr(response, "previous_receipt_hash", None) is not None:
        payload["previous_receipt_hash"] = response.previous_receipt_hash
    return payload


def _read_git_context() -> GitContext:
    """Derive the git context from the GitHub Actions environment.

    repo_ref  <- GITHUB_REPOSITORY (owner/repo)
    commit_sha<- GITHUB_SHA
    base_sha  <- the PR base sha read from the event payload (pull_request.base.sha),
                 else GITHUB_BASE_REF resolved, else None
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


def sign_and_export(
    *,
    api_key: str,
    agent_id: str,
    change_class: str,
    model_id: str | None,
    model_version: str | None,
    tool: str | None,
    intoto_path: str,
    git_ctx: GitContext | None = None,
) -> dict[str, str]:
    """Sign the code-authorship receipt and emit the in-toto Statement.

    Returns a dict of the GitHub outputs: signature-id, verification-url,
    intoto-statement-path.
    """
    import asqav

    ctx = git_ctx or _read_git_context()

    asqav.init(api_key=api_key)
    agent = asqav.Agent.get(agent_id)

    authored_by = build_authored_by(
        agent_id=agent_id,
        model_id=model_id,
        model_version=model_version,
        tool=tool,
    )

    response = agent.sign(
        "code:authorship",
        receipt_type=RECEIPT_TYPE,
        compliance_mode=True,
        policy_decision="none",
        repo_ref=ctx.repo_ref,
        commit_sha=ctx.commit_sha,
        base_sha=ctx.base_sha,
        change_digest=ctx.change_digest,
        change_ref=ctx.change_ref,
        change_approval_ref=ctx.change_ref,
        change_class=change_class,
        authored_by=authored_by,
    )

    signed_payload = _signed_receipt_payload(response)
    statement = build_intoto_statement(
        repo_ref=ctx.repo_ref,
        commit_sha=ctx.commit_sha,
        change_digest=ctx.change_digest,
        signed_receipt=signed_payload,
    )
    with open(intoto_path, "w", encoding="utf-8") as fh:
        json.dump(statement, fh, indent=2, sort_keys=True)

    print(f"Signed code-authorship receipt: {response.signature_id}")
    print(f"Verify at: {response.verification_url}")
    print(f"in-toto Statement written to: {intoto_path}")

    outputs = {
        "signature-id": response.signature_id,
        "verification-url": response.verification_url,
        "intoto-statement-path": intoto_path,
    }
    _write_github_output(outputs)
    return outputs


def main() -> int:
    api_key = os.environ.get("ASQAV_API_KEY", "")
    agent_id = os.environ.get("ASQAV_AGENT_ID", "")
    if not api_key or not agent_id:
        print(
            "ERROR: ASQAV_API_KEY and ASQAV_AGENT_ID must be set.",
            file=sys.stderr,
        )
        return 2

    change_class = os.environ.get("ASQAV_CHANGE_CLASS", "write") or "write"
    model_id = os.environ.get("ASQAV_MODEL_ID") or None
    model_version = os.environ.get("ASQAV_MODEL_VERSION") or None
    tool = os.environ.get("ASQAV_TOOL") or "github-actions"

    workspace = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
    intoto_path = os.environ.get(
        "ASQAV_INTOTO_PATH",
        os.path.join(workspace, "asqav-code-authorship.intoto.jsonl"),
    )

    try:
        sign_and_export(
            api_key=api_key,
            agent_id=agent_id,
            change_class=change_class,
            model_id=model_id,
            model_version=model_version,
            tool=tool,
            intoto_path=intoto_path,
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the runner
        print(f"ERROR signing code-authorship receipt: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
