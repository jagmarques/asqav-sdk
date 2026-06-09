#!/usr/bin/env python3
"""Mint or audit Asqav risk-acceptance receipts from a GitHub Actions run.

Honest scope: mint mode signs what the runner can see (a SHA-256 digest of the
SARIF file, the finding reference, the acceptance reason, and a validity
window) plus the identities the PR event reports. The approver is the
approving reviewer's login from `github.event.review` (or the merging user on
a merged pull_request event) and the initiator is the PR author's login; both
are producer-asserted and recorded, never verified by Asqav. The receipt
proves the acceptance was recorded and key-signed at a point in time with a
declared expiry, nothing more.

Audit mode fetches each receipt id named in a suppression manifest via the
public verify endpoint and exits nonzero when any referenced receipt is
missing, fails verification, is expired (server-evaluated valid_until), or is
superseded by another receipt fetched in the same run. Supersession is
detected client-side from the `supersedes` field on the fetched receipts, so
the audit works against the live verify endpoint alone with no list endpoint
required.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

RECEIPT_TYPE = "protectmcp:lifecycle:risk_acceptance"
#: validation_label the cloud verify cascade emits for a lapsed valid_until.
EXPIRED_LABEL = "signature_expired"

_DURATION_RE = re.compile(r"^(\d+)\s*([smhdw]?)$", re.IGNORECASE)
_DURATION_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


@dataclass
class ApprovalContext:
    """The PR-approval facts the receipt binds to, read from the event payload.

    All identities are producer-asserted GitHub logins; Asqav records them and
    never verifies them.
    """

    approver: str
    initiator: str | None
    accepted_at: str | None
    approval_ref: str | None


def parse_expires(value: str, *, now: datetime | None = None) -> int:
    """Convert the `expires` input into valid_seconds.

    Accepts a duration (90d, 12h, 30m, 3600s, 2w, or a bare integer of
    seconds) or an ISO 8601 date / datetime, which is converted to the number
    of seconds between now and that instant. Raises ValueError on a malformed
    value or a non-future date so the action fails clean instead of minting an
    already-expired acceptance.
    """
    value = value.strip()
    if not value:
        raise ValueError("expires is required: pass a duration (90d) or an ISO date.")
    m = _DURATION_RE.match(value)
    if m:
        seconds = int(m.group(1)) * _DURATION_UNITS[m.group(2).lower()]
        if seconds <= 0:
            raise ValueError(f"expires must be a positive duration (got: {value!r}).")
        return seconds
    try:
        target = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"expires is neither a duration nor an ISO date (got: {value!r})."
        ) from exc
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    seconds = int((target - now).total_seconds())
    if seconds <= 0:
        raise ValueError(f"expires date must be in the future (got: {value!r}).")
    return seconds


def compute_sarif_digest(path: str) -> str:
    """Return `sha256:` + SHA-256 of the SARIF file bytes.

    The digest is an existence proof of the exact scanner output the
    acceptance was minted against; the file itself never leaves the runner.
    """
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return f"sha256:{digest}"


def read_approval_context(event: dict[str, Any] | None = None) -> ApprovalContext:
    """Derive approver and initiator from the GitHub event payload.

    approver   <- review.user.login (pull_request_review event), else
                  pull_request.merged_by.login (merged pull_request event)
    initiator  <- pull_request.user.login (the PR author)
    accepted_at<- review.submitted_at when present
    approval_ref <- the review html_url, else the PR html_url

    Raises ValueError when no approver can be derived, because a
    risk-acceptance receipt without an approver is rejected by the SDK.
    """
    if event is None:
        event = {}
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if event_path and os.path.exists(event_path):
            try:
                with open(event_path, encoding="utf-8") as fh:
                    event = json.load(fh)
            except (OSError, ValueError):
                event = {}

    review = event.get("review") or {}
    pr = event.get("pull_request") or {}

    approver = (review.get("user") or {}).get("login")
    if not approver:
        approver = (pr.get("merged_by") or {}).get("login")
    if not approver:
        raise ValueError(
            "no approver in the event payload: run this action on a "
            "pull_request_review (submitted) event or a merged pull_request "
            "event so the approving login is present."
        )

    initiator = (pr.get("user") or {}).get("login") or None
    accepted_at = review.get("submitted_at") or None
    approval_ref = review.get("html_url") or pr.get("html_url") or None
    return ApprovalContext(
        approver=approver,
        initiator=initiator,
        accepted_at=accepted_at,
        approval_ref=approval_ref,
    )


def _write_github_output(values: dict[str, str]) -> None:
    """Append key=value lines to $GITHUB_OUTPUT if it is set."""
    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        return
    with open(out_path, "a", encoding="utf-8") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")


def mint(
    *,
    api_key: str,
    agent_id: str,
    reason: str,
    finding_id: str,
    sarif_digest: str,
    valid_seconds: int,
    supersedes: str | None,
    risk_snapshot: dict[str, Any] | None,
    approval: ApprovalContext,
) -> dict[str, str]:
    """Sign the risk-acceptance receipt and return the GitHub outputs.

    The sign call mirrors the SDK risk-acceptance guards: the extension fields
    are fenced to receipt_type=protectmcp:lifecycle:risk_acceptance, require
    compliance_mode=True, and sign under policy_decision='none' because the
    receipt records no policy evaluation.
    """
    import asqav

    asqav.init(api_key=api_key)
    agent = asqav.Agent.get(agent_id)

    response = agent.sign(
        "security:risk_acceptance",
        receipt_type=RECEIPT_TYPE,
        compliance_mode=True,
        policy_decision="none",
        approver_id=approval.approver,
        initiator_id=approval.initiator,
        acceptance_reason=reason,
        accepted_at=approval.accepted_at,
        supersedes=supersedes,
        sarif_digest=sarif_digest,
        finding_ref=finding_id,
        approval_ref=approval.approval_ref,
        risk_snapshot=risk_snapshot,
        valid_seconds=valid_seconds,
    )

    print(f"Signed risk-acceptance receipt: {response.signature_id}")
    print(f"Verify at: {response.verification_url}")

    outputs = {
        "signature-id": response.signature_id,
        "verification-url": response.verification_url,
    }
    _write_github_output(outputs)
    return outputs


def parse_manifest(text: str) -> list[str]:
    """Parse a suppression manifest into a list of receipt ids.

    Accepts a JSON array, a JSON object with a `receipts` key, or plain text
    with one id per line (blank lines and `#` comments skipped). Raises
    ValueError on an empty result so an empty manifest never passes silently.
    """
    text = text.strip()
    ids: list[str] = []
    if text.startswith("[") or text.startswith("{"):
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("receipts")
        if not isinstance(data, list):
            raise ValueError(
                "JSON manifest must be an array of ids or {\"receipts\": [...]}."
            )
        ids = [str(x).strip() for x in data if str(x).strip()]
    else:
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    if not ids:
        raise ValueError("suppression manifest contains no receipt ids.")
    return ids


def _fetch_receipt(signature_id: str) -> dict[str, Any] | None:
    """Fetch one receipt via the public verify endpoint.

    Returns a plain dict with the axes the audit consumes, or None when the
    receipt does not exist. Network and API failures propagate so the audit
    fails loud rather than passing on missing data.
    """
    import asqav
    from asqav import APIError

    try:
        result = asqav.verify_signature(signature_id)
    except APIError as exc:
        if exc.status_code == 404:
            return None
        raise
    detail = result.verification_detail
    return {
        "signature_id": result.signature_id,
        "verified": result.verified,
        "validation_label": detail.validation_label if detail else None,
        "payload": result.payload if isinstance(result.payload, dict) else None,
    }


def evaluate_audit(receipts: dict[str, dict[str, Any] | None]) -> list[str]:
    """Return the list of audit failures for the fetched manifest receipts.

    A manifest id fails when the receipt is missing, fails verification, is
    expired (the server-evaluated validation_label), references a receipt type
    other than risk_acceptance, or is named in the `supersedes` field of
    another fetched receipt. Supersession is evaluated client-side across the
    fetched set, on the payload's `supersedes` value.
    """
    failures: list[str] = []
    superseded_by: dict[str, str] = {}
    for sig_id, rec in receipts.items():
        if rec is None or not rec.get("payload"):
            continue
        target = rec["payload"].get("supersedes")
        if isinstance(target, str) and target:
            superseded_by[target] = sig_id

    for sig_id, rec in receipts.items():
        if rec is None:
            failures.append(f"{sig_id}: receipt not found.")
            continue
        label = rec.get("validation_label")
        if label == EXPIRED_LABEL:
            failures.append(f"{sig_id}: expired (validation_label={label}).")
        elif not rec.get("verified"):
            failures.append(f"{sig_id}: failed verification (validation_label={label}).")
        payload = rec.get("payload")
        if payload is not None:
            rtype = payload.get("receipt_type")
            if rtype != RECEIPT_TYPE:
                failures.append(
                    f"{sig_id}: receipt_type is {rtype!r}, expected {RECEIPT_TYPE}."
                )
        if sig_id in superseded_by:
            failures.append(f"{sig_id}: superseded by {superseded_by[sig_id]}.")
    return failures


def audit(*, api_key: str, manifest_path: str) -> list[str]:
    """Run the audit: fetch every manifest receipt, evaluate, return failures."""
    import asqav

    asqav.init(api_key=api_key)
    with open(manifest_path, encoding="utf-8") as fh:
        ids = parse_manifest(fh.read())
    receipts = {sig_id: _fetch_receipt(sig_id) for sig_id in ids}
    return evaluate_audit(receipts)


def _main_mint() -> int:
    api_key = os.environ.get("ASQAV_API_KEY", "")
    agent_id = os.environ.get("ASQAV_AGENT_ID", "")
    reason = os.environ.get("ASQAV_REASON", "").strip()
    finding_id = os.environ.get("ASQAV_FINDING_ID", "").strip()
    sarif_file = os.environ.get("ASQAV_SARIF_FILE", "").strip()
    expires = os.environ.get("ASQAV_EXPIRES", "").strip()
    supersedes = os.environ.get("ASQAV_SUPERSEDES", "").strip() or None
    snapshot_raw = os.environ.get("ASQAV_RISK_SNAPSHOT_JSON", "").strip()

    missing = [
        name
        for name, value in (
            ("asqav-api-key", api_key),
            ("asqav-agent-id", agent_id),
            ("reason", reason),
            ("finding-id", finding_id),
            ("sarif-file", sarif_file),
            ("expires", expires),
        )
        if not value
    ]
    if missing:
        print(f"ERROR: mint mode requires inputs: {', '.join(missing)}.", file=sys.stderr)
        return 2

    try:
        valid_seconds = parse_expires(expires)
        sarif_digest = compute_sarif_digest(sarif_file)
        risk_snapshot = json.loads(snapshot_raw) if snapshot_raw else None
        approval = read_approval_context()
        mint(
            api_key=api_key,
            agent_id=agent_id,
            reason=reason,
            finding_id=finding_id,
            sarif_digest=sarif_digest,
            valid_seconds=valid_seconds,
            supersedes=supersedes,
            risk_snapshot=risk_snapshot,
            approval=approval,
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the runner
        print(f"ERROR minting risk-acceptance receipt: {exc}", file=sys.stderr)
        return 1
    return 0


def _main_audit() -> int:
    api_key = os.environ.get("ASQAV_API_KEY", "")
    manifest_path = os.environ.get("ASQAV_SUPPRESSION_MANIFEST", "").strip()
    if not api_key or not manifest_path:
        print(
            "ERROR: audit mode requires asqav-api-key and suppression-manifest.",
            file=sys.stderr,
        )
        return 2
    try:
        failures = audit(api_key=api_key, manifest_path=manifest_path)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the runner
        print(f"ERROR auditing suppression manifest: {exc}", file=sys.stderr)
        return 1
    if failures:
        for line in failures:
            print(f"FAIL {line}", file=sys.stderr)
        print(
            f"{len(failures)} suppression(s) rest on an expired, superseded, "
            "or invalid risk-acceptance receipt. Remove the suppression or "
            "mint a new acceptance.",
            file=sys.stderr,
        )
        return 1
    print("All manifest receipts verified, unexpired, and unsuperseded.")
    return 0


def main() -> int:
    mode = os.environ.get("ASQAV_MODE", "mint").strip().lower()
    if mode == "mint":
        return _main_mint()
    if mode == "audit":
        return _main_audit()
    print(f"ERROR: unknown mode {mode!r} (expected mint or audit).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
