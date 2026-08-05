"""SDK half of the un-bypassable code-authorship path.

POST /v1/code-authorship is the authoritative ingress: the client supplies an
ADVISORY change digest (sha256 of ``git diff base..head``) and the server
re-fetches the commit, recomputes the canonical diff, and signs an in-toto
Statement whose ``subject[0].digest.sha256`` is the SERVER-recomputed hash.
The client digest is never trusted. ``digest_match`` only reports whether the
advisory value agreed with the server's.

The authoritative capture layer for a code-authorship receipt is
``github_sha_pull``. A receipt whose capture layer is ``in_process_sdk`` or
``passive_telemetry`` is a client self-report and can NEVER be an authoritative
decision receipt (observation only), mirroring the cloud's
``observation_decision_not_allowed`` rule.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from typing import Any

#: API-key scope required to call POST /v1/code-authorship.
CODE_AUTHORSHIP_WRITE_SCOPE: str = "code_authorship:write"

#: Endpoint path (joined onto the configured API base).
CODE_AUTHORSHIP_PATH: str = "/code-authorship"

#: predicateType of the authoritative code-authorship in-toto Statement.
CODE_AUTHORSHIP_PREDICATE_TYPE: str = "https://asqav.com/attestation/code-authorship/v1"

#: in-toto Statement v1 type carried by the envelope.
INTOTO_STATEMENT_TYPE: str = "https://in-toto.io/Statement/v1"

#: asset_class the server stamps on the code-authorship predicate.
CODE_AUTHORSHIP_ASSET_CLASS: str = "code"

#: The ONLY capture layer that makes a code-authorship receipt authoritative.
#: The server stamps it after re-fetching the commit and recomputing the diff.
AUTHORITATIVE_CAPTURE_LAYER: str = "github_sha_pull"

#: Capture layers that are client self-reports. A code-authorship receipt
#: carrying one of these is observation only and never an authoritative
#: decision receipt.
OBSERVATION_ONLY_CAPTURE_LAYERS: frozenset[str] = frozenset(
    {"in_process_sdk", "passive_telemetry"}
)

#: Verdict tokens for :func:`verify_code_authorship_envelope`.
VERDICT_PASS: str = "PASS"
VERDICT_REJECT: str = "REJECT"


def compute_advisory_digest(
    base_sha: str | None,
    head_sha: str,
    diff_text: str | None = None,
) -> str:
    """Compute the ADVISORY change digest ``sha256:<hex>``.

    The advisory digest is sha256 of ``git diff base..head``. When ``diff_text``
    is supplied it is hashed directly (tests, or callers that already hold the
    diff). With no base the bare head sha is hashed so the field is always a
    well-formed ``sha256:<64 hex>`` existence proof. The server recomputes its
    own digest and signs that. This value only lets the server report
    ``digest_match``.
    """
    if diff_text is None:
        if base_sha:
            out = subprocess.run(
                ["git", "diff", f"{base_sha}..{head_sha}"],
                check=True,
                capture_output=True,
            )
            diff_text = out.stdout.decode("utf-8", errors="replace")
        else:
            diff_text = head_sha
    return "sha256:" + hashlib.sha256(diff_text.encode("utf-8")).hexdigest()


    # Strip a ``sha256:`` prefix to the bare hex, or None when absent.
def _bare_hex(digest: str | None) -> str | None:
    if not isinstance(digest, str) or not digest:
        return None
    return digest.split(":", 1)[-1]


@dataclass
class CodeAuthorshipResult:
    """Parsed response from POST /v1/code-authorship.

    ``envelope`` is the server-signed in-toto Statement. ``subject_digest`` is
    the SERVER-recomputed diff hash bound into ``subject[0].digest.sha256``.
    ``digest_match`` reports whether the advisory client digest agreed.
    """

    envelope: dict[str, Any]
    receipt: dict[str, Any]
    kid: str | None
    jwks_url: str | None
    server_digest: str | None
    digest_match: bool
    subject_digest: str | None
    capture_layer: str | None
    asset_class: str | None
    advisory_client_digest: str | None
    raw: dict[str, Any] = field(default_factory=dict)

        # Project the wire response into a result, reading the Statement fields.
    @classmethod
    def from_response(cls, data: dict[str, Any]) -> "CodeAuthorshipResult":
        envelope = data.get("envelope") if isinstance(data.get("envelope"), dict) else {}
        receipt = data.get("receipt") if isinstance(data.get("receipt"), dict) else {}

        subject_digest: str | None = None
        subject = envelope.get("subject")
        if isinstance(subject, list) and subject and isinstance(subject[0], dict):
            digest_obj = subject[0].get("digest")
            if isinstance(digest_obj, dict):
                value = digest_obj.get("sha256")
                subject_digest = value if isinstance(value, str) else None

        predicate = envelope.get("predicate") if isinstance(envelope.get("predicate"), dict) else {}
        capture_layer = predicate.get("capture_layer")
        asset_class = predicate.get("asset_class")
        advisory = predicate.get("advisory_client_digest")

        return cls(
            envelope=envelope,
            receipt=receipt,
            kid=data.get("kid") if isinstance(data.get("kid"), str) else None,
            jwks_url=data.get("jwks_url") if isinstance(data.get("jwks_url"), str) else None,
            server_digest=(
                data.get("server_digest") if isinstance(data.get("server_digest"), str) else None
            ),
            digest_match=bool(data.get("digest_match")),
            subject_digest=subject_digest,
            capture_layer=capture_layer if isinstance(capture_layer, str) else None,
            asset_class=asset_class if isinstance(asset_class, str) else None,
            advisory_client_digest=advisory if isinstance(advisory, str) else None,
            raw=data,
        )

        # True when the bound subject digest equals the server-recomputed digest.
    @property
    def subject_matches_server(self) -> bool:
        if self.subject_digest is None or self.server_digest is None:
            return False
        return self.subject_digest == _bare_hex(self.server_digest)


def submit_code_authorship(
    *,
    repo: str,
    commit_sha: str,
    base_sha: str | None = None,
    change_digest: str | None = None,
    change_class: str | None = None,
    author: str | None = None,
    anchor: str | None = None,
) -> CodeAuthorshipResult:
    """POST an advisory code-authorship record to /v1/code-authorship.

    ``change_digest`` is ADVISORY (sha256 of ``git diff base..head``). The
    server recomputes the authoritative digest and signs it. Requires
    ``asqav.init(api_key=...)`` with a key holding the ``code_authorship:write``
    scope. Returns the parsed authoritative envelope.
    """
    from . import client as _client

    body: dict[str, Any] = {"repo": repo, "commit_sha": commit_sha}
    if base_sha:
        body["base_sha"] = base_sha
    if change_digest:
        body["change_digest"] = change_digest
    if change_class:
        body["change_class"] = change_class
    if author:
        body["author"] = author
    if anchor:
        body["anchor"] = anchor

    data = _client._post(CODE_AUTHORSHIP_PATH, body)
    return CodeAuthorshipResult.from_response(data)


@dataclass
class CodeAuthorshipVerification:
    """Outcome of the code-authorship envelope check.

    ``authoritative`` is true only when the capture layer is ``github_sha_pull``.
    ``observation_only`` is true when the capture layer is a client self-report
    (``in_process_sdk`` / ``passive_telemetry``), which can never be authoritative.
    """

    verdict: str
    authoritative: bool
    observation_only: bool
    capture_layer: str | None
    subject_digest: str | None
    reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == VERDICT_PASS


def verify_code_authorship_envelope(envelope: dict[str, Any]) -> CodeAuthorshipVerification:
    """Verify the code-authorship structural + capture-layer invariant.

    Checks the in-toto Statement shape (``_type``, ``predicateType``), that
    ``subject[0].digest.sha256`` is present, and the capture-layer rule:
    ``github_sha_pull`` is authoritative, ``in_process_sdk`` / ``passive_telemetry``
    are observation only and never authoritative. The cryptographic DSSE
    signature is verified by the standalone verifier or the hosted /verify. The
    cloud is the authoritative verifier and this helper is a convenience.
    """
    reasons: list[str] = []

    if not isinstance(envelope, dict):
        return CodeAuthorshipVerification(
            verdict=VERDICT_REJECT,
            authoritative=False,
            observation_only=False,
            capture_layer=None,
            subject_digest=None,
            reasons=["envelope_not_an_object"],
        )

    if envelope.get("_type") != INTOTO_STATEMENT_TYPE:
        reasons.append("code_authorship_envelope_not_intoto_statement")
    if envelope.get("predicateType") != CODE_AUTHORSHIP_PREDICATE_TYPE:
        reasons.append("code_authorship_wrong_predicate_type")

    subject_digest: str | None = None
    subject = envelope.get("subject")
    if isinstance(subject, list) and subject and isinstance(subject[0], dict):
        digest_obj = subject[0].get("digest")
        if isinstance(digest_obj, dict) and isinstance(digest_obj.get("sha256"), str):
            subject_digest = digest_obj["sha256"]
    if not subject_digest:
        reasons.append("code_authorship_missing_subject_digest")

    predicate = envelope.get("predicate") if isinstance(envelope.get("predicate"), dict) else {}
    capture_layer = predicate.get("capture_layer")
    capture_layer = capture_layer if isinstance(capture_layer, str) else None

    observation_only = capture_layer in OBSERVATION_ONLY_CAPTURE_LAYERS
    authoritative = capture_layer == AUTHORITATIVE_CAPTURE_LAYER

    if observation_only:
        reasons.append("observation_capture_layer_not_authoritative")
    elif capture_layer is None:
        reasons.append("code_authorship_missing_capture_layer")
    elif not authoritative:
        reasons.append("code_authorship_capture_layer_not_github_sha_pull")

    verdict = VERDICT_PASS if not reasons else VERDICT_REJECT
    return CodeAuthorshipVerification(
        verdict=verdict,
        authoritative=authoritative,
        observation_only=observation_only,
        capture_layer=capture_layer,
        subject_digest=subject_digest,
        reasons=reasons,
    )
