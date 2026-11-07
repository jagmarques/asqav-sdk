"""Validation of signing inputs: risk, code authorship and digests."""

from __future__ import annotations

import hashlib
import re
from typing import Any, TypedDict

from ..counterparty import ACKNOWLEDGMENT_RECEIPT_TYPE, CounterpartyBinding
from ._errors import AsqavValidationError
from ._vocabulary import (
    _CODE_AUTHORSHIP_CHANGE_CLASS_NAMESPACE,
    _RISK_SNAPSHOT_NUMERIC_KEYS,
    CODE_AUTHORSHIP_DOCS_URL,
    RISK_ACCEPTANCE_DOCS_URL,
)


def _compute_action_ref(action_type: str, context: dict[str, Any] | None) -> str:
    """Client-side ``sha256:<hex>`` of the canonical Action object.

    The Action shape mirrors `core/canonical.py:hash_action` on the cloud
    so the SDK and cloud agree byte-for-byte under JCS encoding.
    """
    from ..canonicalize import canonicalize_action

    return "sha256:" + hashlib.sha256(canonicalize_action(action_type, context)).hexdigest()


#: Wire format for caller-supplied sha256 digest fields (matches helper output).
_SHA256_HEX_RE = re.compile(r"^sha256:[a-f0-9]{64}$")


#: Bare 64-hex, no prefix; the controls_evaluated quorum.attestation_hash form.
_BARE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


#: Server-built control keys an enumerable controls_evaluated block may carry;
#: mirrors the cloud false_control_attestation_guard allowlist (conformance.py).
_ALLOWED_CONTROL_KEYS: frozenset[str] = frozenset(
    {
        "emergency_halt",
        "delegation_scope",
        "quorum",
        "mandate",
        "policy",
        "content_scan",
        "result",
    }
)


#: Wire format for ``tool_fingerprint``: 32 bare lowercase hex (SHA-256[:32]);
#: unlike the ``sha256:<hex>`` form of the other digests.
_TOOL_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{32}$")


class RiskSnapshot(TypedDict, total=False):
    """Producer-asserted snapshot of third-party risk signals.
    Asqav vouches for none; a numeric needs ``snapshot_source``."""

    #: RFC 3339 read-time of the signals (REQUIRED). Producer-asserted.
    snapshot_at: str
    #: Free-text label naming the source. REQUIRED when any numeric is present.
    snapshot_source: str
    #: EPSS probability as a STRING (floats rejected). Producer-asserted.
    epss: str
    #: CVSS base score as a STRING. Producer-asserted; not computed by Asqav.
    cvss: str
    #: CVSS vector string snapshot. Producer-asserted; not parsed by Asqav.
    cvss_vector: str
    #: Whether the CVE was in the named KEV catalog at snapshot_at, as observed.
    kev_listed: bool
    #: CVE ids the snapshot pertains to; non-empty list[str], entries <= 128.
    cve_ids: list[str]


class AuthoredBy(TypedDict, total=False):
    """Producer-asserted author of a code change (cloud AuthoredBy model).
    A ``model_id`` / ``model_version`` requires ``attestation_source``."""

    #: Human author identity (free-text id / email). Producer-asserted.
    human_id: str
    #: Model id if a model authored the change. Requires attestation_source.
    model_id: str
    #: Model version if a model authored the change; needs attestation_source.
    model_version: str
    #: Free-text label naming where the machine-authorship claim came from.
    #: REQUIRED when model_id / model_version is present.
    attestation_source: str


def _validate_risk_snapshot(rs: Any) -> None:
    """Mirror cloud RiskSnapshot._validate_snapshot_shape + _check_risk_snapshot.
    Rejects a malformed snapshot, numeric without source, or bad cve_ids."""
    if not isinstance(rs, dict):
        raise AsqavValidationError(
            "risk_snapshot_not_object: risk_snapshot must be an object of "
            f"producer-asserted snapshot signals (got: {repr(rs)[:64]})",
            docs_url=RISK_ACCEPTANCE_DOCS_URL,
        )
    if not rs.get("snapshot_at") or not isinstance(rs.get("snapshot_at"), str):
        raise AsqavValidationError(
            "risk_snapshot_missing_snapshot_at: snapshot_at (RFC 3339 string) "
            "is required on a risk_snapshot.",
            docs_url=RISK_ACCEPTANCE_DOCS_URL,
        )
    source = rs.get("snapshot_source")
    numeric_present = any(rs.get(k) is not None for k in _RISK_SNAPSHOT_NUMERIC_KEYS)
    if numeric_present and (not isinstance(source, str) or not source):
        raise AsqavValidationError(
            "risk_snapshot_numeric_requires_snapshot_source: a populated "
            "risk_snapshot signal requires snapshot_source so the value is "
            "never read as an Asqav-derived score.",
            docs_url=RISK_ACCEPTANCE_DOCS_URL,
        )
    for _k in ("epss", "cvss"):
        _v = rs.get(_k)
        if _v is not None and not isinstance(_v, str):
            raise AsqavValidationError(
                f"risk_snapshot_{_k}_not_string: numeric risk signals MUST be "
                f"strings on the wire, never floats (got: {repr(_v)[:64]}).",
                docs_url=RISK_ACCEPTANCE_DOCS_URL,
            )
    cve_ids = rs.get("cve_ids")
    if cve_ids is not None:
        if not isinstance(cve_ids, list) or len(cve_ids) == 0:
            raise AsqavValidationError(
                "risk_snapshot_cve_ids_must_be_non_empty_list: pass a "
                "non-empty list of strings or omit cve_ids.",
                docs_url=RISK_ACCEPTANCE_DOCS_URL,
            )
        for _entry in cve_ids:
            if not isinstance(_entry, str) or not _entry or len(_entry) > 128:
                raise AsqavValidationError(
                    "risk_snapshot_cve_ids_entry_invalid: each cve id must be "
                    f"a non-empty string of length <= 128 (got: {repr(_entry)[:64]}).",
                    docs_url=RISK_ACCEPTANCE_DOCS_URL,
                )


#: The risk-acceptance signed-payload extension fields, fenced to the
#: protectmcp:lifecycle:risk_acceptance receipt type (cloud SignRequest).
_RISK_ACCEPTANCE_RECEIPT_TYPE = "protectmcp:lifecycle:risk_acceptance"


def _validate_risk_acceptance(
    *,
    receipt_type: str | None,
    compliance_mode: bool,
    policy_decision: str,
    approver_id: str | None,
    initiator_id: str | None,
    acceptance_reason: str | None,
    accepted_at: str | None,
    supersedes: str | None,
    sarif_digest: str | None,
    finding_ref: str | None,
    approval_ref: str | None,
    risk_snapshot: Any,
) -> None:
    """Mirror cloud SignRequest._validate_risk_acceptance_extensions guards.
    Needs sha256 sarif_digest, approver_id, reason, policy none, compliance."""
    if sarif_digest is not None and not _SHA256_HEX_RE.match(sarif_digest):
        raise AsqavValidationError(
            "sarif_digest_not_sha256_wire_form: must look like "
            f"'sha256:<64 lowercase hex>' (got: {repr(sarif_digest)[:64]}).",
            docs_url=RISK_ACCEPTANCE_DOCS_URL,
        )
    if risk_snapshot is not None:
        _validate_risk_snapshot(risk_snapshot)
    risk_fields_present = any(
        v is not None
        for v in (
            approver_id,
            initiator_id,
            acceptance_reason,
            accepted_at,
            supersedes,
            sarif_digest,
            finding_ref,
            approval_ref,
            risk_snapshot,
        )
    )
    if risk_fields_present and receipt_type != _RISK_ACCEPTANCE_RECEIPT_TYPE:
        raise AsqavValidationError(
            "risk_acceptance_fields_require_risk_acceptance_receipt: "
            "approver_id / initiator_id / acceptance_reason / accepted_at / "
            "supersedes / sarif_digest / finding_ref / approval_ref / "
            "risk_snapshot are only valid on "
            f"receipt_type={_RISK_ACCEPTANCE_RECEIPT_TYPE}.",
            docs_url=RISK_ACCEPTANCE_DOCS_URL,
        )
    if receipt_type == _RISK_ACCEPTANCE_RECEIPT_TYPE:
        if not (approver_id and acceptance_reason):
            raise AsqavValidationError(
                "risk_acceptance_missing_required_field: "
                f"receipt_type={_RISK_ACCEPTANCE_RECEIPT_TYPE} requires "
                "approver_id and acceptance_reason.",
                docs_url=RISK_ACCEPTANCE_DOCS_URL,
            )
        if policy_decision != "none":
            raise AsqavValidationError(
                "risk_acceptance_requires_no_policy_decision: "
                f"receipt_type={_RISK_ACCEPTANCE_RECEIPT_TYPE} records no policy "
                "evaluation; sign it with policy_decision='none'.",
                docs_url=RISK_ACCEPTANCE_DOCS_URL,
            )
        if not compliance_mode:
            raise AsqavValidationError(
                "risk_acceptance_requires_compliance_mode: "
                f"receipt_type={_RISK_ACCEPTANCE_RECEIPT_TYPE} fields project "
                "into the signed receipt only under compliance_mode=True; "
                "passing compliance_mode=False would silently drop them.",
                docs_url=RISK_ACCEPTANCE_DOCS_URL,
            )
        if initiator_id is not None and approver_id is not None and initiator_id == approver_id:
            raise AsqavValidationError(
                "risk_acceptance_self_approval_guard: initiator_id equals "
                "approver_id; a self-approved risk acceptance is incoherent "
                f"(got: {repr(approver_id)[:64]})",
                docs_url=RISK_ACCEPTANCE_DOCS_URL,
            )


#: The code-authorship signed-payload extension fields, fenced to the
#: protectmcp:lifecycle:code_authorship receipt type (cloud SignRequest).
_CODE_AUTHORSHIP_RECEIPT_TYPE = "protectmcp:lifecycle:code_authorship"


def _validate_counterparty_binding(
    binding: "CounterpartyBinding | dict[str, Any] | None",
    *,
    receipt_type: str | None,
    compliance_mode: bool,
) -> dict[str, Any] | None:
    """Fence the binding to compliance acknowledgment receipts and return its wire form.

    Mirrors the cloud's admit_counterparty_binding: a binding outside an
    acknowledgment, or without compliance_mode, is refused before any network call.
    """
    if binding is None:
        return None
    if receipt_type != ACKNOWLEDGMENT_RECEIPT_TYPE or not compliance_mode:
        raise ValueError(
            "counterparty_binding_requires_acknowledgment_receipt: counterparty_binding is "
            f"only valid on receipt_type={ACKNOWLEDGMENT_RECEIPT_TYPE} with compliance_mode=True."
        )
    wire = binding.to_wire() if isinstance(binding, CounterpartyBinding) else dict(binding)
    for key in ("envelope_hash", "receipt_ref"):
        if not isinstance(wire.get(key), str) or not wire[key]:
            raise ValueError(f"counterparty_binding_malformed: {key} must be a non-empty string.")
    return wire


def _validate_code_authorship(
    *,
    receipt_type: str | None,
    compliance_mode: bool,
    policy_decision: str,
    repo_ref: str | None,
    commit_sha: str | None,
    base_sha: str | None,
    change_digest: str | None,
    change_ref: str | None,
    change_approval_ref: str | None,
    change_class: str | None,
    authored_by: Any,
) -> None:
    """Mirror cloud SignRequest._validate_code_authorship_extensions guards.
    Needs repo_ref, commit_sha, policy none, compliance; checks digest/class."""
    if change_digest is not None and not _SHA256_HEX_RE.match(change_digest):
        raise AsqavValidationError(
            "change_digest_not_sha256_wire_form: must look like "
            f"'sha256:<64 lowercase hex>' (got: {repr(change_digest)[:64]}).",
            docs_url=CODE_AUTHORSHIP_DOCS_URL,
        )
    if change_class is not None and change_class not in _CODE_AUTHORSHIP_CHANGE_CLASS_NAMESPACE:
        raise AsqavValidationError(
            "code_authorship_change_class_invalid: change_class must be one of "
            "read|write|delete|execute|deploy "
            f"(got: {repr(change_class)[:64]}).",
            docs_url=CODE_AUTHORSHIP_DOCS_URL,
        )
    if authored_by is not None:
        _validate_authored_by(authored_by)
    code_fields_present = any(
        v is not None
        for v in (
            repo_ref,
            commit_sha,
            base_sha,
            change_digest,
            change_ref,
            change_approval_ref,
            change_class,
            authored_by,
        )
    )
    if code_fields_present and receipt_type != _CODE_AUTHORSHIP_RECEIPT_TYPE:
        raise AsqavValidationError(
            "code_authorship_fields_require_code_authorship_receipt: "
            "repo_ref / commit_sha / base_sha / change_digest / change_ref / "
            "change_approval_ref / change_class / authored_by are only valid on "
            f"receipt_type={_CODE_AUTHORSHIP_RECEIPT_TYPE}.",
            docs_url=CODE_AUTHORSHIP_DOCS_URL,
        )
    if receipt_type == _CODE_AUTHORSHIP_RECEIPT_TYPE:
        if not (repo_ref and commit_sha):
            raise AsqavValidationError(
                "code_authorship_missing_required_field: "
                f"receipt_type={_CODE_AUTHORSHIP_RECEIPT_TYPE} requires "
                "repo_ref and commit_sha.",
                docs_url=CODE_AUTHORSHIP_DOCS_URL,
            )
        if policy_decision != "none":
            raise AsqavValidationError(
                "code_authorship_requires_no_policy_decision: "
                f"receipt_type={_CODE_AUTHORSHIP_RECEIPT_TYPE} records no policy "
                "evaluation; sign it with policy_decision='none'.",
                docs_url=CODE_AUTHORSHIP_DOCS_URL,
            )
        if not compliance_mode:
            raise AsqavValidationError(
                "code_authorship_requires_compliance_mode: "
                f"receipt_type={_CODE_AUTHORSHIP_RECEIPT_TYPE} fields project "
                "into the signed receipt only under compliance_mode=True; "
                "passing compliance_mode=False would silently drop them.",
                docs_url=CODE_AUTHORSHIP_DOCS_URL,
            )


def _validate_authored_by(ab: Any) -> None:
    """Mirror the cloud AuthoredBy validator (routes/agents.py).
    A model_id / model_version requires authored_by.attestation_source."""
    if not isinstance(ab, dict):
        raise AsqavValidationError(
            "authored_by_not_object: authored_by must be an object naming the "
            f"producer-asserted author of the change (got: {repr(ab)[:64]}).",
            docs_url=CODE_AUTHORSHIP_DOCS_URL,
        )
    model_present = bool(ab.get("model_id")) or bool(ab.get("model_version"))
    source = ab.get("attestation_source")
    if model_present and (not isinstance(source, str) or not source):
        raise AsqavValidationError(
            "authored_by_model_requires_attestation_source: a populated "
            "model_id / model_version requires authored_by.attestation_source "
            "so the machine-authorship claim is never read as Asqav-verified.",
            docs_url=CODE_AUTHORSHIP_DOCS_URL,
        )


def _compute_tool_fingerprint(tool_name: str, tool_schema: dict[str, Any] | None) -> str:
    """32-hex fingerprint over `{tool_name, schema}` (NSA CSI 6030316-26).
    First 32 hex chars of the JCS (RFC 8785) SHA-256, bare."""
    from ..canonicalize import canonicalize

    payload = canonicalize({"tool_name": tool_name, "schema": tool_schema or {}})
    return hashlib.sha256(payload).hexdigest()[:32]


def _compute_config_manifest_digest(manifest: dict[str, Any]) -> str:
    """Compute the canonical ``sha256:<hex>`` of a configuration manifest.
    JCS (RFC 8785) digest; NSA CSI 6030316-26 audits capability creep."""
    from ..canonicalize import canonicalize

    return "sha256:" + hashlib.sha256(canonicalize(manifest)).hexdigest()


def _compute_cve_inventory_digest(cve_list: list[Any]) -> str:
    """Compute the canonical ``sha256:<hex>`` of a CVE inventory snapshot.
    JCS (RFC 8785) digest of the record list (NSA CSI 6030316-26)."""
    from ..canonicalize import canonicalize

    return "sha256:" + hashlib.sha256(canonicalize(cve_list)).hexdigest()


def _generate_nonce() -> str:
    """Return a 24-hex-char (12 random bytes) value for the `nonce` wire field.
    The cloud answers 409 while a live signature holds it; expiry frees it."""
    import secrets as _secrets

    return _secrets.token_hex(12)
