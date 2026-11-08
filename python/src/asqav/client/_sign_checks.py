"""Client-side digests and validators for the fields a signing call carries."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ._errors import AsqavValidationError
from ._vocabulary import (
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
