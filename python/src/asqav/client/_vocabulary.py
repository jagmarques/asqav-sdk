"""Receipt namespaces, decision maps and payload field lookups."""

from __future__ import annotations

from typing import Any, Literal

from .._sdk_constants import SDK_CONSTANTS
from ..verifier.verify_receipt import SKEW_BOUND_SECONDS as SKEW_BOUND_SECONDS

# Receipt type namespace mirrored from the cloud SignRequest validator.
RECEIPT_TYPE_NAMESPACE: frozenset[str] = frozenset(
    {
        "protectmcp:decision",
        "protectmcp:restraint",
        "protectmcp:lifecycle",
        "protectmcp:lifecycle:configuration_change",
        #: risk-acceptance / exception receipt; no-policy lifecycle opt-out.
        "protectmcp:lifecycle:risk_acceptance",
        #: code-authorship receipt; producer-asserted record of who authored it.
        "protectmcp:lifecycle:code_authorship",
        "protectmcp:acknowledgment",
        "protectmcp:observation",
        #: observation receipts with a result_digest use :result_bound
        "protectmcp:observation:result_bound",
    }
)


#: Docs page anchoring the risk-acceptance validation errors (rule 3.9).
RISK_ACCEPTANCE_DOCS_URL: str = SDK_CONSTANTS["RISK_ACCEPTANCE_DOCS_URL"]


#: Docs page anchoring the code-authorship validation errors (rule 3.9).
CODE_AUTHORSHIP_DOCS_URL: str = "https://asqav.com/docs/code-authorship-receipt"


#: Closed vocabulary for code_authorship `change_class`. Mirrors the cloud
#: SignRequest closed set; anything outside is rejected client-side.
_CODE_AUTHORSHIP_CHANGE_CLASS_NAMESPACE: frozenset[str] = frozenset(
    {
        "read",
        "write",
        "delete",
        "execute",
        "deploy",
    }
)


#: Numeric risk-snapshot signals whose presence demands a snapshot_source label.
#: Mirrors cloud core/conformance.py _RISK_SNAPSHOT_NUMERIC_KEYS.
_RISK_SNAPSHOT_NUMERIC_KEYS: tuple[str, ...] = (
    "epss",
    "cvss",
    "cvss_vector",
    "kev_listed",
)


#: Canonical `incident_class` vocabulary under DORA Annex II field 3.23.
#: Cloud rejects anything outside this set for receipts bound to DORA.
DORA_INCIDENT_CLASS_NAMESPACE: frozenset[str] = frozenset(
    {
        "cybersecurity_related",
        "process_failure",
        "system_failure",
        "external_event",
        "payment_related",
        "other",
    }
)


#: HIPAA Security Rule incident_class vocabulary (45 CFR 164.304).
HIPAA_INCIDENT_CLASS_NAMESPACE: frozenset[str] = frozenset({"hipaa_security_incident"})


#: Union of every canonical incident_class token the SDK accepts.
INCIDENT_CLASS_NAMESPACE: frozenset[str] = (
    DORA_INCIDENT_CLASS_NAMESPACE | HIPAA_INCIDENT_CLASS_NAMESPACE
)


#: Sandbox-state enum: {enabled, disabled, unavailable}.
SANDBOX_STATE_NAMESPACE: frozenset[str] = frozenset({"enabled", "disabled", "unavailable"})


#: Durable-anchoring witnesses Asqav ships; the cloud `witness_policy` accepts
#: only this set (`rekor` is rejected). Mirrors the governance.json contract.
WITNESS_NAMESPACE: frozenset[str] = frozenset({"rfc3161", "opentimestamps"})


#: Producer-side `capture_topology` mirrored from the cloud SignRequest Literal.
#: ADVISORY: the server stamps the authoritative value from the ingress route.
CAPTURE_TOPOLOGY_NAMESPACE: frozenset[str] = frozenset(
    {
        "in_process_sdk",
        "network_proxy",
        "browser_extension",
        "mcp_proxy",
        "passive_telemetry",
        "github_sha_pull",
    }
)


#: Internal `policy_decision` vocabulary; `none` is the lifecycle opt-out
#: gated to receipt_type startswith `protectmcp:lifecycle`.
PolicyDecision = Literal["permit", "deny", "rate_limit", "none"]


#: Wire-shape `decision` vocabulary; `observation` is reserved to lifecycle
#: receipts emitted under `policy_decision="none"`.
Decision = Literal["allow", "deny", "rate_limit", "observation"]


#: Capture-topology vocabulary as a Literal for typed callers.
CaptureTopology = Literal[
    "in_process_sdk",
    "network_proxy",
    "browser_extension",
    "mcp_proxy",
    "passive_telemetry",
    "github_sha_pull",
]


#: vocabulary | wire `{allow, deny, rate_limit, observation}` keyed by
#: internal `{permit, allow, deny, rate_limit, none}`. Unknown -> "deny".
DECISION_MAP: dict[str, str] = {
    "permit": "allow",
    "allow": "allow",
    "deny": "deny",
    "rate_limit": "rate_limit",
    "none": "observation",
}


def _map_policy_decision_to_decision(policy_decision: str | None) -> str:
    """Translate internal `policy_decision` to spec-shape `decision`.

    Used when emitting compliance receipts. Falls back to "deny" so a
    misconfigured caller never publishes an out-of-vocabulary token.
    """
    return DECISION_MAP.get((policy_decision or "").lower(), "deny")

    # Read a string ``key`` from the sign response ``payload``, else None.


def _payload_field(data: dict[str, Any], key: str) -> str | None:
    payload = data.get("payload")
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if isinstance(value, str) else None

    # Top-level ``action_ref`` wins; compliance-mode clouds nest it in payload.


def _resolve_action_ref(data: dict[str, Any]) -> str | None:
    top = data.get("action_ref")
    if top is not None:
        return top
    return _payload_field(data, "action_ref")

    # Top-level casing wins; fall back to the copy nested under payload.


def _resolve_previous_receipt_hash(data: dict[str, Any]) -> str | None:
    return (
        data.get("previousReceiptHash")
        or data.get("previous_receipt_hash")
        or _payload_field(data, "previousReceiptHash")
        or _payload_field(data, "previous_receipt_hash")
    )


#: REQUIRED fields on a Compliance Receipt envelope (wire form): ``type``
#: discriminator, ``decision`` verdict. Used by `verify_compliance_receipt`.
_COMPLIANCE_REQUIRED_FIELDS: tuple[str, ...] = (
    "type",
    "issuer_id",
    "agent_id",
    "action_ref",
    "payload_digest",
    "policy_digest",
    "previousReceiptHash",
    "decision",
    "issued_at",
)
