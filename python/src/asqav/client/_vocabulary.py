"""Receipt namespaces and decision vocabularies."""

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
RISK_ACCEPTANCE_DOCS_URL: str = "https://asqav.com/docs/risk-acceptance-receipt"

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
