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
