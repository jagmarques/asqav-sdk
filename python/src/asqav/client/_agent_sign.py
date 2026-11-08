"""Agent.sign: one signed action."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from math import ceil
from typing import Any

from ..counterparty import CounterpartyBinding
from . import _transport
from ._common import logger
from ._errors import require_field
from ._models import SignatureResponse, _parse_bitcoin_anchor
from ._sign_body import (
    _build_sign_body,
    _digest_result_value,
    _prepare_sign_context,
    _sign_body_tap,
    _validate_sign_anchoring,
    _validate_sign_extensions,
    _validate_sign_vocabulary,
)
from ._sign_checks import (
    AuthoredBy,
    RiskSnapshot,
    _compute_action_ref,
    _compute_tool_fingerprint,
    _generate_nonce,
    _validate_code_authorship,
    _validate_counterparty_binding,
    _validate_risk_acceptance,
)
from ._vocabulary import (
    _map_policy_decision_to_decision,
    _resolve_action_ref,
    _resolve_previous_receipt_hash,
)


class _AgentSign:
    def sign(
        self,
        action_type: str,
        context: dict[str, Any] | None = None,
        system_prompt_hash: str | None = None,
        model_params: dict | None = None,
        tool_inputs_hash: str | None = None,
        trace_id: str | None = None,
        parent_id: str | None = None,
        counterparty: dict[str, Any] | None = None,
        tool_name: str | None = None,
        model_name: str | None = None,
        *,
        context_schema: "dict[str, Any] | Callable[[dict[str, Any]], None] | None" = None,
        co_signers: list[str] | None = None,
        mandate_id: str | None = None,
        user_intent: dict[str, Any] | None = None,
        # nonce: per-call token; the cloud answers 409 while a live receipt
        # for this agent in this org holds it (python/README.md, nonce)
        nonce: str | None = None,
        valid_seconds: int | None = None,
        expected_executor_pubkey_b64: str | None = None,
        compliance_mode: bool = True,
        receipt_type: str | None = None,
        action_ref: str | None = None,
        payload_digest: str | dict[str, Any] | None = None,
        issuer_id: str | None = None,
        iteration_id: str | None = None,
        sandbox_state: str | None = None,
        risk_class: str | None = None,
        incident_class: str | list[str] | None = None,
        reason: str | None = None,
        policy_decision: str = "permit",
        capture_topology: str | None = None,
        result_digest: str | None = None,
        result: bytes | str | dict | list | None = None,
        expires_at: str | float | None = None,
        tool_schema: dict[str, Any] | None = None,
        tool_fingerprint: str | None = None,
        config_manifest_digest: str | None = None,
        cve_inventory_digest: str | None = None,
        executable_hash: str | None = None,
        sbom_digest: str | None = None,
        slsa_provenance_pointer: str | None = None,
        supply_chain_pointer: str | None = None,
        mitre_techniques: list[str] | None = None,
        mitre_atlas: list[str] | None = None,
        owasp_llm_top10: list[str] | None = None,
        owasp_agentic_top10: list[str] | None = None,
        nist_ai_rmf: list[str] | None = None,
        iso_42001: list[str] | None = None,
        eu_ai_act_articles: list[str] | None = None,
        rfc3161_timestamp: str | None = None,
        witness_policy: dict[str, Any] | None = None,
        approver_id: str | None = None,
        initiator_id: str | None = None,
        acceptance_reason: str | None = None,
        accepted_at: str | None = None,
        supersedes: str | None = None,
        sarif_digest: str | None = None,
        finding_ref: str | None = None,
        approval_ref: str | None = None,
        risk_snapshot: "RiskSnapshot | dict[str, Any] | None" = None,
        invocation_ref: str | None = None,
        repo_ref: str | None = None,
        commit_sha: str | None = None,
        base_sha: str | None = None,
        change_digest: str | None = None,
        change_ref: str | None = None,
        change_approval_ref: str | None = None,
        change_class: str | None = None,
        authored_by: "AuthoredBy | dict[str, Any] | None" = None,
        counterparty_binding: "CounterpartyBinding | dict[str, Any] | None" = None,
    ) -> SignatureResponse:
        """Sign one action; the cloud creates the signature with ML-DSA.

        Each argument is described in python/docs/sign-arguments.md.
        """
        params = dict(locals())
        del params["self"]
        _check_sign_params(params)
        body = _sign_request_body(params, agent_id=self.agent_id, session_id=self._session_id)
        tap = _sign_body_tap.get()
        if tap is not None:
            tap(body)
        data = _transport._post(f"/agents/{self.agent_id}/sign", body)
        response = _signature_response(data)
        try:
            from .. import hooks as _hooks_mod

            _hooks_mod._dispatch_after(params["action_type"], response)
        except Exception:
            logger.warning("asqav after-hook dispatch failed (fail-open)", exc_info=True)
        return response


_COMPLIANCE_FIELDS = (
    "payload_digest",
    "issuer_id",
    "iteration_id",
    "sandbox_state",
    "risk_class",
    "incident_class",
    "reason",
    "capture_topology",
    # NSA CSI U/OO/6030316-26 alignment.
    "result_digest",
    "tool_fingerprint",
    "config_manifest_digest",
    "cve_inventory_digest",
    "executable_hash",
    "sbom_digest",
    "slsa_provenance_pointer",
    "supply_chain_pointer",
    # Threat-framework taxonomy mappings.
    "mitre_techniques",
    "mitre_atlas",
    "owasp_llm_top10",
    "owasp_agentic_top10",
    "nist_ai_rmf",
    "iso_42001",
    "eu_ai_act_articles",
    "rfc3161_timestamp",
    "witness_policy",
    # Risk-acceptance extension fields (fenced to the receipt type).
    "approver_id",
    "initiator_id",
    "acceptance_reason",
    "accepted_at",
    "supersedes",
    "sarif_digest",
    "finding_ref",
    "approval_ref",
    "risk_snapshot",
    "invocation_ref",
    # Code-authorship extension fields (fenced to the receipt type).
    "repo_ref",
    "commit_sha",
    "base_sha",
    "change_digest",
    "change_ref",
    "change_approval_ref",
    "change_class",
    "authored_by",
    "counterparty_binding",
)


def _check_sign_params(p: dict[str, Any]) -> None:
    """Validate the sign arguments, then fill the derived ones in place."""
    p["action_type"], p["context"] = _prepare_sign_context(
        action_type=p["action_type"],
        context=p["context"],
        context_schema=p["context_schema"],
        system_prompt_hash=p["system_prompt_hash"],
        model_params=p["model_params"],
        tool_inputs_hash=p["tool_inputs_hash"],
        trace_id=p["trace_id"],
        parent_id=p["parent_id"],
        counterparty=p["counterparty"],
        tool_name=p["tool_name"],
        model_name=p["model_name"],
    )
    # Default-send the outcome (787): result_digest wins, else digest it.
    if p["result_digest"] is None and p["result"] is not None:
        p["result_digest"] = _digest_result_value(p["result"])
    _validate_sign_vocabulary(
        receipt_type=p["receipt_type"],
        policy_decision=p["policy_decision"],
        reason=p["reason"],
        sandbox_state=p["sandbox_state"],
        capture_topology=p["capture_topology"],
        incident_class=p["incident_class"],
        config_manifest_digest=p["config_manifest_digest"],
        result_digest=p["result_digest"],
    )
    _check_receipt_extensions(p)
    p["valid_seconds"] = _valid_seconds(p["valid_seconds"], p["expires_at"])
    # Rule 11 lockstep: tokens mirror cloud <field>_not_sha256_wire_form.
    _validate_sign_extensions(
        config_manifest_digest=p["config_manifest_digest"],
        cve_inventory_digest=p["cve_inventory_digest"],
        executable_hash=p["executable_hash"],
        sbom_digest=p["sbom_digest"],
        tool_fingerprint=p["tool_fingerprint"],
        slsa_provenance_pointer=p["slsa_provenance_pointer"],
        supply_chain_pointer=p["supply_chain_pointer"],
        mitre_techniques=p["mitre_techniques"],
        mitre_atlas=p["mitre_atlas"],
        owasp_llm_top10=p["owasp_llm_top10"],
        owasp_agentic_top10=p["owasp_agentic_top10"],
        nist_ai_rmf=p["nist_ai_rmf"],
        iso_42001=p["iso_42001"],
        eu_ai_act_articles=p["eu_ai_act_articles"],
    )
    _validate_sign_anchoring(
        rfc3161_timestamp=p["rfc3161_timestamp"],
        witness_policy=p["witness_policy"],
    )
    if p["compliance_mode"] and p["action_ref"] is None:
        p["action_ref"] = _compute_action_ref(p["action_type"], p["context"])
    if (
        p["tool_fingerprint"] is None
        and p["tool_name"] is not None
        and p["tool_schema"] is not None
    ):
        p["tool_fingerprint"] = _compute_tool_fingerprint(p["tool_name"], p["tool_schema"])
    # Fill the nonce wire field when omitted; the cloud has no re-use check.
    if p["nonce"] is None:
        p["nonce"] = _generate_nonce()


def _check_receipt_extensions(p: dict[str, Any]) -> None:
    """Fence the risk-acceptance, code-authorship and acknowledgment fields."""
    # Lockstep with cloud SignRequest._validate_risk_acceptance_extensions.
    _validate_risk_acceptance(
        receipt_type=p["receipt_type"],
        compliance_mode=p["compliance_mode"],
        policy_decision=p["policy_decision"],
        approver_id=p["approver_id"],
        initiator_id=p["initiator_id"],
        acceptance_reason=p["acceptance_reason"],
        accepted_at=p["accepted_at"],
        supersedes=p["supersedes"],
        sarif_digest=p["sarif_digest"],
        finding_ref=p["finding_ref"],
        approval_ref=p["approval_ref"],
        risk_snapshot=p["risk_snapshot"],
    )
    # Lockstep with cloud SignRequest._validate_code_authorship_extensions.
    _validate_code_authorship(
        receipt_type=p["receipt_type"],
        compliance_mode=p["compliance_mode"],
        policy_decision=p["policy_decision"],
        repo_ref=p["repo_ref"],
        commit_sha=p["commit_sha"],
        base_sha=p["base_sha"],
        change_digest=p["change_digest"],
        change_ref=p["change_ref"],
        change_approval_ref=p["change_approval_ref"],
        change_class=p["change_class"],
        authored_by=p["authored_by"],
    )
    p["counterparty_binding"] = _validate_counterparty_binding(
        p["counterparty_binding"],
        receipt_type=p["receipt_type"],
        compliance_mode=p["compliance_mode"],
    )


def _valid_seconds(valid_seconds: int | None, expires_at: str | float | None) -> int | None:
    """The duration to send: ``valid_seconds``, or ``expires_at`` turned into one."""
    # Rule 10: an absolute horizon and a duration together are ambiguous.
    if valid_seconds is not None and expires_at is not None:
        raise ValueError(
            "expiry_collision_guard: pass either valid_seconds or expires_at, not both (rule 10)"
        )
    if expires_at is None:
        return valid_seconds
    if isinstance(expires_at, (int, float)):
        expires_at_dt = datetime.fromtimestamp(float(expires_at), tz=timezone.utc)
    else:
        text = expires_at[:-1] + "+00:00" if expires_at.endswith("Z") else expires_at
        expires_at_dt = datetime.fromisoformat(text)
        if expires_at_dt.tzinfo is None:
            expires_at_dt = expires_at_dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max(1, ceil((expires_at_dt - now).total_seconds()))


def _sign_request_body(
    p: dict[str, Any], *, agent_id: str, session_id: str | None
) -> dict[str, Any]:
    """The JSON body of ``POST /agents/{id}/sign``."""
    compliance_fields: dict[str, Any] = {}
    if p["compliance_mode"]:
        compliance_fields = {
            "compliance_mode": True,
            "receipt_type": p["receipt_type"] or "protectmcp:decision",
            "policy_decision": p["policy_decision"],
            "action_ref": p["action_ref"],
        }
        compliance_fields.update({k: p[k] for k in _COMPLIANCE_FIELDS if p[k] is not None})
    body = _build_sign_body(
        action_type=p["action_type"],
        context=p["context"],
        session_id=session_id,
        agent_id=agent_id,
        user_intent=p["user_intent"],
        compliance_fields=compliance_fields or None,
    )
    if p["co_signers"]:
        body["co_signers"] = list(p["co_signers"])
    for key in ("mandate_id", "nonce", "valid_seconds", "expected_executor_pubkey_b64"):
        if p[key] is not None:
            body[key] = p[key]
    return body


def _signature_response(data: dict[str, Any]) -> SignatureResponse:
    """Decode the cloud's sign response."""
    anchor = data.get("rfc3161_timestamp") or {}
    return SignatureResponse(
        signature=require_field(data, "signature"),
        signature_id=require_field(data, "signature_id"),
        action_id=require_field(data, "action_id"),
        timestamp=require_field(data, "timestamp"),
        verification_url=require_field(data, "verification_url"),
        algorithm=data.get("algorithm"),
        chain_hash=data.get("chain_hash") or data.get("record_hash"),
        rfc3161_tsa=anchor.get("tsa"),
        rfc3161_serial=anchor.get("serial_number"),
        scan_result=data.get("scan_result"),
        policy_digest=data.get("policy_digest"),
        policy_decision=data.get("policy_decision", "permit"),
        authorization_ref=data.get("authorization_ref"),
        bitcoin_anchor=_parse_bitcoin_anchor(data.get("bitcoin_anchor")),
        required_co_signers=data.get("required_co_signers"),
        co_signatures=data.get("co_signatures"),
        countersign_url=data.get("countersign_url"),
        user_intent_verified=data.get("user_intent_verified"),
        compliance_mode=bool(data.get("compliance_mode", False)),
        receipt_type=data.get("receipt_type"),
        action_ref=_resolve_action_ref(data),
        payload_digest=data.get("payload_digest"),
        issuer_id=data.get("issuer_id"),
        iteration_id=data.get("iteration_id"),
        sandbox_state=data.get("sandbox_state"),
        risk_class=data.get("risk_class"),
        incident_class=data.get("incident_class"),
        reason=data.get("reason"),
        # Accept camelCase `previousReceiptHash` and its snake_case alias.
        previous_receipt_hash=_resolve_previous_receipt_hash(data),
        # Spec-shape `decision`; older clouds only emit `policy_decision`.
        decision=data.get("decision")
        or (
            _map_policy_decision_to_decision(data.get("policy_decision"))
            if data.get("compliance_mode")
            else None
        ),
        payload=data.get("payload"),
        anchors=data.get("anchors"),
    )
