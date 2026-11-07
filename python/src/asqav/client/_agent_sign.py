"""Agent.sign: one signed action."""

from __future__ import annotations

from collections.abc import Callable
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
        # NSA CSI U/OO/6030316-26 alignment.
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
        # Risk-acceptance / exception receipt extension fields. Valid only on
        # receipt_type=protectmcp:lifecycle:risk_acceptance (fenced below).
        approver_id: str | None = None,
        initiator_id: str | None = None,
        acceptance_reason: str | None = None,
        accepted_at: str | None = None,
        supersedes: str | None = None,
        sarif_digest: str | None = None,
        finding_ref: str | None = None,
        approval_ref: str | None = None,
        risk_snapshot: "RiskSnapshot | dict[str, Any] | None" = None,
        # Invocation pointer, producer-asserted and unfenced: names one
        # invocation only, promises nothing about uniqueness or exactly-once.
        invocation_ref: str | None = None,
        # Code-authorship receipt extension fields. Valid only on
        # receipt_type=protectmcp:lifecycle:code_authorship (fenced below).
        repo_ref: str | None = None,
        commit_sha: str | None = None,
        base_sha: str | None = None,
        change_digest: str | None = None,
        change_ref: str | None = None,
        change_approval_ref: str | None = None,
        change_class: str | None = None,
        authored_by: "AuthoredBy | dict[str, Any] | None" = None,
        # Agent-to-agent ack: binds this receipt to the peer receipt it
        # acknowledges. Valid only on receipt_type=protectmcp:acknowledgment.
        counterparty_binding: "CounterpartyBinding | dict[str, Any] | None" = None,
    ) -> SignatureResponse:
        """Sign an action cryptographically.

        The signature is created server-side with ML-DSA.

        Args:
            action_type: Type of action (e.g., "read:data", "api:call").
                Accepts semantic pattern names like "sql-read" which
                auto-expand to their glob equivalents.
            context: Additional context for the action.
            context_schema: Optional schema for the ``context`` dict.
                When supplied, the context is validated and then key-sorted
                before signing so audit trails are consistent and queryable.
                Two forms are accepted:

                Built-in descriptor (dependency-free)::

                    {"user_id": {"type": "string", "required": True},
                     "score":   {"type": "number"}}

                Field types: string | number | boolean | object | array.
                ``required`` defaults to False. Raises
                ``AsqavValidationError`` on mismatch BEFORE the network call.

                Custom callable (bring-your-own validator)::

                    def my_validator(ctx: dict) -> None:
                        import jsonschema
                        jsonschema.validate(ctx, MY_FULL_SCHEMA)

                    agent.sign("action", ctx, context_schema=my_validator)

                A callable receives the (merged, hook-dispatched) context
                and must raise on invalid input. Omit for today's behavior.
            system_prompt_hash: Optional hash of the system prompt for
                reproducibility tracking.
            model_params: Optional dict of model parameters (temperature,
                top_p, etc.) for reproducibility tracking.
            tool_inputs_hash: Optional hash of tool inputs for
                reproducibility tracking.
            trace_id: Optional trace ID for correlating actions across
                a multi-step workflow. Use generate_trace_id() to create one.
            parent_id: Optional parent action ID for linking child actions
                to their parent in a workflow. Powers the agent graph view.
            counterparty: Optional identity of the endpoint the agent is
                calling. Included in the signed record so an auditor can
                verify the agent reached a specific endpoint, not an
                imposter. Recommended shape::

                    {"kind": "tls_spki_sha256", "value": "<hex>"}
                    {"kind": "pubkey_fingerprint", "value": "<hex>"}
                    {"kind": "url_sni", "value": "api.example.com"}

                Any dict is accepted; unknown kinds are recorded verbatim.
            tool_name: Optional name of the tool being invoked when this
                action is a tool call. Surfaced in the dashboard graph.
                Only the name string travels; tool arguments do not.
            model_name: Optional name of the model being called when this
                action is an LLM call (e.g. "gpt-4", "claude-opus-4-7").
                Only the name string travels; the prompt does not.
            capture_topology: Producer-side topology. One of
                ``in_process_sdk``, ``network_proxy``, ``browser_extension``,
                ``mcp_proxy``, ``passive_telemetry``, ``github_sha_pull``.
                Stamped on the audit-pack manifest entry; never on the
                signed payload. Client-supplied capture_topology is ADVISORY:
                the server stamps the authoritative value from the ingress
                route. ``passive_telemetry`` requires
                ``receipt_type='protectmcp:observation'`` (false-attestation guard).
            result_digest: NSA CSI U/OO/6030316-26 alignment.
                ``sha256:<hex>`` of the tool output. Caller-supplied and
                forwarded verbatim to the cloud. When omitted and ``result``
                is given, the digest is computed from ``result`` and sent
                by default (criterion 787: every action gets an outcome).
            result: Optional tool result (bytes, str, or JSON-able value).
                Its canonical ``sha256:<hex>`` is computed and sent as
                ``result_digest`` unless ``result_digest`` is given
                explicitly. Sign-then-run callers that have no result yet
                omit both and attach the outcome later via
                :meth:`attach_result`.
            expires_at: Receipt validity horizon. ISO-8601 string or POSIX
                float. The cloud owns absolute time-binding and accepts
                only a duration, so the SDK converts this absolute horizon
                to a client-computed ``valid_seconds`` and sends that
                instead. Mutually exclusive with ``valid_seconds`` (rule
                10). Pass exactly one of the two: ``valid_seconds`` for
                "expire N seconds after signing", or ``expires_at`` for an
                explicit horizon (converted client-side). Passing both
                raises ``expiry_collision_guard``. Passing neither falls
                back to the server-side default
                (``valid_seconds=86400``).
            tool_schema: Optional JSON schema describing the tool surface.
                When provided alongside ``tool_name`` and
                ``tool_fingerprint`` is omitted, the SDK computes the
                fingerprint client-side via JCS canonicalization.
            tool_fingerprint: Explicit fingerprint over the canonical
                ``{tool_name, schema}`` blob. When omitted but
                ``tool_name`` + ``tool_schema`` are present, the SDK
                derives it client-side. MUST be 32 bare lowercase hex
                chars (SHA-256[:32], no ``sha256:`` prefix), matching the
                cloud wire form; use ``_compute_tool_fingerprint`` to
                avoid drift.
            config_manifest_digest: NSA CSI alignment. ``sha256:<hex>`` of
                the agent's runtime configuration manifest. REQUIRED for
                ``receipt_type=protectmcp:lifecycle:configuration_change``
                (false-attestation rule 9). MUST match ``sha256:<64-hex>``
                (rule 11); use ``_compute_config_manifest_digest`` to
                avoid drift.
            cve_inventory_digest: NSA CSI alignment. ``sha256:<hex>`` of
                the agent's CVE inventory snapshot at signing time. MUST
                match ``sha256:<64-hex>`` (rule 11); use
                ``_compute_cve_inventory_digest`` to avoid drift.
            executable_hash: Build-provenance 4-tuple. ``sha256:<hex>``
                of the executable that invoked the action. Binds
                build-side provenance into the signed receipt. MUST
                match ``sha256:<64-hex>`` (rule 11).
            sbom_digest: Build-provenance 4-tuple. ``sha256:<hex>`` of
                the canonical CycloneDX or SPDX SBOM document. MUST
                match ``sha256:<64-hex>`` (rule 11).
            slsa_provenance_pointer: Build-provenance 4-tuple. https
                URL to the SLSA attestation envelope for the executing
                build.
            supply_chain_pointer: Build-provenance 4-tuple. https URL
                to the in-toto, Sigstore, or Rekor entry covering the
                executing build.
            mitre_techniques: Caller-supplied list of MITRE ATT&CK
                technique ids (e.g. ``["T1059", "T1078"]``). Self-declared;
                never Asqav-verified. Auto-flips
                ``framework_mappings_self_declared`` to True on the receipt.
            mitre_atlas: Caller-supplied list of MITRE ATLAS ids for
                AI-system threats (e.g. ``["AML.T0051", "AML.T0043"]``).
                Self-declared; never Asqav-verified.
            owasp_llm_top10: Caller-supplied list of OWASP Top 10 for LLM
                ids (e.g. ``["LLM01", "LLM02"]``). Self-declared.
            owasp_agentic_top10: Caller-supplied OWASP Agentic Top 10 ids ASI01
                through ASI10 without an edition suffix. Self-declared.
            nist_ai_rmf: Caller-supplied list of NIST AI RMF function ids
                (e.g. ``["GOVERN-1.1", "MEASURE-2.7"]``). Self-declared.
            iso_42001: Caller-supplied list of ISO/IEC 42001 control ids
                (e.g. ``["A.6.2.6"]``). Self-declared.
            eu_ai_act_articles: Caller-supplied list of EU AI Act article
                ids (e.g. ``["Article-12", "Article-15", "Article-50"]``). Self-declared.
            rfc3161_timestamp: Caller-supplied base64-encoded RFC 3161
                TimeStampResp (DER). Preserved on the receipt for offline
                TSA chain verification independent of cloud-issued anchors.
            witness_policy: Caller-declared N-of-M durable-anchoring quorum.
                Shape ``{"required": int, "witnesses": [<subset of
                "rfc3161", "opentimestamps">]}``. ``required`` must be in
                ``[1, len(witnesses)]``; ``witnesses`` must be a non-empty
                subset of the two shipped witnesses. ``rekor`` is rejected
                (not a shipped witness). The receipt reaches
                ``witness_quorum_met`` only when ``required`` witnesses hold
                a real inclusion proof. Optional; omit for today's
                behaviour.

        Returns:
            SignatureResponse with the signature.
        """
        action_type, context = _prepare_sign_context(
            action_type=action_type,
            context=context,
            context_schema=context_schema,
            system_prompt_hash=system_prompt_hash,
            model_params=model_params,
            tool_inputs_hash=tool_inputs_hash,
            trace_id=trace_id,
            parent_id=parent_id,
            counterparty=counterparty,
            tool_name=tool_name,
            model_name=model_name,
        )

        # Surface bad arguments client-side; the cloud re-validates.
        # Default-send the outcome (787): result_digest wins, else digest it.
        if result_digest is None and result is not None:
            result_digest = _digest_result_value(result)
        _validate_sign_vocabulary(
            receipt_type=receipt_type,
            policy_decision=policy_decision,
            reason=reason,
            sandbox_state=sandbox_state,
            capture_topology=capture_topology,
            incident_class=incident_class,
            config_manifest_digest=config_manifest_digest,
            result_digest=result_digest,
        )
        # Risk-acceptance receipt: shape + namespace fence + no-policy opt-out,
        # lockstep with cloud SignRequest._validate_risk_acceptance_extensions.
        _validate_risk_acceptance(
            receipt_type=receipt_type,
            compliance_mode=compliance_mode,
            policy_decision=policy_decision,
            approver_id=approver_id,
            initiator_id=initiator_id,
            acceptance_reason=acceptance_reason,
            accepted_at=accepted_at,
            supersedes=supersedes,
            sarif_digest=sarif_digest,
            finding_ref=finding_ref,
            approval_ref=approval_ref,
            risk_snapshot=risk_snapshot,
        )
        # Code-authorship receipt: shape + namespace fence + no-policy opt-out,
        # lockstep with cloud SignRequest._validate_code_authorship_extensions.
        _validate_code_authorship(
            receipt_type=receipt_type,
            compliance_mode=compliance_mode,
            policy_decision=policy_decision,
            repo_ref=repo_ref,
            commit_sha=commit_sha,
            base_sha=base_sha,
            change_digest=change_digest,
            change_ref=change_ref,
            change_approval_ref=change_approval_ref,
            change_class=change_class,
            authored_by=authored_by,
        )

        counterparty_binding = _validate_counterparty_binding(
            counterparty_binding, receipt_type=receipt_type, compliance_mode=compliance_mode
        )

        # Rule 10: an absolute horizon and a duration together are ambiguous.
        if valid_seconds is not None and expires_at is not None:
            raise ValueError(
                "expiry_collision_guard: pass either valid_seconds or "
                "expires_at, not both (rule 10)"
            )

        # Convert absolute expires_at to valid_seconds (the cloud owns time).
        if expires_at is not None:
            from datetime import datetime, timezone
            from math import ceil

            if isinstance(expires_at, (int, float)):
                expires_at_dt = datetime.fromtimestamp(float(expires_at), tz=timezone.utc)
            else:
                _s = expires_at
                if _s.endswith("Z"):
                    _s = _s[:-1] + "+00:00"
                expires_at_dt = datetime.fromisoformat(_s)
                if expires_at_dt.tzinfo is None:
                    expires_at_dt = expires_at_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            valid_seconds = max(1, ceil((expires_at_dt - now).total_seconds()))

        # Rule 11 lockstep: tokens mirror cloud <field>_not_sha256_wire_form.
        _validate_sign_extensions(
            config_manifest_digest=config_manifest_digest,
            cve_inventory_digest=cve_inventory_digest,
            executable_hash=executable_hash,
            sbom_digest=sbom_digest,
            tool_fingerprint=tool_fingerprint,
            slsa_provenance_pointer=slsa_provenance_pointer,
            supply_chain_pointer=supply_chain_pointer,
            mitre_techniques=mitre_techniques,
            mitre_atlas=mitre_atlas,
            owasp_llm_top10=owasp_llm_top10,
            owasp_agentic_top10=owasp_agentic_top10,
            nist_ai_rmf=nist_ai_rmf,
            iso_42001=iso_42001,
            eu_ai_act_articles=eu_ai_act_articles,
        )
        _validate_sign_anchoring(
            rfc3161_timestamp=rfc3161_timestamp,
            witness_policy=witness_policy,
        )

        # Derive action_ref under compliance_mode when caller omits it.
        if compliance_mode and action_ref is None:
            action_ref = _compute_action_ref(action_type, context)

        # Derive tool_fingerprint client-side when the caller supplied
        # tool_name + tool_schema but no explicit fingerprint.
        if tool_fingerprint is None and tool_name is not None and tool_schema is not None:
            tool_fingerprint = _compute_tool_fingerprint(tool_name, tool_schema)

        # Fill the nonce wire field when omitted; the cloud has no re-use check.
        if nonce is None:
            nonce = _generate_nonce()

        compliance_fields: dict[str, Any] = {}
        if compliance_mode:
            compliance_fields["compliance_mode"] = True
            compliance_fields["receipt_type"] = receipt_type or "protectmcp:decision"
            compliance_fields["policy_decision"] = policy_decision
            compliance_fields["action_ref"] = action_ref
            for k, v in (
                ("payload_digest", payload_digest),
                ("issuer_id", issuer_id),
                ("iteration_id", iteration_id),
                ("sandbox_state", sandbox_state),
                ("risk_class", risk_class),
                ("incident_class", incident_class),
                ("reason", reason),
                ("capture_topology", capture_topology),
                # NSA CSI U/OO/6030316-26 alignment.
                ("result_digest", result_digest),
                ("tool_fingerprint", tool_fingerprint),
                ("config_manifest_digest", config_manifest_digest),
                ("cve_inventory_digest", cve_inventory_digest),
                ("executable_hash", executable_hash),
                ("sbom_digest", sbom_digest),
                ("slsa_provenance_pointer", slsa_provenance_pointer),
                ("supply_chain_pointer", supply_chain_pointer),
                # Threat-framework taxonomy mappings.
                ("mitre_techniques", mitre_techniques),
                ("mitre_atlas", mitre_atlas),
                ("owasp_llm_top10", owasp_llm_top10),
                ("owasp_agentic_top10", owasp_agentic_top10),
                ("nist_ai_rmf", nist_ai_rmf),
                ("iso_42001", iso_42001),
                ("eu_ai_act_articles", eu_ai_act_articles),
                ("rfc3161_timestamp", rfc3161_timestamp),
                ("witness_policy", witness_policy),
                # Risk-acceptance extension fields (fenced to the receipt type).
                ("approver_id", approver_id),
                ("initiator_id", initiator_id),
                ("acceptance_reason", acceptance_reason),
                ("accepted_at", accepted_at),
                ("supersedes", supersedes),
                ("sarif_digest", sarif_digest),
                ("finding_ref", finding_ref),
                ("approval_ref", approval_ref),
                ("risk_snapshot", risk_snapshot),
                # Unfenced invocation pointer (see signature comment).
                ("invocation_ref", invocation_ref),
                # Code-authorship extension fields (fenced to the receipt type).
                ("repo_ref", repo_ref),
                ("commit_sha", commit_sha),
                ("base_sha", base_sha),
                ("change_digest", change_digest),
                ("change_ref", change_ref),
                ("change_approval_ref", change_approval_ref),
                ("change_class", change_class),
                ("authored_by", authored_by),
                ("counterparty_binding", counterparty_binding),
            ):
                if v is not None:
                    compliance_fields[k] = v

        body = _build_sign_body(
            action_type=action_type,
            context=context,
            session_id=self._session_id,
            agent_id=self.agent_id,
            user_intent=user_intent,
            compliance_fields=compliance_fields or None,
        )
        if co_signers:
            body["co_signers"] = list(co_signers)
        # Issuer mandate the sign is authorised under; the cloud re-verifies it
        # and builds the signed authorized_under_mandate attestation there.
        if mandate_id is not None:
            body["mandate_id"] = mandate_id
        # Opaque caller token; the cloud answers 409 if a live signature has it.
        if nonce is not None:
            body["nonce"] = nonce
        if valid_seconds is not None:
            body["valid_seconds"] = valid_seconds
        # Counterparty pin: server decodes b64.
        if expected_executor_pubkey_b64 is not None:
            body["expected_executor_pubkey_b64"] = expected_executor_pubkey_b64
        tap = _sign_body_tap.get()
        if tap is not None:
            tap(body)
        data = _transport._post(f"/agents/{self.agent_id}/sign", body)

        response = SignatureResponse(
            signature=require_field(data, "signature"),
            signature_id=require_field(data, "signature_id"),
            action_id=require_field(data, "action_id"),
            timestamp=require_field(data, "timestamp"),
            verification_url=require_field(data, "verification_url"),
            algorithm=data.get("algorithm"),
            chain_hash=data.get("chain_hash") or data.get("record_hash"),
            rfc3161_tsa=(data.get("rfc3161_timestamp") or {}).get("tsa"),
            rfc3161_serial=(data.get("rfc3161_timestamp") or {}).get("serial_number"),
            scan_result=data.get("scan_result"),
            policy_digest=data.get("policy_digest"),
            policy_decision=data.get("policy_decision", "permit"),
            authorization_ref=data.get("authorization_ref"),
            bitcoin_anchor=_parse_bitcoin_anchor(data.get("bitcoin_anchor")),
            required_co_signers=data.get("required_co_signers"),
            co_signatures=data.get("co_signatures"),
            countersign_url=data.get("countersign_url"),
            user_intent_verified=data.get("user_intent_verified"),
            # Accept camelCase `previousReceiptHash` and its snake_case alias.
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
            previous_receipt_hash=_resolve_previous_receipt_hash(data),
            # Spec-shape `decision`; fall back to local translation for
            # older clouds that only emit `policy_decision`.
            decision=(
                data.get("decision")
                or (
                    _map_policy_decision_to_decision(data.get("policy_decision"))
                    if data.get("compliance_mode")
                    else None
                )
            ),
            payload=data.get("payload"),
            anchors=data.get("anchors"),
        )

        # Dispatch after-hooks (fail-open).
        try:
            from .. import hooks as _hooks_mod

            _hooks_mod._dispatch_after(action_type, response)
        except Exception:
            logger.warning("asqav after-hook dispatch failed (fail-open)", exc_info=True)

        return response
