"""Building and validating the body Agent.sign posts."""

from __future__ import annotations

import contextvars
import hashlib
import hmac
from collections.abc import Callable
from typing import Any

from ..patterns import resolve_pattern
from . import _transport
from ._common import _OWASP_AGENTIC_TOP10_IDS, logger
from ._sign_checks import _SHA256_HEX_RE, _TOOL_FINGERPRINT_RE
from ._vocabulary import (
    CAPTURE_TOPOLOGY_NAMESPACE,
    INCIDENT_CLASS_NAMESPACE,
    RECEIPT_TYPE_NAMESPACE,
    SANDBOX_STATE_NAMESPACE,
    WITNESS_NAMESPACE,
)

#: Observer handed the body Agent.sign will POST (the hook gate hashes it).
_sign_body_tap: contextvars.ContextVar[Callable[[dict[str, Any]], None] | None] = (
    contextvars.ContextVar("asqav_sign_body_tap", default=None)
)


#: Hash-only request metadata allowlist; mirrors the server-enforced whitelist.
_HASH_ONLY_METADATA_WHITELIST: frozenset[str] = frozenset(
    {"agent_id", "session_id", "action_type", "model_name", "tool_name", "parent_id"}
)


def _build_sign_body(
    *,
    action_type: str,
    context: dict[str, Any] | None,
    session_id: str | None,
    agent_id: str,
    user_intent: dict[str, Any] | None = None,
    compliance_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON body for ``POST /agents/{id}/sign``.

    full-payload sends ``context``; hash-only sends a digest plus a
    whitelisted metadata bag.
    """
    if _transport._mode == "hash-only":
        from ..canonicalize import canonicalize_action

        canonical_bytes = canonicalize_action(action_type, context)
        payload_size = len(canonical_bytes)
        if _transport._org_salt is not None:
            digest_hex = hmac.new(_transport._org_salt, canonical_bytes, hashlib.sha256).hexdigest()
            hash_algo = "hmac-sha256"
        else:
            digest_hex = hashlib.sha256(canonical_bytes).hexdigest()
            hash_algo = "sha256"
        # Wire digest keeps the sha256:<hex> shape; hash_algo names the keying.
        digest = f"sha256:{digest_hex}"
        metadata: dict[str, Any] = {
            "agent_id": agent_id,
            "action_type": action_type,
        }
        if session_id is not None:
            metadata["session_id"] = session_id
        # Opt in via underscored sentinel keys; no other context travels.
        for src_key, dst_key in (
            ("_model_name", "model_name"),
            ("_tool_name", "tool_name"),
            ("_parent_id", "parent_id"),
        ):
            if context and src_key in context:
                value = context[src_key]
                if isinstance(value, str):
                    metadata[dst_key] = value
        metadata = {k: v for k, v in metadata.items() if k in _HASH_ONLY_METADATA_WHITELIST}
        body: dict[str, Any] = {
            "action_type": action_type,
            "hash": digest,
            "hash_algo": hash_algo,
            "payload_size": payload_size,
            "metadata": metadata,
            "session_id": session_id,
        }
        if user_intent is not None:
            body["user_intent"] = user_intent
        if compliance_fields:
            for key, value in compliance_fields.items():
                if value is not None:
                    body[key] = value
        return body

    # full-payload (default for self-hosted)
    body = {
        "action_type": action_type,
        "context": context or {},
        "session_id": session_id,
    }
    if user_intent is not None:
        body["user_intent"] = user_intent
    if compliance_fields:
        for key, value in compliance_fields.items():
            if value is not None:
                body[key] = value
    return body


def _prepare_sign_context(
    *,
    action_type: str,
    context: dict[str, Any] | None,
    context_schema: 'dict[str, Any] | Callable[[dict[str, Any]], None] | None' = None,
    system_prompt_hash: str | None = None,
    model_params: dict | None = None,
    tool_inputs_hash: str | None = None,
    trace_id: str | None = None,
    parent_id: str | None = None,
    counterparty: dict[str, Any] | None = None,
    tool_name: str | None = None,
    model_name: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Return the action type and context exactly as ``Agent.sign`` signs them.

    Shared with the PreToolUse gate so its digest matches the POSTed body.
    """
    action_type = resolve_pattern(action_type)
    if (
        system_prompt_hash
        or model_params
        or tool_inputs_hash
        or trace_id
        or parent_id
        or counterparty
        or tool_name
        or model_name
    ):
        context = dict(context) if context else {}
        if system_prompt_hash:
            context["_system_prompt_hash"] = system_prompt_hash
        if model_params:
            context["_model_params"] = model_params
        if tool_inputs_hash:
            context["_tool_inputs_hash"] = tool_inputs_hash
        if trace_id:
            context["_trace_id"] = trace_id
        if parent_id:
            context["_parent_id"] = parent_id
        if counterparty:
            context["_counterparty"] = counterparty
        if tool_name:
            context["_tool_name"] = tool_name
        if model_name:
            context["_model_name"] = model_name

    # Dispatch before-hooks (fail-open).
    try:
        from .. import hooks as _hooks_mod

        context = _hooks_mod._dispatch_before(action_type, dict(context) if context else {})
    except Exception:
        logger.warning("asqav before-hook dispatch failed (fail-open)", exc_info=True)

    # Opt-in schema validation + normalization (criterion 328).
    # Runs after hook dispatch so the validated form is what gets signed.
    if context_schema is not None:
        from .._schema import normalize_context, validate_context_schema

        if callable(context_schema) and not isinstance(context_schema, dict):
            # User-supplied validator function (e.g. wrapping jsonschema).
            context_schema(dict(context) if context else {})
        else:
            validate_context_schema(context, context_schema)
        context = normalize_context(context)

    # Pluggable detector gate (331), fail-closed; after normalize.
    from .._detectors import run_detectors

    _detector_records = run_detectors(action_type, context)
    if _detector_records:
        # Stamp verdicts into context so they travel inside the signed body.
        context = dict(context) if context else {}
        context["_detectors"] = _detector_records
    return action_type, context


def _digest_result_value(result: bytes | str | dict | list) -> str:
    """Compute the canonical ``sha256:<hex>`` of a tool result.
    JCS for dicts and lists, UTF-8 for strings, raw bytes; else str()."""
    import hashlib

    if isinstance(result, bytes):
        raw = result
    elif isinstance(result, str):
        raw = result.encode("utf-8")
    else:
        from ..canonicalize import canonicalize

        try:
            raw = canonicalize(result)
        except (TypeError, ValueError, RecursionError):
            raw = str(result).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _validate_sign_vocabulary(
    *,
    receipt_type: str | None,
    policy_decision: str,
    reason: str | None,
    sandbox_state: str | None,
    capture_topology: str | None,
    incident_class: str | list[str] | None,
    config_manifest_digest: str | None,
    result_digest: str | None,
) -> None:
    if receipt_type is not None and receipt_type not in RECEIPT_TYPE_NAMESPACE:
        raise ValueError(
            f"invalid_receipt_type: must be one of {sorted(RECEIPT_TYPE_NAMESPACE)}"
        )
    if policy_decision in {"deny", "rate_limit"} and not reason:
        raise ValueError(
            "missing_reason: policy_decision=deny|rate_limit requires a `reason` code."
        )
    # Fail fast on sandbox_state vocabulary before the HTTP roundtrip.
    if sandbox_state is not None and sandbox_state not in SANDBOX_STATE_NAMESPACE:
        raise ValueError(
            f"invalid_sandbox_state: must be one of {sorted(SANDBOX_STATE_NAMESPACE)}."
        )
    # Fail fast on capture_topology vocabulary before the HTTP roundtrip.
    if capture_topology is not None and capture_topology not in CAPTURE_TOPOLOGY_NAMESPACE:
        raise ValueError(
            f"invalid_capture_topology: must be one of {sorted(CAPTURE_TOPOLOGY_NAMESPACE)}."
        )
    # Rule 8 (lockstep with cloud SignRequest validator): passive_telemetry
    # pairs only with protectmcp:observation[:result_bound].
    if (
        capture_topology == "passive_telemetry"
        and receipt_type is not None
        and receipt_type
        not in {
            "protectmcp:observation",
            "protectmcp:observation:result_bound",
        }
    ):
        offending = receipt_type.split(":", 1)[-1]
        raise ValueError(
            "false_attestation_guard: capture_topology=passive_telemetry "
            "receipts must use receipt_type=protectmcp:observation"
            "[:result_bound], "
            f"not :{offending} (rule 8)"
        )
    # Fail fast on incident_class vocabulary before the HTTP roundtrip.
    # The field accepts a JSON string or a JSON array of such strings.
    if incident_class is not None:
        _tokens = incident_class if isinstance(incident_class, list) else [incident_class]
        for _t in _tokens:
            if _t not in INCIDENT_CLASS_NAMESPACE:
                raise ValueError(
                    "invalid_incident_class: must be one of "
                    f"{sorted(INCIDENT_CLASS_NAMESPACE)} "
                    "(per DORA RTS JC 2024-33 Annex II field 3.23 "
                    "or HIPAA 45 CFR 164.304)."
                )
    # Rule 9 lockstep with the cloud SignRequest cross-field validator.
    if (
        receipt_type == "protectmcp:lifecycle:configuration_change"
        and config_manifest_digest is None
    ):
        raise ValueError(
            "configuration_change_missing_config_manifest_digest: "
            "receipt_type=protectmcp:lifecycle:configuration_change "
            "requires config_manifest_digest (sha256:<64 hex>)."
        )
    if receipt_type == "protectmcp:observation:result_bound" and result_digest is None:
        raise ValueError(
            "result_bound_missing_result_digest: "
            "receipt_type=protectmcp:observation:result_bound requires "
            "result_digest (sha256:<64 hex>)."
        )


def _validate_sign_extensions(
    *,
    config_manifest_digest: str | None,
    cve_inventory_digest: str | None,
    executable_hash: str | None,
    sbom_digest: str | None,
    tool_fingerprint: str | None,
    slsa_provenance_pointer: str | None,
    supply_chain_pointer: str | None,
    mitre_techniques: list[str] | None,
    mitre_atlas: list[str] | None,
    owasp_llm_top10: list[str] | None,
    owasp_agentic_top10: list[str] | None,
    nist_ai_rmf: list[str] | None,
    iso_42001: list[str] | None,
    eu_ai_act_articles: list[str] | None,
) -> None:
    for _name, _value in (
        ("config_manifest_digest", config_manifest_digest),
        ("cve_inventory_digest", cve_inventory_digest),
        ("executable_hash", executable_hash),
        ("sbom_digest", sbom_digest),
    ):
        if _value is not None and not _SHA256_HEX_RE.match(_value):
            raise ValueError(
                f"{_name}_not_sha256_wire_form: must look like 'sha256:<64 lowercase hex>'."
            )
    if tool_fingerprint is not None and not _TOOL_FINGERPRINT_RE.match(tool_fingerprint):
        raise ValueError(
            "tool_fingerprint_not_32_hex_chars: must be 32 lowercase hex chars (SHA-256[:32])."
        )
    for _name, _value in (
        ("slsa_provenance_pointer", slsa_provenance_pointer),
        ("supply_chain_pointer", supply_chain_pointer),
    ):
        if _value is not None and not (
            _value.startswith("https://") or _value.startswith("http://")
        ):
            raise ValueError(f"pointer_url_guard: {_name} must be an http(s) URL")

    # Threat-framework taxonomy validators lockstep with cloud SignRequest.
    for _name, _value in (
        ("mitre_techniques", mitre_techniques),
        ("mitre_atlas", mitre_atlas),
        ("owasp_llm_top10", owasp_llm_top10),
        ("owasp_agentic_top10", owasp_agentic_top10),
        ("nist_ai_rmf", nist_ai_rmf),
        ("iso_42001", iso_42001),
        ("eu_ai_act_articles", eu_ai_act_articles),
    ):
        if _value is None:
            continue
        if not isinstance(_value, list) or len(_value) == 0:
            raise ValueError(
                f"{_name}_must_be_non_empty_list: pass a non-empty "
                "list of strings or omit the field."
            )
        for _item in _value:
            if not isinstance(_item, str) or not _item or len(_item) > 128:
                raise ValueError(
                    f"{_name}_entry_invalid: each entry must be a "
                    "non-empty string of length <= 128."
                )
            if _name == "owasp_agentic_top10" and _item not in _OWASP_AGENTIC_TOP10_IDS:
                raise ValueError(
                    f"{_name}_entry_invalid: use bare ASI01 through ASI10; no edition suffix."
                )


def _validate_sign_anchoring(
    *,
    rfc3161_timestamp: str | None,
    witness_policy: dict[str, Any] | None,
) -> None:
    if rfc3161_timestamp is not None:
        import base64 as _b64
        import binascii as _binascii

        if not isinstance(rfc3161_timestamp, str) or not rfc3161_timestamp:
            raise ValueError(
                "rfc3161_timestamp_not_base64: must be a non-empty "
                "base64-encoded TimeStampResp."
            )
        try:
            _b64.b64decode(rfc3161_timestamp, validate=True)
        except (_binascii.Error, ValueError) as _exc:
            raise ValueError(
                f"rfc3161_timestamp_not_base64: must be valid base64 (decode failed: {_exc})."
            ) from _exc

    # witness_policy: {required, witnesses: subset of rfc3161/opentimestamps}.
    if witness_policy is not None:
        if not isinstance(witness_policy, dict):
            raise ValueError(
                'witness_policy_invalid: must be a dict {"required": int, "witnesses": [...]}.'
            )
        _witnesses = witness_policy.get("witnesses")
        _required = witness_policy.get("required")
        if not isinstance(_witnesses, list) or len(_witnesses) == 0:
            raise ValueError(
                "witness_policy_witnesses_must_be_non_empty_list: "
                "witnesses must be a non-empty list drawn from "
                f"{sorted(WITNESS_NAMESPACE)}."
            )
        for _w in _witnesses:
            if _w not in WITNESS_NAMESPACE:
                raise ValueError(
                    f"witness_policy_unknown_witness: '{_w}' is not a "
                    "shipped witness; witnesses must be a subset of "
                    f"{sorted(WITNESS_NAMESPACE)} (rekor is rejected)."
                )
        if len(set(_witnesses)) != len(_witnesses):
            raise ValueError(
                "witness_policy_duplicate_witness: witnesses must not contain duplicates."
            )
        if not isinstance(_required, int) or isinstance(_required, bool):
            raise ValueError(
                "witness_policy_required_must_be_int: required must be "
                "an integer in [1, len(witnesses)]."
            )
        if not (1 <= _required <= len(_witnesses)):
            raise ValueError(
                "witness_policy_required_out_of_range: required must be "
                f"in [1, {len(_witnesses)}] (got {_required})."
            )
