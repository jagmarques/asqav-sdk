"""Agent methods built on sign: countersign, batches, output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..phases import PhaseChain

from ..patterns import resolve_pattern
from . import _transport
from ._common import _hash_value
from ._errors import require_field
from ._models import SignatureResponse, _parse_bitcoin_anchor
from ._sign_body import _build_sign_body
from ._vocabulary import _map_policy_decision_to_decision


class _AgentSigning:
    def countersign(self, signature_id: str) -> SignatureResponse:
        """Countersign an existing signature record.

        The server looks up ``signature_id``, verifies that this agent is
        in the original ``required_co_signers`` list and shares the same
        organization, then signs the SAME canonical bytes the original
        signer signed. No new prompts or tool args travel over the wire;
        only signatures are added.

        Args:
            signature_id: The ``sig_...`` id returned by the original
                ``sign(co_signers=[...])`` call.

        Returns:
            SignatureResponse describing the updated record, including
            the full ``co_signatures`` list after this agent's signature
            was appended.
        """
        data = _transport._post(
            f"/agents/{self.agent_id}/countersign/{signature_id}",
            {},
        )
        return SignatureResponse(
            signature=require_field(data, "signature"),
            signature_id=require_field(data, "signature_id"),
            action_id=require_field(data, "action_id"),
            timestamp=require_field(data, "timestamp"),
            verification_url=require_field(data, "verification_url"),
            algorithm=data.get("algorithm"),
            chain_hash=data.get("chain_hash") or data.get("record_hash"),
            rfc3161_tsa=(data.get("rfc3161_timestamp") or {}).get("tsa"),
            rfc3161_serial=(data.get("rfc3161_timestamp") or {}).get("serial_number"),
            policy_digest=data.get("policy_digest"),
            policy_decision=data.get("policy_decision", "permit"),
            authorization_ref=data.get("authorization_ref"),
            bitcoin_anchor=_parse_bitcoin_anchor(data.get("bitcoin_anchor")),
            required_co_signers=data.get("required_co_signers"),
            co_signatures=data.get("co_signatures"),
            countersign_url=data.get("countersign_url"),
            user_intent_verified=data.get("user_intent_verified"),
            decision=(
                data.get("decision")
                or (
                    _map_policy_decision_to_decision(data.get("policy_decision"))
                    if data.get("compliance_mode")
                    else None
                )
            ),
        )

    def sign_batch(
        self,
        actions: list[dict[str, Any]],
    ) -> list[SignatureResponse | None]:
        """Sign multiple actions in one HTTP roundtrip.

        Each item in ``actions`` is a dict with ``action_type`` (required) and
        optional ``context``. The server signs each action through the full
        single-sign pipeline. One bad action does not abort the batch.

        Returns a list parallel to ``actions``. Each element is either a
        SignatureResponse on success or None on failure. To inspect failures,
        compare positions to the original input.

        Args:
            actions: List of {"action_type": str, "context": dict | None}.
                    Max 100 per call.

        Returns:
            List of SignatureResponse or None, same length as ``actions``.
        """
        if not actions:
            raise ValueError("actions must contain at least one item")
        if len(actions) > 100:
            raise ValueError("actions must contain at most 100 items per call")

        body_actions: list[dict[str, Any]] = []
        for a in actions:
            if "action_type" not in a:
                raise ValueError("each action must include 'action_type'")
            # Same per-item builder as sign(): the context is digested locally.
            body_actions.append(
                _build_sign_body(
                    action_type=resolve_pattern(a["action_type"]),
                    context=a.get("context") or {},
                    session_id=self._session_id,
                    agent_id=self.agent_id,
                )
            )

        data = _transport._post(
            f"/agents/{self.agent_id}/sign-batch",
            {"actions": body_actions},
        )

        results: list[SignatureResponse | None] = []
        for sig in data.get("signatures", []):
            if sig is None:
                results.append(None)
            else:
                results.append(
                    SignatureResponse(
                        signature=require_field(sig, "signature"),
                        signature_id=require_field(sig, "signature_id"),
                        action_id=require_field(sig, "action_id"),
                        timestamp=require_field(sig, "timestamp"),
                        verification_url=require_field(sig, "verification_url"),
                        algorithm=sig.get("algorithm"),
                        chain_hash=sig.get("chain_hash") or sig.get("record_hash"),
                        rfc3161_tsa=(sig.get("rfc3161_timestamp") or {}).get("tsa"),
                        rfc3161_serial=(sig.get("rfc3161_timestamp") or {}).get("serial_number"),
                        scan_result=sig.get("scan_result"),
                        policy_digest=sig.get("policy_digest"),
                        policy_decision=sig.get("policy_decision", "permit"),
                        authorization_ref=sig.get("authorization_ref"),
                        bitcoin_anchor=_parse_bitcoin_anchor(sig.get("bitcoin_anchor")),
                        decision=(
                            sig.get("decision")
                            or (
                                _map_policy_decision_to_decision(sig.get("policy_decision"))
                                if sig.get("compliance_mode")
                                else None
                            )
                        ),
                    )
                )
        return results

    def sign_output(
        self,
        action_type: str,
        input_hash: str,
        output: Any,
        context: dict[str, Any] | None = None,
    ) -> SignatureResponse:
        """Sign a tool output with a hash of the original input for binding.

        Creates a signed record that binds the output to a specific input,
        proving the output hasn't been modified after generation.

        Args:
            action_type: Type of action (e.g., "tool:search", "api:call").
            input_hash: SHA-256 hash of the original input for binding.
            output: The output to hash and sign.
            context: Additional context to include in the signed payload.

        Returns:
            SignatureResponse with the signature.
        """
        try:
            output_hash = _hash_value(output)
        except Exception:
            import warnings

            warnings.warn("asqav: failed to hash output, signing without output_hash")
            output_hash = ""

        sign_context: dict[str, Any] = {
            "output_hash": output_hash,
            "input_hash": input_hash,
        }
        if context:
            sign_context.update(context)

        return self.sign(
            action_type=action_type,
            context=sign_context,
            result=output,
        )

    def sign_with_phases(
        self,
        action_type: str,
        context: dict[str, Any] | None = None,
        trace_id: str | None = None,
        result: Any | None = None,
    ) -> "PhaseChain":
        """Three-phase flow; result binds a digest on :execution (787)."""
        from ..phases import sign_with_phases

        return sign_with_phases(self, action_type, context, trace_id, result=result)

    def attach_result(
        self,
        action_ref: str,
        result: bytes | str | dict | list,
        *,
        action_type: str = "protectmcp:observation",
        tool_name: str | None = None,
        result_digest: str | None = None,
        compliance_mode: bool = True,
    ) -> SignatureResponse:
        """Sign a follow-up result receipt linked to an earlier receipt (787).
        Uses ``result_bound`` and ``action_ref``; ``result_digest`` wins."""
        return self.sign(
            action_type,
            result=result,
            result_digest=result_digest,
            receipt_type="protectmcp:observation:result_bound",
            action_ref=action_ref,
            tool_name=tool_name,
            compliance_mode=compliance_mode,
        )
