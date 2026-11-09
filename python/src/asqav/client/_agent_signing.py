"""Agent methods built on sign: countersign, batches, output."""

from __future__ import annotations

from typing import Any

from ..patterns import resolve_pattern
from ..phases import sign_with_phases as _sign_with_phases
from . import _transport
from ._agent_sign import _signature_response
from ._common import _hash_value
from ._models import SignatureResponse
from ._sign_body import _build_sign_body


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
        return _signature_response(data)

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
                results.append(_signature_response(sig))
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

    sign_with_phases = _sign_with_phases

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
