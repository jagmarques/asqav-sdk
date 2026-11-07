"""Agent methods that issue tokens and acknowledgments."""

from __future__ import annotations

from typing import Any

from ..counterparty import ACKNOWLEDGMENT_RECEIPT_TYPE, compute_counterparty_binding
from . import _transport
from ._common import _parse_timestamp
from ._models import SDTokenResponse, SignatureResponse, TokenResponse


class _AgentTokens:
    def issue_token(
        self,
        scope: list[str] | None = None,
        ttl: int = 3600,
    ) -> TokenResponse:
        """Issue a PQC-JWT token for this agent.

        The token is signed server-side with ML-DSA.

        Args:
            scope: Capabilities to include (default: all).
            ttl: Token time-to-live in seconds.

        Returns:
            TokenResponse with the signed token.
        """
        data = _transport._post(
            f"/agents/{self.agent_id}/tokens",
            {
                "scope": scope or self.capabilities,
                "ttl": ttl,
            },
        )

        return TokenResponse(
            token=data["token"],
            expires_at=_parse_timestamp(data["expires_at"]),
            algorithm=data["algorithm"],
        )

    def issue_sd_token(
        self,
        claims: dict[str, Any] | None = None,
        disclosable: list[str] | None = None,
        ttl: int = 3600,
    ) -> SDTokenResponse:
        """Issue a PQC-SD-JWT token with selective disclosure (Enterprise tier).

        SD-JWT tokens allow agents to selectively reveal claims when
        presenting the token to external services, maintaining privacy.

        Args:
            claims: Claims to include in the token.
            disclosable: List of claim names that can be selectively disclosed.
            ttl: Token time-to-live in seconds.

        Returns:
            SDTokenResponse with the token and disclosures.

        Example:
            sd_token = agent.issue_sd_token(
                claims={"role": "admin", "org": "acme"},
                disclosable=["role", "org"]
            )

            # Present to partner - only show role
            proof = sd_token.present(["role"])
        """
        data = _transport._post(
            f"/agents/{self.agent_id}/tokens/sd",
            {
                "claims": claims or {},
                "disclosable": disclosable or [],
                "ttl": ttl,
            },
        )

        return SDTokenResponse(
            token=data["token"],
            jwt=data["jwt"],
            disclosures=data["disclosures"],
            expires_at=_parse_timestamp(data["expires_at"]),
        )

    def acknowledge(
        self,
        peer_receipt: dict[str, Any],
        *,
        receipt_ref: str | None = None,
        action_type: str = "a2a:acknowledge",
        context: dict[str, Any] | None = None,
        expect_ack_from: str | None = None,
        transport_label: str | None = None,
        profile_version: str | None = None,
        terms: dict[str, Any] | None = None,
    ) -> SignatureResponse:
        """Sign this agent's acknowledgment of a peer agent's receipt.
        ``peer_receipt`` is the peer's envelope; the binding commits to it."""
        ref = receipt_ref or peer_receipt.get("signature_id")
        if not isinstance(ref, str) or not ref:
            raise ValueError(
                "receipt_ref is required: pass the peer receipt's signature_id"
            )
        binding = compute_counterparty_binding(
            peer_receipt,
            receipt_ref=ref,
            expect_ack_from=expect_ack_from,
            transport_label=transport_label,
            profile_version=profile_version,
            terms=terms,
        )
        return self.sign(
            action_type,
            context=context,
            compliance_mode=True,
            receipt_type=ACKNOWLEDGMENT_RECEIPT_TYPE,
            counterparty_binding=binding,
            result=peer_receipt,
        )
