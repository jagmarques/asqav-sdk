"""Acceptor-side admission control for an inbound peer receipt.
Adds expiry, seq-downgrade and challenge rules on top of the shared verifier."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from . import strict_json
from .verifier.oracle import ADAPTERS, verify
from .verifier.oracle.core import VERDICT_VERIFIED, VERDICT_VERIFIED_KEYED
from .verifier.verify_receipt import _parse_stamp

__all__ = [
    "DEFAULT_RECEIPT_HEADER",
    "AcceptorDecision",
    "AcceptorMiddleware",
    "check_peer_receipt",
    "encode_receipt_header",
]

#: Verdicts an acceptor may admit; `verified_keyed` counts (own keyed digest
#: is consistent, but not third-party re-derivable).
_ADMISSIBLE_VERDICTS = frozenset({VERDICT_VERIFIED, VERDICT_VERIFIED_KEYED})


@dataclass(frozen=True)
class AcceptorDecision:
    """Whether to admit an inbound action, and the single reason why not.
    ``first_failing_edge`` is the first failed check; ``rule`` the refuser."""

    accepted: bool
    reason: str
    verdict: str
    failure_class: str | None = None
    first_failing_edge: str | None = None
    rule: str | None = None


# _parse_stamp is the verifier's canonical stamp parser (imported above): one
# grammar for every time axis, so expiry refuses exactly what skew refuses
def _payload_of(receipt: dict) -> dict:
    payload = receipt.get("payload")
    return payload if isinstance(payload, dict) else receipt


def check_peer_receipt(
    receipt: dict,
    *,
    key_provider: Any = None,
    predecessor: dict | None = None,
    challenge: str | None = None,
    now: datetime | None = None,
) -> AcceptorDecision:
    """Decide whether an inbound action carrying ``receipt`` may be admitted.
    Refuses on the first failing rule, in a fixed order."""
    result = verify(receipt, ADAPTERS, key_provider=key_provider, predecessor=predecessor)

    if result.verdict not in _ADMISSIBLE_VERDICTS:
        edge = result.first_failing_edge
        return AcceptorDecision(
            accepted=False,
            reason=f"peer receipt did not verify at {edge or 'an unnamed check'}",
            verdict=result.verdict,
            failure_class=result.failure_class,
            first_failing_edge=edge,
            rule="verifier",
        )

    payload = _payload_of(receipt)

    # Expiry, which the verdict deliberately does not carry (criterion 426).
    expires_at = payload.get("expires_at")
    if expires_at is not None:
        stamp = _parse_stamp(expires_at)
        if stamp is None:
            return AcceptorDecision(
                accepted=False,
                reason=(
                    f"unreadable expires_at {expires_at!r}; refused rather "
                    "than read as no expiry"
                ),
                verdict=result.verdict,
                failure_class="unverifiable",
                first_failing_edge="expiry",
                rule="expiry",
            )
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        if current > stamp:
            return AcceptorDecision(
                accepted=False,
                reason=f"peer receipt expired at {expires_at}",
                verdict=result.verdict,
                failure_class="invalid",
                first_failing_edge="expiry",
                rule="expiry",
            )

    # A peer that carried a counter and stopped is the one case where absence is
    # not the legacy case: it is the transition that makes a gap uncheckable.
    if predecessor is not None:
        prev_seq = _payload_of(predecessor).get("seq")
        if isinstance(prev_seq, int) and not isinstance(payload.get("seq"), int):
            return AcceptorDecision(
                accepted=False,
                reason=(
                    f"peer stopped emitting seq after {prev_seq}; contiguity cannot "
                    "be checked across this link"
                ),
                verdict=result.verdict,
                failure_class="unverifiable",
                first_failing_edge="seq",
                rule="seq_downgrade",
            )

    # A challenge that goes unanswered proved nothing, so requiring it is the
    # whole point of having issued one.
    if challenge is not None:
        answered = payload.get("challenge_nonce")
        if answered is None:
            return AcceptorDecision(
                accepted=False,
                reason="acceptor issued a challenge and the receipt answers none",
                verdict=result.verdict,
                failure_class="unverifiable",
                first_failing_edge="challenge_nonce",
                rule="challenge",
            )
        if answered != challenge:
            return AcceptorDecision(
                accepted=False,
                reason="receipt answers a different challenge than the one issued",
                verdict=result.verdict,
                failure_class="invalid",
                first_failing_edge="challenge_nonce",
                rule="challenge",
            )

    return AcceptorDecision(
        accepted=True,
        reason="peer receipt verified and satisfies every acceptor rule",
        verdict=result.verdict,
        failure_class=None,
        first_failing_edge=None,
    )


# --- ASGI adapter, no framework dependency. The decision
# stays in check_peer_receipt, so mounting this and calling it directly agree.

#: Header the peer presents its receipt in, JSON or base64-of-JSON.
DEFAULT_RECEIPT_HEADER = "x-asqav-receipt"


def encode_receipt_header(receipt: dict) -> str:
    """Encode a receipt for the ``x-asqav-receipt`` header a peer's acceptor reads.

    Base64 over the JSON text, so the header stays one ASCII token; the
    acceptor decodes it back to the same object (raw JSON is accepted too).
    """
    text = json.dumps(receipt, separators=(",", ":"), ensure_ascii=False)
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _decode_receipt(raw: bytes) -> dict | None:
    """Parse a receipt header, accepting raw JSON or base64-wrapped JSON."""
    text = raw.decode("utf-8", errors="replace").strip()
    for candidate in (text, None):
        if candidate is None:
            try:
                text = base64.b64decode(raw, validate=True).decode("utf-8")
            except Exception:
                return None
            candidate = text
        try:
            # Strict loader: bare json.loads would admit out-of-domain integers
            parsed = strict_json.loads(candidate)
        except (ValueError, TypeError):
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


class AcceptorMiddleware:
    """ASGI middleware refusing a request whose peer receipt is not admissible.
    Fails CLOSED; ``predecessor_for``/``challenge_for`` are caller-supplied."""

    def __init__(
        self,
        app: Any,
        *,
        key_provider: Any = None,
        header: str = DEFAULT_RECEIPT_HEADER,
        predecessor_for: Any = None,
        challenge_for: Any = None,
        status_code: int = 403,
        exempt_paths: tuple[str, ...] = (),
    ) -> None:
        self.app = app
        self.key_provider = key_provider
        self.header = header.lower().encode("latin-1")
        self.predecessor_for = predecessor_for
        self.challenge_for = challenge_for
        self.status_code = status_code
        self.exempt_paths = exempt_paths

    async def _refuse(self, send: Any, reason: str, rule: str | None) -> None:
        body = json.dumps(
            {"error": "peer_receipt_refused", "reason": reason, "rule": rule}
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("latin-1")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        # Lifespan and websocket scopes carry no receipt to check; passing them
        # through is not a bypass because no inbound ACTION rides on them.
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("path", "") in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        raw = None
        for name, value in scope.get("headers") or []:
            if name.lower() == self.header:
                raw = value
                break
        if raw is None:
            await self._refuse(send, "no peer receipt presented", "missing")
            return

        receipt = _decode_receipt(raw)
        if receipt is None:
            await self._refuse(send, "peer receipt header is not a JSON object", "malformed")
            return

        decision = check_peer_receipt(
            receipt,
            key_provider=self.key_provider,
            predecessor=self.predecessor_for(receipt) if self.predecessor_for else None,
            challenge=self.challenge_for(receipt) if self.challenge_for else None,
        )
        if not decision.accepted:
            await self._refuse(send, decision.reason, decision.rule)
            return

        scope = dict(scope)
        scope.setdefault("state", {})
        scope["state"] = {**scope["state"], "asqav_acceptor_decision": decision}
        await self.app(scope, receive, send)
