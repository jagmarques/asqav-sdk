"""Acceptor-side admission control for an inbound peer receipt.

An Acceptor is the party on the receiving end of an agent-to-agent action. This
module answers one question: may this inbound action be admitted, given the
receipt the peer presented?

It is deliberately NOT a thin wrapper over the verifier. Three of its rules do
not follow from the verdict alone, and each exists because a peer could
otherwise weaken the evidence without ever producing an unverified receipt:

  Expiry.   The verifier reports expiry on its own axis and never folds it into
            the verdict, so a lapsed receipt still reads `verified`. That is
            correct for a verifier - the signature really is good - and wrong for
            an acceptor, which is deciding about an action happening NOW.

  Seq downgrade. A peer that has been emitting a counter and then stops emitting
            one makes contiguity uncheckable across that link. Absence has to
            stay legal in general (receipts predate the member), but an acceptor
            holding a predecessor that carried one is watching the exact
            transition that hides a withheld receipt.

  Challenge. A challenge the acceptor issued but the receipt does not answer is
            a challenge that proved nothing. "Verify it when present" alone
            would let a peer skip freshness by simply omitting the member.

The verification itself is the shared oracle's, not a reimplementation, so an
acceptor and an offline auditor cannot disagree about the same bytes.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .verifier.oracle import ADAPTERS, verify
from .verifier.oracle.core import VERDICT_VERIFIED, VERDICT_VERIFIED_KEYED

__all__ = [
    "DEFAULT_RECEIPT_HEADER",
    "AcceptorDecision",
    "AcceptorMiddleware",
    "check_peer_receipt",
]

#: Verdicts an acceptor may admit at all. `verified_keyed` is included because a
#: keyed digest is internally consistent; it is the peer's own hash, so it proves
#: the same binding to the acceptor while not being third-party re-derivable.
_ADMISSIBLE_VERDICTS = frozenset({VERDICT_VERIFIED, VERDICT_VERIFIED_KEYED})


@dataclass(frozen=True)
class AcceptorDecision:
    """Whether to admit an inbound action, and the single reason why not.

    ``first_failing_edge`` names the earliest check that stopped the receipt, so
    a refusal points at one edge rather than handing back a wall of axes.
    ``rule`` names which acceptor rule refused, and is ``"verifier"`` when the
    receipt simply did not verify.
    """

    accepted: bool
    reason: str
    verdict: str
    failure_class: str | None = None
    first_failing_edge: str | None = None
    rule: str | None = None


def _parse_stamp(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


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

    ``predecessor`` is the last receipt this acceptor admitted from the same
    peer chain; supplying it is what lets the seq and chain axes mean anything,
    since neither can detect a gap against nothing. ``challenge`` is the nonce
    this acceptor issued for this exchange, if it issued one.

    Refuses on the first rule that fails, in a fixed order, so the reason is
    deterministic for the same inputs.
    """
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


# --- ASGI adapter -------------------------------------------------------------
#
# ASGI is a protocol, not a library, so this turns the decision into deployable
# middleware for FastAPI/Starlette/Quart with no dependency on any of them. The
# decision stays in check_peer_receipt: an acceptor that mounts this and one that
# calls the function directly must refuse the same receipts.

#: Header the peer presents its receipt in, JSON or base64-of-JSON.
DEFAULT_RECEIPT_HEADER = "x-asqav-receipt"


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
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


class AcceptorMiddleware:
    """ASGI middleware refusing a request whose peer receipt is not admissible.

    Fails CLOSED: a request carrying no receipt, or one whose header does not
    parse, is refused. An acceptor that admitted an unsigned request while
    refusing a badly-signed one would be strictly worse than having no middleware
    at all, because the cheapest bypass would be to send nothing.

    ``predecessor_for`` supplies the last receipt admitted from the same peer
    chain; without it the seq and chain axes have nothing to compare against and
    a gap cannot be detected. ``challenge_for`` supplies the nonce this acceptor
    issued for the exchange, if any. Both take the parsed receipt and are
    deliberately caller-supplied: where that state lives is the deployer's
    choice, and guessing it here would be wrong for most of them.
    """

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
