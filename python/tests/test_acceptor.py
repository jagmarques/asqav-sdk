"""Gates for acceptor-side admission control (criterion 472, B15).

Driven off the real conformance corpus rather than hand-built dicts: every
receipt here is signed over its own bytes by the corpus's published keys, so a
refusal has to come from the rule under test and not from a broken signature
nobody noticed.

The three acceptor-only rules carry the weight. Each one refuses a receipt the
VERIFIER is content with, which is the whole reason the module exists - a peer
can weaken the evidence without ever producing an unverified receipt.
"""
from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from asqav.acceptor import check_peer_receipt
from asqav.verifier.oracle import ADAPTERS, verify
from asqav.verifier.oracle.core import VERDICT_VERIFIED, VERDICT_VERIFIED_KEYED

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"


def _vec(name: str):
    """A corpus vector as (receipt, jwks, predecessor|None)."""
    d = _CORPUS / name
    receipt = json.loads((d / "receipt.json").read_text())
    jwks = json.loads((d / "jwks.json").read_text())
    pred_path = d / "predecessor.json"
    predecessor = json.loads(pred_path.read_text()) if pred_path.exists() else None
    return receipt, jwks, predecessor


# --- properly signed fixtures -------------------------------------------------
#
# The acceptor rules refuse receipts the VERIFIER is content with, so testing
# them needs receipts that actually verify. Editing a corpus payload cannot do
# it: the edit breaks the signature, the verifier refuses first, and the rule
# under test never runs. So these mint real receipts with the corpus's own
# published seed, exactly as gen_seq_vectors.py does.

_SEED_PHRASE = b"asqav conformance corpus v1 seq-continuity signing seed"
_KID = "asqav-seq-vec-key"
_ISSUER = "Asqav Ltd"
_ZERO_DIGEST = hashlib.sha256(b"").hexdigest()


def _sk() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(_SEED_PHRASE).digest())


def _jcs(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _signed(action_ref: str = "act_1", previous: str = "0" * 64, **extra) -> dict:
    """A real, correctly signed asqav-native receipt carrying ``extra``."""
    payload = {
        "type": "protectmcp:decision",
        "issued_at": "2026-08-30T12:00:00+00:00",
        "issuer_id": _ISSUER,
        "agent_id": "agt_acceptor_001",
        "action_ref": action_ref,
        "payload_digest": {"hash": _ZERO_DIGEST, "size": 0},
        "policy_digest": f"sha256:{_ZERO_DIGEST}",
        "previousReceiptHash": previous,
        "decision": "allow",
        "tool_name": "demo.action",
    }
    payload.update(extra)
    return {
        "payload": payload,
        "signature": {
            "alg": "Ed25519",
            "kid": _KID,
            "sig": base64.b64encode(_sk().sign(_jcs(payload))).decode(),
        },
        "anchors": [],
    }


def _signed_jwks() -> dict:
    pub = _sk().public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return {
        "keys": [
            {
                "kid": _KID,
                "issuer_id": _ISSUER,
                "alg": "Ed25519",
                "status": "active",
                "public_key": base64.b64encode(pub).decode(),
            }
        ]
    }


def _chained(first_extra: dict, second_extra: dict):
    """A predecessor and the successor that links to it, both properly signed."""
    first = _signed("act_1", "0" * 64, **first_extra)
    link = hashlib.sha256(_jcs(first["payload"])).hexdigest()
    second = _signed("act_2", link, **second_extra)
    return second, _signed_jwks(), first


class TestAdmitsWhatVerifies:
    def test_a_clean_peer_receipt_is_admitted(self) -> None:
        receipt, jwks, pred = _vec("asqav-17-seq-contiguous")
        got = check_peer_receipt(receipt, key_provider=jwks, predecessor=pred)
        assert got.accepted, got.reason
        assert got.first_failing_edge is None
        assert got.rule is None

    def test_the_admitted_receipt_really_did_verify(self) -> None:
        """Guards against admitting on a verdict the oracle never reached.

        Without this, a bug that defaulted `accepted` to True would still pass
        the test above.
        """
        receipt, jwks, pred = _vec("asqav-17-seq-contiguous")
        result = verify(receipt, ADAPTERS, key_provider=jwks, predecessor=pred)
        assert result.verdict in (VERDICT_VERIFIED, VERDICT_VERIFIED_KEYED)


class TestRefusesWhatDoesNotVerify:
    def test_a_withheld_receipt_gap_is_refused(self) -> None:
        receipt, jwks, pred = _vec("asqav-18-seq-gap")
        got = check_peer_receipt(receipt, key_provider=jwks, predecessor=pred)
        assert not got.accepted
        assert got.first_failing_edge == "seq"
        assert got.rule == "verifier"

    def test_a_substituted_key_is_refused(self) -> None:
        """The signature verifies against the published key; only the binding fails.

        Exactly the case an acceptor must not wave through: nothing about the
        signature looks wrong.
        """
        receipt, jwks, _ = _vec("asqav-22-key-substituted")
        got = check_peer_receipt(receipt, key_provider=jwks)
        assert not got.accepted
        assert got.first_failing_edge == "key_binding"
        assert got.failure_class == "invalid"

    def test_the_refusal_names_one_edge_not_a_wall_of_axes(self) -> None:
        receipt, jwks, pred = _vec("asqav-18-seq-gap")
        got = check_peer_receipt(receipt, key_provider=jwks, predecessor=pred)
        assert got.first_failing_edge in {a.axis for a in
                                          verify(receipt, ADAPTERS, key_provider=jwks,
                                                 predecessor=pred).axes}


class TestExpiryIsAnAcceptorRule:
    """The verifier reports expiry on its own axis and never folds the verdict.

    That is right for a verifier and wrong for an acceptor, which is deciding
    about an action happening now. These tests pin the difference rather than
    assuming it, on receipts that genuinely verify.
    """

    def _expiring(self, when: datetime):
        stamp = when.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return _chained({"seq": 7}, {"seq": 8, "expires_at": stamp})

    def test_the_verifier_still_calls_an_expired_receipt_verified(self) -> None:
        """The premise the acceptor rule rests on, asserted rather than assumed.

        If the verifier ever starts folding expiry, this fails and says so
        instead of the rule below quietly becoming redundant.
        """
        doc, jwks, pred = self._expiring(datetime.now(timezone.utc) - timedelta(days=1))
        result = verify(doc, ADAPTERS, key_provider=jwks, predecessor=pred)
        expiry = result.axis("expiry")
        assert expiry is not None, "no expiry axis; the premise no longer holds"
        assert expiry.result == "FAIL", expiry.note
        assert result.verdict in (VERDICT_VERIFIED, VERDICT_VERIFIED_KEYED), (
            "the verifier folded expiry into the verdict; the acceptor rule is moot"
        )

    def test_an_expired_receipt_is_refused_by_the_acceptor(self) -> None:
        doc, jwks, pred = self._expiring(datetime.now(timezone.utc) - timedelta(days=1))
        got = check_peer_receipt(doc, key_provider=jwks, predecessor=pred)
        assert not got.accepted
        assert got.rule == "expiry", got.reason

    def test_an_unexpired_receipt_is_admitted(self) -> None:
        """The control. Without it the rule could refuse everything and pass."""
        doc, jwks, pred = self._expiring(datetime.now(timezone.utc) + timedelta(days=1))
        got = check_peer_receipt(doc, key_provider=jwks, predecessor=pred)
        assert got.accepted, got.reason

    def test_the_caller_supplied_clock_is_what_decides(self) -> None:
        """Pins that `now` is honoured, so the rule is testable without sleeping
        and an operator can reason about skew."""
        stamp = datetime(2030, 1, 1, tzinfo=timezone.utc)
        doc, jwks, pred = self._expiring(stamp)
        before = check_peer_receipt(
            doc, key_provider=jwks, predecessor=pred, now=stamp - timedelta(days=1)
        )
        after = check_peer_receipt(
            doc, key_provider=jwks, predecessor=pred, now=stamp + timedelta(days=1)
        )
        assert before.accepted, before.reason
        assert not after.accepted and after.rule == "expiry"

    def test_an_unreadable_expiry_is_refused_rather_than_ignored(self) -> None:
        doc, jwks, pred = _chained({"seq": 7}, {"seq": 8, "expires_at": "not-a-timestamp"})
        got = check_peer_receipt(doc, key_provider=jwks, predecessor=pred)
        assert not got.accepted
        assert got.rule == "expiry", got.reason


class TestSeqDowngrade:
    """A peer that carried a counter and stops is the one case where absence is
    not the legacy case: it is the transition that makes a gap uncheckable."""

    def test_dropping_seq_after_carrying_one_is_refused(self) -> None:
        doc, jwks, pred = _chained({"seq": 7}, {})
        assert isinstance(pred["payload"].get("seq"), int)
        assert doc["payload"].get("seq") is None
        got = check_peer_receipt(doc, key_provider=jwks, predecessor=pred)
        assert not got.accepted
        assert got.rule == "seq_downgrade", got.reason

    def test_a_peer_that_never_carried_seq_is_still_admitted(self) -> None:
        """Absence stays legal in general, or every receipt minted before the
        counter shipped would be refused by an acceptor."""
        doc, jwks, pred = _chained({}, {})
        got = check_peer_receipt(doc, key_provider=jwks, predecessor=pred)
        assert got.accepted, got.reason

    def test_a_contiguous_pair_is_admitted(self) -> None:
        """The control for the downgrade rule: carrying a counter is not itself
        grounds for refusal."""
        doc, jwks, pred = _chained({"seq": 7}, {"seq": 8})
        got = check_peer_receipt(doc, key_provider=jwks, predecessor=pred)
        assert got.accepted, got.reason

    def test_no_predecessor_means_no_downgrade_verdict(self) -> None:
        """With nothing to compare against, silence is not evidence."""
        doc = _signed()
        got = check_peer_receipt(doc, key_provider=_signed_jwks())
        assert got.rule != "seq_downgrade"


class TestChallenge:
    """A challenge the acceptor issued but the receipt does not answer proved
    nothing, so 'verify it when present' alone would let a peer skip freshness."""

    def test_an_unanswered_challenge_is_refused(self) -> None:
        doc, jwks, pred = _chained({"seq": 7}, {"seq": 8})
        got = check_peer_receipt(
            doc, key_provider=jwks, predecessor=pred, challenge="chal-abc"
        )
        assert not got.accepted
        assert got.rule == "challenge", got.reason

    def test_the_wrong_challenge_is_refused(self) -> None:
        doc, jwks, pred = _chained({"seq": 7}, {"seq": 8, "challenge_nonce": "chal-WRONG"})
        got = check_peer_receipt(
            doc, key_provider=jwks, predecessor=pred, challenge="chal-abc"
        )
        assert not got.accepted
        assert got.rule == "challenge"
        assert got.failure_class == "invalid"

    def test_the_matching_challenge_is_admitted(self) -> None:
        doc, jwks, pred = _chained({"seq": 7}, {"seq": 8, "challenge_nonce": "chal-abc"})
        got = check_peer_receipt(
            doc, key_provider=jwks, predecessor=pred, challenge="chal-abc"
        )
        assert got.accepted, got.reason

    def test_an_unsolicited_challenge_answer_is_not_grounds_for_refusal(self) -> None:
        """A receipt may carry a nonce for a peer that issued one; an acceptor
        that issued none has nothing to compare and must not invent a mismatch."""
        doc, jwks, pred = _chained({"seq": 7}, {"seq": 8, "challenge_nonce": "chal-other"})
        got = check_peer_receipt(doc, key_provider=jwks, predecessor=pred)
        assert got.accepted, got.reason

    def test_no_issued_challenge_means_no_challenge_refusal(self) -> None:
        doc, jwks, pred = _chained({"seq": 7}, {"seq": 8})
        got = check_peer_receipt(doc, key_provider=jwks, predecessor=pred)
        assert got.rule != "challenge"


class TestRuleOrderIsDeterministic:
    def test_the_verifier_refusal_precedes_the_acceptor_rules(self) -> None:
        """A receipt that fails verification AND is expired must report the
        verifier, so the reason for the same inputs never depends on evaluation
        order."""
        doc, jwks, pred = _chained(
            {"seq": 7}, {"seq": 11, "expires_at": "2000-01-01T00:00:00Z"}
        )
        got = check_peer_receipt(doc, key_provider=jwks, predecessor=pred)
        assert not got.accepted
        assert got.rule == "verifier", got.reason
        assert got.first_failing_edge == "seq"

    @pytest.mark.parametrize("_run", range(3))
    def test_the_same_inputs_give_the_same_decision(self, _run: int) -> None:
        receipt, jwks, pred = _vec("asqav-18-seq-gap")
        first = check_peer_receipt(receipt, key_provider=jwks, predecessor=pred)
        again = check_peer_receipt(receipt, key_provider=jwks, predecessor=pred)
        assert first == again
