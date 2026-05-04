"""Tests for the IETF v2 chain shape in `asqav.replay`.

Spec: draft-marques-asqav-compliance-receipts-00 §5.7.

The chain link is ``sha256(JCS(predecessor_signed_envelope))``. The
first record's ``previousReceiptHash`` is ``"0" * 64``
(``FIRST_RECEIPT_SEED``). The legacy shape is kept available behind
``legacy_chain=True`` so historic bundles still verify.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav._jcs import canonical_json
from asqav.client import SignedActionResponse
from asqav.replay import (
    FIRST_RECEIPT_SEED,
    ReplayStep,
    ReplayTimeline,
    _build_timeline,
    _step_chain_hash,
    _verify_hash_chain,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_signed_action(
    *, idx: int = 0, action_type: str = "api:call", payload=None
) -> SignedActionResponse:
    return SignedActionResponse(
        signature_id=f"sid_{idx:03d}",
        agent_id="agent_001",
        action_id=f"act_{idx:03d}",
        action_type=action_type,
        payload=payload or {"step": idx},
        algorithm="ML-DSA-65",
        signed_at=1700000000.0 + idx * 10,
        signature_preview="sig_b64",
        verification_url=f"https://asqav.com/verify/sid_{idx:03d}",
    )


# ---------------------------------------------------------------------------
# FIRST_RECEIPT_SEED matches the spec
# ---------------------------------------------------------------------------


def test_first_receipt_seed_is_64_zero_hex() -> None:
    """`FIRST_RECEIPT_SEED == "0" * 64` per §5.7."""
    assert FIRST_RECEIPT_SEED == "0" * 64
    assert len(FIRST_RECEIPT_SEED) == 64
    assert all(c == "0" for c in FIRST_RECEIPT_SEED)


# ---------------------------------------------------------------------------
# IETF v2 default behaviour
# ---------------------------------------------------------------------------


def test_v2_chain_hash_matches_jcs_of_envelope() -> None:
    """Each link is `sha256(JCS(envelope_with_prev_link))`."""
    sigs = [_make_signed_action(idx=0), _make_signed_action(idx=1)]
    sig_dicts = [
        {
            "signature_id": s.signature_id,
            "action_type": s.action_type,
            "payload": s.payload,
            "signed_at": s.signed_at,
        }
        for s in sigs
    ]
    _verify_hash_chain(sig_dicts)  # smoke

    # Recompute by hand and compare against the timeline's stored prev hashes.
    timeline = _build_timeline("agent_001", "sess_1", sigs)
    assert len(timeline.steps) == 2

    expected_first_hash = hashlib.sha256(
        canonical_json(
            {
                "signature_id": sigs[0].signature_id,
                "action_type": sigs[0].action_type,
                "context": sigs[0].payload,
                "timestamp": sigs[0].signed_at,
                "previousReceiptHash": FIRST_RECEIPT_SEED,
            }
        )
    ).hexdigest()
    assert timeline.steps[1].prev_chain_hash == expected_first_hash


def test_v2_seed_is_used_under_default() -> None:
    """The first envelope hashes against the all-zero seed, not empty."""
    sigs = [_make_signed_action(idx=0)]
    timeline = _build_timeline("agent_001", "sess_1", sigs)
    # First step has no predecessor -> prev_chain_hash is None.
    assert timeline.steps[0].prev_chain_hash is None

    # But the implicit seed used in derivation IS the spec value:
    expected_first_hash = hashlib.sha256(
        canonical_json(
            {
                "signature_id": sigs[0].signature_id,
                "action_type": sigs[0].action_type,
                "context": sigs[0].payload,
                "timestamp": sigs[0].signed_at,
                "previousReceiptHash": FIRST_RECEIPT_SEED,
            }
        )
    ).hexdigest()

    # If we add a second sig and rebuild, its prev_chain_hash should be
    # exactly that hash.
    sigs.append(_make_signed_action(idx=1))
    timeline = _build_timeline("agent_001", "sess_1", sigs)
    assert timeline.steps[1].prev_chain_hash == expected_first_hash


def test_v2_verify_chain_succeeds_on_freshly_built_timeline() -> None:
    sigs = [_make_signed_action(idx=i) for i in range(4)]
    timeline = _build_timeline("agent_001", "sess_1", sigs)
    assert timeline.verify_chain() is True
    assert timeline.chain_integrity is True
    for step in timeline.steps:
        assert step.chain_valid is True


def test_v2_verify_chain_detects_tamper() -> None:
    """If a step's stored prev_chain_hash is altered, verify_chain fails."""
    sigs = [_make_signed_action(idx=i) for i in range(3)]
    timeline = _build_timeline("agent_001", "sess_1", sigs)
    # Tamper: corrupt the link on step 1.
    timeline.steps[1].prev_chain_hash = "f" * 64
    assert timeline.verify_chain() is False
    assert timeline.steps[1].chain_valid is False


def test_v2_chain_differs_from_legacy_for_same_signatures() -> None:
    """The v2 envelope shape produces different bytes than the legacy
    envelope so a forged legacy chain cannot pass v2 verification."""
    sigs = [_make_signed_action(idx=i) for i in range(3)]
    v2 = _build_timeline("agent_001", "sess_1", sigs, legacy_chain=False)
    legacy = _build_timeline("agent_001", "sess_1", sigs, legacy_chain=True)
    # First step has no predecessor in either (None), so compare step 1.
    assert v2.steps[1].prev_chain_hash != legacy.steps[1].prev_chain_hash


# ---------------------------------------------------------------------------
# legacy_chain flag
# ---------------------------------------------------------------------------


def test_legacy_chain_uses_old_envelope_shape() -> None:
    """legacy_chain=True reproduces the pre-IETF chain hash byte-for-byte."""
    sigs = [_make_signed_action(idx=0), _make_signed_action(idx=1)]
    timeline = _build_timeline(
        "agent_001", "sess_1", sigs, legacy_chain=True
    )

    expected = hashlib.sha256(
        json.dumps(
            {
                "signature_id": sigs[0].signature_id,
                "action_type": sigs[0].action_type,
                "timestamp": sigs[0].signed_at,
                "prev_hash": "",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert timeline.steps[1].prev_chain_hash == expected


def test_legacy_chain_verify_succeeds() -> None:
    sigs = [_make_signed_action(idx=i) for i in range(3)]
    timeline = _build_timeline(
        "agent_001", "sess_1", sigs, legacy_chain=True
    )
    assert timeline.verify_chain(legacy_chain=True) is True


def test_legacy_chain_verify_fails_when_called_under_v2() -> None:
    """A timeline built under legacy shape does not verify under v2."""
    sigs = [_make_signed_action(idx=i) for i in range(3)]
    timeline = _build_timeline(
        "agent_001", "sess_1", sigs, legacy_chain=True
    )
    # Timeline stores legacy hashes. Verifying under v2 must reject.
    assert timeline.verify_chain(legacy_chain=False) is False


# ---------------------------------------------------------------------------
# _step_chain_hash helper
# ---------------------------------------------------------------------------


def test_step_chain_hash_v2_matches_envelope() -> None:
    step = ReplayStep(
        index=0,
        action_type="api:call",
        context={"x": 1},
        timestamp=1700000000.0,
        signature_id="sid_001",
        verification_url="https://asqav.com/verify/sid_001",
        chain_valid=True,
        explanation="x",
    )
    expected = hashlib.sha256(
        canonical_json(
            {
                "signature_id": "sid_001",
                "action_type": "api:call",
                "context": {"x": 1},
                "timestamp": 1700000000.0,
                "previousReceiptHash": FIRST_RECEIPT_SEED,
            }
        )
    ).hexdigest()
    assert (
        _step_chain_hash(step, FIRST_RECEIPT_SEED, legacy_chain=False)
        == expected
    )


def test_step_chain_hash_legacy_matches_old_shape() -> None:
    step = ReplayStep(
        index=0,
        action_type="api:call",
        context={"x": 1},
        timestamp=1700000000.0,
        signature_id="sid_001",
        verification_url="x",
        chain_valid=True,
        explanation="x",
    )
    expected = hashlib.sha256(
        json.dumps(
            {
                "signature_id": "sid_001",
                "action_type": "api:call",
                "timestamp": 1700000000.0,
                "prev_hash": "",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert _step_chain_hash(step, "", legacy_chain=True) == expected


# ---------------------------------------------------------------------------
# Empty timeline
# ---------------------------------------------------------------------------


def test_empty_timeline_verifies_under_both_modes() -> None:
    timeline = ReplayTimeline(agent_id="agent_001", session_id="sess_1")
    assert timeline.verify_chain() is True
    assert timeline.verify_chain(legacy_chain=True) is True
