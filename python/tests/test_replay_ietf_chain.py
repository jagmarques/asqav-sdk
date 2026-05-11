"""Tests for the IETF v2 chain shape in `asqav.replay`.

The chain link is ``sha256(JCS(predecessor_signed_envelope))``. The
first record's ``previousReceiptHash`` is ``"0" * 64``
(``FIRST_RECEIPT_SEED``).
"""

from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav._jcs import canonical_json
from asqav.client import SignedActionResponse
from asqav.replay import (
    FIRST_RECEIPT_SEED,
    ReplayTimeline,
    _build_timeline,
    _verify_hash_chain,
)

# === Fixtures ===


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


# === FIRST_RECEIPT_SEED matches the spec ===


def test_first_receipt_seed_is_64_zero_hex() -> None:
    """`FIRST_RECEIPT_SEED == "0" * 64` per §5.7."""
    assert FIRST_RECEIPT_SEED == "0" * 64
    assert len(FIRST_RECEIPT_SEED) == 64
    assert all(c == "0" for c in FIRST_RECEIPT_SEED)


# === IETF v2 default behaviour ===


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



# === _step_chain_hash helper ===





# === _step_chain_hash helper ===





# === Empty timeline ===


def test_empty_timeline_verifies() -> None:
    timeline = ReplayTimeline(agent_id="agent_001", session_id="sess_1")
    assert timeline.verify_chain() is True
    assert timeline.compliance_chain_valid is True


# === H-NEW-1: signed_envelope path verifies byte-for-byte against the cloud ===


def _compliance_envelope(
    *,
    idx: int,
    prev_hash: str,
    policy_decision: str = "allow",
) -> dict:
    """Build the cloud-shape envelope the SDK records under compliance_mode."""
    return {
        "action_id": f"act_{idx:03d}",
        "agent_id": "agent_001",
        "action_type": "api:call",
        "context": {"step": idx},
        "timestamp": "2026-05-04T00:00:00",
        "policy_digest": "sha256:abc",
        "policy_decision": policy_decision,
        "authorization_ref": None,
        "type": "compliance",
        "issued_at": "2026-05-04T00:00:00",
        "issuer_id": "Asqav Test Org",
        "action_ref": "sha256:def",
        "payload_digest": "sha256:def",
        "previousReceiptHash": prev_hash,
        "receipt_type": "compliance",
    }


def test_envelope_path_matches_cloud_v2_hash_byte_for_byte() -> None:
    """When ReplayStep.signed_envelope is set, the chain hash equals
    compute_signature_record_hash_v2 byte-for-byte (sha256(JCS(env)))."""
    env0 = _compliance_envelope(idx=0, prev_hash=FIRST_RECEIPT_SEED)
    env1_prev = hashlib.sha256(canonical_json(env0)).hexdigest()
    env1 = _compliance_envelope(idx=1, prev_hash=env1_prev)

    sigs = [_make_signed_action(idx=i) for i in range(2)]
    timeline = _build_timeline(
        "agent_001", "sess_1", sigs, signed_envelopes=[env0, env1]
    )
    assert timeline.steps[0].signed_envelope == env0
    assert timeline.steps[1].signed_envelope == env1
    assert timeline.steps[1].prev_chain_hash == env1_prev
    assert timeline.verify_chain() is True
    assert timeline.compliance_chain_valid is True


def test_envelope_path_detects_field_tamper_outside_synthetic_shape() -> None:
    """Tampering `policy_decision` (a field NOT in the synthetic envelope)
    is caught only when signed_envelope is attached. Without it the
    synthetic shape would silently pass."""
    env0 = _compliance_envelope(idx=0, prev_hash=FIRST_RECEIPT_SEED)
    env1_prev = hashlib.sha256(canonical_json(env0)).hexdigest()
    env1 = _compliance_envelope(idx=1, prev_hash=env1_prev)

    sigs = [_make_signed_action(idx=i) for i in range(2)]
    timeline = _build_timeline(
        "agent_001", "sess_1", sigs, signed_envelopes=[env0, env1]
    )

    # Tamper: flip policy_decision on step 0's envelope.
    timeline.steps[0].signed_envelope = {**env0, "policy_decision": "deny"}

    # verify_chain re-derives prev_hash from the (tampered) envelope and
    # the stored prev_chain_hash on step 1 no longer matches.
    assert timeline.verify_chain() is False
    assert timeline.steps[1].chain_valid is False
    assert timeline.compliance_chain_valid is False


def test_synthetic_shape_misses_policy_decision_tamper() -> None:
    """Documents the weakness the H-NEW-1 fix closes: without
    signed_envelope, a `policy_decision` flip is invisible because the
    synthetic envelope does not include it."""
    sigs = [_make_signed_action(idx=i) for i in range(2)]
    timeline = _build_timeline("agent_001", "sess_1", sigs)
    # No envelopes attached; chain verifies under synthetic shape.
    assert timeline.verify_chain() is True
    # But compliance_chain_valid is False because envelopes were absent.
    assert timeline.compliance_chain_valid is False
    assert all(s.signed_envelope is None for s in timeline.steps)


def test_mixed_envelope_steps_drop_compliance_chain_valid() -> None:
    """A timeline with envelopes on some steps and not others drops
    compliance_chain_valid."""
    env0 = _compliance_envelope(idx=0, prev_hash=FIRST_RECEIPT_SEED)

    sigs = [_make_signed_action(idx=i) for i in range(2)]
    timeline = _build_timeline(
        "agent_001", "sess_1", sigs, signed_envelopes=[env0, None]
    )
    timeline.verify_chain()
    assert timeline.steps[0].signed_envelope is not None
    assert timeline.steps[1].signed_envelope is None
    assert timeline.compliance_chain_valid is False


def test_to_dict_includes_signed_envelope_and_compliance_flag() -> None:
    """Round-trip metadata so audit-pack consumers see envelope + flag."""
    env0 = _compliance_envelope(idx=0, prev_hash=FIRST_RECEIPT_SEED)
    sigs = [_make_signed_action(idx=0)]
    timeline = _build_timeline(
        "agent_001", "sess_1", sigs, signed_envelopes=[env0]
    )
    d = timeline.to_dict()
    assert d["compliance_chain_valid"] is True
    assert d["steps"][0]["signed_envelope"] == env0

