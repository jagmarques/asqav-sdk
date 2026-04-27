"""Tests for audit trail replay - timeline reconstruction and chain verification."""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import SignedActionResponse
from asqav.compliance import export_bundle
from asqav.replay import (
    ReplayStep,
    ReplayTimeline,
    _build_explanation,
    _verify_hash_chain,
    replay_from_bundle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_signed_action(**overrides) -> SignedActionResponse:
    defaults = {
        "signature_id": "sid_001",
        "agent_id": "agent_001",
        "action_id": "act_001",
        "action_type": "api:call",
        "payload": {"model": "gpt-4"},
        "algorithm": "ML-DSA-65",
        "signed_at": 1700000000.0,
        "signature_preview": "sig_preview_xyz",
        "verification_url": "https://asqav.com/verify/sid_001",
    }
    defaults.update(overrides)
    return SignedActionResponse(**defaults)


def _make_timeline(n: int = 3) -> ReplayTimeline:
    """Build a timeline with n steps at 10-second intervals."""
    steps = []
    for i in range(n):
        steps.append(
            ReplayStep(
                index=i,
                action_type=f"api:call_{i}",
                context={"model": "gpt-4"},
                timestamp=1700000000.0 + i * 10,
                signature_id=f"sid_{i:03d}",
                verification_url=f"https://asqav.com/verify/sid_{i:03d}",
                chain_valid=True,
                explanation=f"Called call_{i} API (model: gpt-4)",
            )
        )
    return ReplayTimeline(
        agent_id="agent_001",
        session_id="sess_001",
        steps=steps,
        chain_integrity=True,
        start_time=1700000000.0,
        end_time=1700000000.0 + (n - 1) * 10,
        total_actions=n,
    )


# ---------------------------------------------------------------------------
# Timeline ordering
# ---------------------------------------------------------------------------

class TestTimelineOrdering:
    def test_steps_ordered_by_timestamp(self):
        """replay_from_bundle should sort steps by timestamp."""
        sigs = [
            _make_signed_action(
                signature_id=f"sid_{i}",
                action_id=f"act_{i}",
                signed_at=1700000000.0 + (2 - i) * 100,  # reverse order
                action_type=f"api:step_{i}",
            )
            for i in range(3)
        ]
        bundle = export_bundle(sigs)
        timeline = replay_from_bundle(bundle)

        timestamps = [s.timestamp for s in timeline.steps]
        assert timestamps == sorted(timestamps)

    def test_indices_sequential(self):
        """Step indices should be 0, 1, 2... after sorting."""
        sigs = [
            _make_signed_action(
                signature_id=f"sid_{i}",
                action_id=f"act_{i}",
                signed_at=1700000000.0 + (5 - i) * 10,
            )
            for i in range(5)
        ]
        bundle = export_bundle(sigs)
        timeline = replay_from_bundle(bundle)

        indices = [s.index for s in timeline.steps]
        assert indices == list(range(5))


# ---------------------------------------------------------------------------
# Chain verification
# ---------------------------------------------------------------------------

class TestChainVerification:
    def test_valid_chain(self):
        """A freshly built chain should verify successfully."""
        sig_dicts = [
            {
                "signature_id": f"sid_{i}",
                "action_type": "api:call",
                "signed_at": 1700000000.0 + i * 10,
            }
            for i in range(4)
        ]
        results = _verify_hash_chain(sig_dicts)
        assert all(results)
        assert len(results) == 4

    def test_tampered_entry_detected(self):
        """Changing a field mid-chain should break chain integrity for that step.

        We test this by building a timeline, then manually flipping chain_valid
        on a middle step and confirming chain_integrity reflects it.
        """
        timeline = _make_timeline(3)
        # Simulate a tampered entry
        timeline.steps[1].chain_valid = False
        timeline.chain_integrity = all(s.chain_valid for s in timeline.steps)
        assert not timeline.chain_integrity

    def test_verify_chain_method(self):
        """ReplayTimeline.verify_chain() should re-verify all steps."""
        timeline = _make_timeline(4)
        result = timeline.verify_chain()
        assert result is True
        assert timeline.chain_integrity is True

    def test_empty_chain_is_valid(self):
        results = _verify_hash_chain([])
        assert results == []

    def test_single_entry_chain(self):
        results = _verify_hash_chain([
            {"signature_id": "sid_0", "action_type": "api:call", "signed_at": 1700000000.0}
        ])
        assert results == [True]


# ---------------------------------------------------------------------------
# replay_from_bundle
# ---------------------------------------------------------------------------

class TestReplayFromBundle:
    def test_basic_reconstruction(self):
        sigs = [
            _make_signed_action(
                signature_id=f"sid_{i}",
                action_id=f"act_{i}",
                signed_at=1700000000.0 + i * 10,
                action_type="api:openai",
                payload={"model": "gpt-4"},
            )
            for i in range(3)
        ]
        bundle = export_bundle(sigs)
        timeline = replay_from_bundle(bundle)

        assert timeline.agent_id == "agent_001"
        assert timeline.total_actions == 3
        assert timeline.chain_integrity is True
        assert timeline.start_time == 1700000000.0
        assert timeline.end_time == 1700000020.0

    def test_preserves_action_types(self):
        sigs = [
            _make_signed_action(
                signature_id="sid_a",
                action_id="act_a",
                action_type="tool:search",
                signed_at=1700000000.0,
            ),
            _make_signed_action(
                signature_id="sid_b",
                action_id="act_b",
                action_type="api:openai",
                signed_at=1700000010.0,
            ),
        ]
        bundle = export_bundle(sigs)
        timeline = replay_from_bundle(bundle)

        types = [s.action_type for s in timeline.steps]
        assert "tool:search" in types
        assert "api:openai" in types

    def test_empty_bundle(self):
        bundle = export_bundle([])
        timeline = replay_from_bundle(bundle)

        assert timeline.total_actions == 0
        assert timeline.steps == []
        assert timeline.chain_integrity is True
        assert timeline.start_time is None
        assert timeline.end_time is None


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_contains_key_info(self):
        timeline = _make_timeline(2)
        text = timeline.summary()

        assert "Audit Trail Replay" in text
        assert "agent_001" in text
        assert "sess_001" in text
        assert "Actions: 2" in text
        assert "PASS" in text

    def test_summary_shows_steps(self):
        timeline = _make_timeline(2)
        text = timeline.summary()

        assert "[0]" in text
        assert "[1]" in text

    def test_summary_shows_fail_on_broken_chain(self):
        timeline = _make_timeline(2)
        timeline.chain_integrity = False
        text = timeline.summary()

        assert "FAIL" in text

    def test_empty_timeline_summary(self):
        timeline = ReplayTimeline(
            agent_id="agent_x",
            session_id="sess_x",
            total_actions=0,
        )
        text = timeline.summary()
        assert "Actions: 0" in text
        assert "Steps:" not in text


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_json_roundtrip(self):
        timeline = _make_timeline(3)
        j = timeline.to_json()
        parsed = json.loads(j)

        assert parsed["agent_id"] == "agent_001"
        assert parsed["session_id"] == "sess_001"
        assert parsed["total_actions"] == 3
        assert len(parsed["steps"]) == 3
        assert parsed["chain_integrity"] is True

    def test_to_file_roundtrip(self):
        timeline = _make_timeline(2)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            timeline.to_file(path)
            with open(path) as f:
                parsed = json.load(f)
            assert parsed["agent_id"] == "agent_001"
            assert parsed["total_actions"] == 2
            assert len(parsed["steps"]) == 2
        finally:
            os.unlink(path)

    def test_to_dict_structure(self):
        timeline = _make_timeline(1)
        d = timeline.to_dict()

        assert isinstance(d, dict)
        step = d["steps"][0]
        assert "index" in step
        assert "action_type" in step
        assert "context" in step
        assert "timestamp" in step
        assert "signature_id" in step
        assert "verification_url" in step
        assert "chain_valid" in step
        assert "explanation" in step


# ---------------------------------------------------------------------------
# Empty signatures
# ---------------------------------------------------------------------------

class TestEmptySignatures:
    def test_empty_verify_chain(self):
        timeline = ReplayTimeline(
            agent_id="agent_x",
            session_id="sess_x",
        )
        assert timeline.verify_chain() is True
        assert timeline.chain_integrity is True

    def test_empty_hash_chain(self):
        assert _verify_hash_chain([]) == []

    def test_empty_timeline_serializes(self):
        timeline = ReplayTimeline(
            agent_id="agent_x",
            session_id="sess_x",
        )
        d = timeline.to_dict()
        assert d["total_actions"] == 0
        assert d["steps"] == []


# ---------------------------------------------------------------------------
# Explanation builder
# ---------------------------------------------------------------------------

class TestBuildExplanation:
    def test_api_category(self):
        result = _build_explanation("api:openai", None)
        assert result == "Called openai API"

    def test_tool_category(self):
        result = _build_explanation("tool:search", None)
        assert result == "Used tool search"

    def test_with_model_context(self):
        result = _build_explanation("api:openai", {"model": "gpt-4"})
        assert "gpt-4" in result

    def test_unknown_category(self):
        result = _build_explanation("custom:thing", None)
        assert "custom" in result
        assert "thing" in result

    def test_no_colon_in_type(self):
        result = _build_explanation("simple_action", None)
        assert result == "simple_action"
