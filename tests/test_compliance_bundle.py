"""Tests for compliance bundle export with Merkle root verification."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import SignatureResponse, SignedActionResponse
from asqav.compliance import (
    FRAMEWORKS,
    ComplianceBundle,
    _compute_merkle_root,
    _receipt_hash,
    export_bundle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sig_response(**overrides) -> SignatureResponse:
    defaults = {
        "signature": "sig_abc123",
        "signature_id": "sid_001",
        "action_id": "act_001",
        "timestamp": 1700000000.0,
        "verification_url": "https://asqav.com/verify/sid_001",
    }
    defaults.update(overrides)
    return SignatureResponse(**defaults)


def _make_signed_action(**overrides) -> SignedActionResponse:
    defaults = {
        "signature_id": "sid_002",
        "agent_id": "agent_001",
        "action_id": "act_002",
        "action_type": "api:call",
        "payload": {"model": "gpt-4"},
        "algorithm": "ML-DSA-65",
        "signed_at": 1700000000.0,
        "signature_preview": "sig_preview_xyz",
        "verification_url": "https://asqav.com/verify/sid_002",
    }
    defaults.update(overrides)
    return SignedActionResponse(**defaults)


# ---------------------------------------------------------------------------
# Merkle root tests
# ---------------------------------------------------------------------------

class TestMerkleRoot:
    def test_single_hash(self):
        h = hashlib.sha256(b"hello").hexdigest()
        assert _compute_merkle_root([h]) == h

    def test_two_hashes(self):
        h1 = hashlib.sha256(b"a").hexdigest()
        h2 = hashlib.sha256(b"b").hexdigest()
        expected = hashlib.sha256((h1 + h2).encode()).hexdigest()
        assert _compute_merkle_root([h1, h2]) == expected

    def test_three_hashes_duplicates_last(self):
        h1 = hashlib.sha256(b"a").hexdigest()
        h2 = hashlib.sha256(b"b").hexdigest()
        h3 = hashlib.sha256(b"c").hexdigest()
        # Level 1: pair (h1,h2) and (h3,h3)
        p1 = hashlib.sha256((h1 + h2).encode()).hexdigest()
        p2 = hashlib.sha256((h3 + h3).encode()).hexdigest()
        expected = hashlib.sha256((p1 + p2).encode()).hexdigest()
        assert _compute_merkle_root([h1, h2, h3]) == expected

    def test_four_hashes(self):
        hashes = [hashlib.sha256(x.encode()).hexdigest() for x in "abcd"]
        p1 = hashlib.sha256((hashes[0] + hashes[1]).encode()).hexdigest()
        p2 = hashlib.sha256((hashes[2] + hashes[3]).encode()).hexdigest()
        expected = hashlib.sha256((p1 + p2).encode()).hexdigest()
        assert _compute_merkle_root(hashes) == expected

    def test_empty_list(self):
        expected = hashlib.sha256(b"").hexdigest()
        assert _compute_merkle_root([]) == expected

    def test_deterministic(self):
        hashes = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(7)]
        r1 = _compute_merkle_root(hashes)
        r2 = _compute_merkle_root(hashes)
        assert r1 == r2


# ---------------------------------------------------------------------------
# export_bundle tests
# ---------------------------------------------------------------------------

class TestExportBundle:
    def test_with_signature_responses(self):
        sigs = [_make_sig_response(signature_id=f"sid_{i}") for i in range(3)]
        bundle = export_bundle(sigs, framework="eu_ai_act_art12")

        assert bundle.framework == "eu_ai_act_art12"
        assert bundle.receipt_count == 3
        assert len(bundle.receipts) == 3
        assert bundle.merkle_root
        assert bundle.framework_metadata == FRAMEWORKS["eu_ai_act_art12"]

    def test_with_signed_action_responses(self):
        actions = [_make_signed_action(signature_id=f"sid_{i}") for i in range(2)]
        bundle = export_bundle(actions, framework="soc2")

        assert bundle.framework == "soc2"
        assert bundle.receipt_count == 2
        assert bundle.framework_metadata["name"] == "SOC 2 Type II"

    def test_with_dicts(self):
        dicts = [
            {"signature_id": "sid_d1", "action": "test"},
            {"signature_id": "sid_d2", "action": "test2"},
        ]
        bundle = export_bundle(dicts, framework="dora_ict")
        assert bundle.receipt_count == 2
        assert bundle.receipts[0]["signature_id"] == "sid_d1"

    def test_mixed_types(self):
        sigs = [
            _make_sig_response(),
            _make_signed_action(),
            {"signature_id": "sid_dict", "custom": True},
        ]
        bundle = export_bundle(sigs)
        assert bundle.receipt_count == 3

    def test_invalid_framework_raises(self):
        try:
            export_bundle([], framework="nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "nonexistent" in str(e)

    def test_empty_signatures(self):
        bundle = export_bundle([], framework="eu_ai_act_art14")
        assert bundle.receipt_count == 0
        assert len(bundle.receipts) == 0
        assert bundle.merkle_root == hashlib.sha256(b"").hexdigest()

    def test_verification_section(self):
        sigs = [_make_sig_response()]
        bundle = export_bundle(sigs)

        assert "merkle_root" in bundle.verification
        assert "hash_algorithm" in bundle.verification
        assert bundle.verification["hash_algorithm"] == "sha256"
        assert len(bundle.verification["receipt_hashes"]) == 1
        assert bundle.verification["merkle_root"] == bundle.merkle_root

    def test_merkle_root_matches_receipts(self):
        """Verify an auditor can recompute the Merkle root from receipts."""
        sigs = [_make_sig_response(signature_id=f"sid_{i}") for i in range(5)]
        bundle = export_bundle(sigs)

        recomputed_hashes = [_receipt_hash(r) for r in bundle.receipts]
        recomputed_root = _compute_merkle_root(recomputed_hashes)
        assert recomputed_root == bundle.merkle_root

    def test_default_framework(self):
        bundle = export_bundle([_make_sig_response()])
        assert bundle.framework == "eu_ai_act_art12"

    def test_all_frameworks_valid(self):
        for fw_key in FRAMEWORKS:
            bundle = export_bundle([_make_sig_response()], framework=fw_key)
            assert bundle.framework == fw_key
            assert bundle.framework_metadata == FRAMEWORKS[fw_key]


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict(self):
        bundle = export_bundle([_make_sig_response()])
        d = bundle.to_dict()
        assert isinstance(d, dict)
        assert d["framework"] == "eu_ai_act_art12"
        assert d["receipt_count"] == 1
        assert "merkle_root" in d

    def test_to_json(self):
        bundle = export_bundle([_make_sig_response()])
        j = bundle.to_json()
        parsed = json.loads(j)
        assert parsed["framework"] == "eu_ai_act_art12"
        assert parsed["receipt_count"] == 1

    def test_to_file(self):
        bundle = export_bundle([_make_sig_response(), _make_signed_action()])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            bundle.to_file(path)
            with open(path) as f:
                parsed = json.load(f)
            assert parsed["framework"] == "eu_ai_act_art12"
            assert parsed["receipt_count"] == 2
            assert parsed["merkle_root"] == bundle.merkle_root
        finally:
            os.unlink(path)

    def test_to_json_roundtrip(self):
        bundle = export_bundle(
            [_make_sig_response(), _make_signed_action()],
            framework="dora_ict",
        )
        j = bundle.to_json()
        parsed = json.loads(j)
        assert parsed["verification"]["merkle_root"] == bundle.merkle_root
        assert len(parsed["verification"]["receipt_hashes"]) == 2
