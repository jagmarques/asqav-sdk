"""Verify conformance vectors are internally consistent.

Any third-party implementation should be able to reproduce the canonical bytes
and SHA-256 for each input. If this test fails, vectors.json was edited without
regenerating the derived fields.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

VECTORS_PATH = Path(__file__).parent.parent / "conformance" / "vectors.json"


def _jcs(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def test_vectors_file_exists() -> None:
    assert VECTORS_PATH.exists(), f"missing {VECTORS_PATH}"


def test_vectors_schema_metadata() -> None:
    d = json.loads(VECTORS_PATH.read_text())
    assert d["version"] >= 1
    assert d["canonicalization"] == "RFC 8785 JCS"
    assert d["hash_algorithm"] == "SHA-256"
    assert isinstance(d["vectors"], list)
    assert len(d["vectors"]) >= 3


def test_each_vector_canonical_matches_input() -> None:
    d = json.loads(VECTORS_PATH.read_text())
    for v in d["vectors"]:
        expected_canon = _jcs(v["input"])
        assert v["canonical"] == expected_canon, (
            f"{v['name']}: canonical drift. Regenerate vectors.json."
        )


def test_each_vector_sha256_matches_canonical() -> None:
    d = json.loads(VECTORS_PATH.read_text())
    for v in d["vectors"]:
        expected_hash = hashlib.sha256(v["canonical"].encode("utf-8")).hexdigest()
        assert v["sha256"] == expected_hash, (
            f"{v['name']}: sha256 drift. Regenerate vectors.json."
        )


def test_each_vector_declares_expected_verify() -> None:
    """Every vector must declare whether verification should succeed or fail."""
    d = json.loads(VECTORS_PATH.read_text())
    for v in d["vectors"]:
        assert "expected_verify" in v, f"{v['name']} missing expected_verify"
        assert isinstance(v["expected_verify"], bool)
        assert "reason" in v and v["reason"], f"{v['name']} missing reason"


def test_coverage_includes_adversarial_cases() -> None:
    """vectors.json must cover at least 5 distinct adversarial failure modes."""
    d = json.loads(VECTORS_PATH.read_text())
    fail_names = {v["name"] for v in d["vectors"] if v.get("expected_verify") is False}
    required = {
        "tampered_signature",
        "swapped_public_key",
        "stale_card",
        "nonce_mismatch",
        "card_version_downgrade",
    }
    missing = required - fail_names
    assert not missing, f"missing adversarial vectors: {missing}"


def test_nonce_mismatch_vector_has_peer_sent_nonce() -> None:
    """The nonce-mismatch vector must declare what the peer sent vs what was echoed."""
    d = json.loads(VECTORS_PATH.read_text())
    v = next(x for x in d["vectors"] if x["name"] == "nonce_mismatch")
    assert "peer_sent_nonce" in v
    assert v["input"].get("client_nonce") != v["peer_sent_nonce"]
