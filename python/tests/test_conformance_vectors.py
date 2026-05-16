"""Verify conformance vectors are internally consistent.

Any third-party implementation should be able to reproduce the canonical bytes
and SHA-256 for each input. If this test fails, vectors.json was edited without
regenerating the derived fields.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

VECTORS_PATH = Path(__file__).parent.parent.parent / "conformance" / "vectors.json"

# Closed Literal mirrored from the cloud SignRequest and IETF draft -04 appendix.
CAPTURE_TOPOLOGY_VOCABULARY: frozenset[str] = frozenset(
    {
        "in_process_sdk",
        "network_proxy",
        "browser_extension",
        "ebpf_observer",
        "mcp_proxy",
    }
)


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


# === capture_topology vocabulary parity (IETF -04 appendix + cloud SignRequest) ===


def _capture_vectors() -> list[dict]:
    """Return all conformance vectors that exercise the capture_topology field."""
    d = json.loads(VECTORS_PATH.read_text())
    return [v for v in d["vectors"] if v["name"].startswith("capture_topology_")]


def test_capture_topology_covers_full_closed_vocabulary() -> None:
    """All five IETF -04 capture topologies must appear as accepted vectors."""
    accepted = {
        v["capture_topology"]
        for v in _capture_vectors()
        if v.get("expected_verify") is True
    }
    assert accepted == CAPTURE_TOPOLOGY_VOCABULARY, (
        f"capture_topology vector coverage drift: {accepted ^ CAPTURE_TOPOLOGY_VOCABULARY}"
    )


@pytest.mark.parametrize("value", sorted(CAPTURE_TOPOLOGY_VOCABULARY))
def test_capture_topology_value_round_trips_via_manifest(value: str) -> None:
    """Each accepted vector stamps capture_topology on the manifest entry.

    The token must never appear inside the signed payload bytes.
    """
    matching = [
        v
        for v in _capture_vectors()
        if v.get("capture_topology") == value and v.get("expected_verify") is True
    ]
    assert len(matching) == 1, f"expected exactly one accepted vector for {value}"
    v = matching[0]
    assert "manifest" in v["input"], f"{v['name']}: capture_topology must live in `manifest`"
    assert v["input"]["manifest"]["capture_topology"] == value
    assert "capture_topology" not in v["input"].get("context", {}), (
        f"{v['name']}: capture_topology MUST NOT appear inside the signed payload"
    )


def test_capture_topology_unknown_value_is_a_failure_vector() -> None:
    """An out-of-vocabulary capture_topology token is a rejected conformance vector."""
    rejected = [
        v
        for v in _capture_vectors()
        if v.get("expected_verify") is False
    ]
    assert rejected, "missing the unknown-value rejection vector"
    v = rejected[0]
    assert v["capture_topology"] not in CAPTURE_TOPOLOGY_VOCABULARY
