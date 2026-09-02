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

# Closed Literal mirrored from the cloud SignRequest. passive_telemetry is observation-only;
# github_sha_pull is the server-stamped authoritative code-authorship capture layer.
CAPTURE_TOPOLOGY_VOCABULARY: frozenset[str] = frozenset(
    {
        "in_process_sdk",
        "network_proxy",
        "browser_extension",
        "mcp_proxy",
        "github_sha_pull",
    }
)


    # Independent of asqav.canonicalize on purpose: a third party must be able to
    # reproduce these bytes, so importing the SDK helper here would be circular.
def _jcs(obj: object) -> str:
    if isinstance(obj, dict):
        # RFC 8785 3.2.3 orders member names by UTF-16 code unit, not code point.
        items = sorted(obj.items(), key=lambda kv: str(kv[0]).encode("utf-16-be"))
        return "{" + ",".join(
            json.dumps(str(k), ensure_ascii=False) + ":" + _jcs(v) for k, v in items
        ) + "}"
    if isinstance(obj, list):
        return "[" + ",".join(_jcs(v) for v in obj) + "]"
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def test_vectors_file_exists() -> None:
    assert VECTORS_PATH.exists(), f"missing {VECTORS_PATH}"


def test_vectors_schema_metadata() -> None:
    d = json.loads(VECTORS_PATH.read_text())
    assert d["version"] >= 1
    assert d["canonicalization"] == "RFC 8785 JCS"
    assert d["hash_algorithm"] == "SHA-256"
    assert isinstance(d["vectors"], list)
    assert len(d["vectors"]) >= 3



    # A vector whose DOCUMENT is refused (asqav-25) carries `input_text` and no parsed
    # `input`, because a document that never parses has no canonical form to pin.
def _canonicalizing_vectors(d: dict) -> list[dict]:
    out = [v for v in d["vectors"] if "input" in v]
    assert out, "no canonicalizing vectors found"
    return out


    # Every vector carries exactly one of `input` or `input_text`, never both, never neither.
def test_every_vector_declares_its_input_form() -> None:
    d = json.loads(VECTORS_PATH.read_text())
    for v in d["vectors"]:
        has_parsed = "input" in v
        has_text = "input_text" in v
        assert has_parsed != has_text, (
            f"{v['name']}: needs exactly one of input / input_text"
        )
        if has_text:
            assert v["expected_verify"] is False, (
                f"{v['name']}: an input_text vector pins a refusal, so expected_verify is False"
            )
            assert "canonical" not in v and "sha256" not in v, (
                f"{v['name']}: a refused document has no canonical form"
            )

def test_each_vector_canonical_matches_input() -> None:
    d = json.loads(VECTORS_PATH.read_text())
    for v in _canonicalizing_vectors(d):
        expected_canon = _jcs(v["input"])
        assert v["canonical"] == expected_canon, (
            f"{v['name']}: canonical drift. Regenerate vectors.json."
        )


def test_each_vector_sha256_matches_canonical() -> None:
    d = json.loads(VECTORS_PATH.read_text())
    for v in _canonicalizing_vectors(d):
        expected_hash = hashlib.sha256(v["canonical"].encode("utf-8")).hexdigest()
        assert v["sha256"] == expected_hash, (
            f"{v['name']}: sha256 drift. Regenerate vectors.json."
        )


    # Every vector must declare whether verification should succeed or fail.
def test_each_vector_declares_expected_verify() -> None:
    d = json.loads(VECTORS_PATH.read_text())
    for v in d["vectors"]:
        assert "expected_verify" in v, f"{v['name']} missing expected_verify"
        assert isinstance(v["expected_verify"], bool)
        assert "reason" in v and v["reason"], f"{v['name']} missing reason"


    # vectors.json must cover at least 5 distinct adversarial failure modes.
def test_coverage_includes_adversarial_cases() -> None:
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


    # The nonce-mismatch vector must declare what the peer sent vs what was echoed.
def test_nonce_mismatch_vector_has_peer_sent_nonce() -> None:
    d = json.loads(VECTORS_PATH.read_text())
    v = next(x for x in d["vectors"] if x["name"] == "nonce_mismatch")
    assert "peer_sent_nonce" in v
    assert v["input"].get("client_nonce") != v["peer_sent_nonce"]


# === capture_topology vocabulary parity (IETF -04 appendix + cloud SignRequest) ===


    # Return all conformance vectors that exercise the capture_topology field.
def _capture_vectors() -> list[dict]:
    d = json.loads(VECTORS_PATH.read_text())
    return [v for v in d["vectors"] if v["name"].startswith("capture_topology_")]


    # All five IETF -04 capture topologies must appear as accepted vectors.
def test_capture_topology_covers_full_closed_vocabulary() -> None:
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


    # An out-of-vocabulary capture_topology token is a rejected conformance vector.
def test_capture_topology_unknown_value_is_a_failure_vector() -> None:
    rejected = [
        v
        for v in _capture_vectors()
        if v.get("expected_verify") is False
    ]
    assert rejected, "missing the unknown-value rejection vector"
    v = rejected[0]
    assert v["capture_topology"] not in CAPTURE_TOPOLOGY_VOCABULARY
