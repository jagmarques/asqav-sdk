"""SDK parity for the build-provenance 4-tuple wire fields.

The cloud accepts ``executable_hash``, ``sbom_digest``,
``slsa_provenance_pointer``, and ``supply_chain_pointer`` as optional
fields on the signed Compliance Receipt. The SDK forwards each verbatim,
validates the ``sha256:<64-hex>`` shape on the two digest fields, and
validates http(s) URL form on the two pointer fields.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import Agent

GOOD_EXECUTABLE_HASH = "sha256:" + "a" * 64
GOOD_SBOM_DIGEST = "sha256:" + "b" * 64
GOOD_SLSA_URL = "https://attestations.example.com/slsa/build-1.intoto.jsonl"
GOOD_SUPPLY_CHAIN_URL = "https://rekor.sigstore.dev/api/v1/log/entries/abc"


def _agent() -> Agent:
    return Agent(
        agent_id="agent_exec_hash",
        name="exec-hash-agent",
        public_key="pk_xxx",
        key_id="key_exec",
        algorithm="ML-DSA-65",
        capabilities=["build:*"],
        created_at=1700000000.0,
    )


def _ok_response() -> dict:
    return {
        "signature": "sig_b64",
        "signature_id": "sig_abc",
        "action_id": "act_abc",
        "timestamp": 1700000000.0,
        "verification_url": "https://asqav.com/verify/sig_abc",
        "algorithm": "ML-DSA-65",
        "chain_hash": "deadbeef",
    }


def test_executable_hash_forwarded_to_wire() -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "build:provenance",
            {"k": "v"},
            compliance_mode=True,
            executable_hash=GOOD_EXECUTABLE_HASH,
        )
    assert captured["body"]["executable_hash"] == GOOD_EXECUTABLE_HASH


def test_sbom_digest_forwarded_to_wire() -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "build:provenance",
            {"k": "v"},
            compliance_mode=True,
            sbom_digest=GOOD_SBOM_DIGEST,
        )
    assert captured["body"]["sbom_digest"] == GOOD_SBOM_DIGEST


def test_slsa_provenance_pointer_forwarded_to_wire() -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "build:provenance",
            {"k": "v"},
            compliance_mode=True,
            slsa_provenance_pointer=GOOD_SLSA_URL,
        )
    assert captured["body"]["slsa_provenance_pointer"] == GOOD_SLSA_URL


def test_supply_chain_pointer_forwarded_to_wire() -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "build:provenance",
            {"k": "v"},
            compliance_mode=True,
            supply_chain_pointer=GOOD_SUPPLY_CHAIN_URL,
        )
    assert captured["body"]["supply_chain_pointer"] == GOOD_SUPPLY_CHAIN_URL


def test_executable_hash_malformed_rejected_before_post() -> None:
    with pytest.raises(ValueError, match="digest_format_guard.*executable_hash"):
        _agent().sign(
            "build:provenance",
            {"k": "v"},
            compliance_mode=True,
            executable_hash="deadbeef",
        )


def test_sbom_digest_malformed_rejected_before_post() -> None:
    with pytest.raises(ValueError, match="digest_format_guard.*sbom_digest"):
        _agent().sign(
            "build:provenance",
            {"k": "v"},
            compliance_mode=True,
            sbom_digest="not-a-digest",
        )


def test_slsa_provenance_pointer_non_url_rejected() -> None:
    with pytest.raises(ValueError, match="pointer_url_guard.*slsa_provenance_pointer"):
        _agent().sign(
            "build:provenance",
            {"k": "v"},
            compliance_mode=True,
            slsa_provenance_pointer="not a url",
        )


def test_supply_chain_pointer_non_url_rejected() -> None:
    with pytest.raises(ValueError, match="pointer_url_guard.*supply_chain_pointer"):
        _agent().sign(
            "build:provenance",
            {"k": "v"},
            compliance_mode=True,
            supply_chain_pointer="ftp://nope",
        )


def test_all_four_present_canonical_decision() -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "build:provenance",
            {"k": "v"},
            compliance_mode=True,
            executable_hash=GOOD_EXECUTABLE_HASH,
            sbom_digest=GOOD_SBOM_DIGEST,
            slsa_provenance_pointer=GOOD_SLSA_URL,
            supply_chain_pointer=GOOD_SUPPLY_CHAIN_URL,
        )
    body = captured["body"]
    assert body["executable_hash"] == GOOD_EXECUTABLE_HASH
    assert body["sbom_digest"] == GOOD_SBOM_DIGEST
    assert body["slsa_provenance_pointer"] == GOOD_SLSA_URL
    assert body["supply_chain_pointer"] == GOOD_SUPPLY_CHAIN_URL
