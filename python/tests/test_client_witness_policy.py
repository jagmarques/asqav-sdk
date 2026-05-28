"""SDK parity for the ``witness_policy`` wire field.

The cloud accepts an optional ``witness_policy`` object on the signed
Compliance Receipt with shape
``{"required": int, "witnesses": [<subset of "rfc3161",
"opentimestamps">]}``. The SDK forwards it verbatim and validates the
shape client-side: ``required`` in ``[1, len(witnesses)]``, ``witnesses``
a non-empty subset of the two shipped witnesses, and ``rekor`` rejected.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import Agent


def _agent() -> Agent:
    return Agent(
        agent_id="agent_witness",
        name="witness-agent",
        public_key="pk_xxx",
        key_id="key_witness",
        algorithm="ML-DSA-65",
        capabilities=["sign:*"],
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


def test_witness_policy_forwarded_to_wire() -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    policy = {"required": 1, "witnesses": ["rfc3161", "opentimestamps"]}
    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "deploy:release",
            {"k": "v"},
            compliance_mode=True,
            witness_policy=policy,
        )
    assert captured["body"]["witness_policy"] == policy


def test_witness_policy_single_witness_forwarded() -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    policy = {"required": 1, "witnesses": ["opentimestamps"]}
    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "deploy:release",
            {"k": "v"},
            compliance_mode=True,
            witness_policy=policy,
        )
    assert captured["body"]["witness_policy"] == policy


def test_witness_policy_rekor_rejected() -> None:
    with pytest.raises(ValueError, match="witness_policy_unknown_witness"):
        _agent().sign(
            "deploy:release",
            {"k": "v"},
            compliance_mode=True,
            witness_policy={"required": 1, "witnesses": ["rekor"]},
        )


def test_witness_policy_rekor_in_mixed_list_rejected() -> None:
    with pytest.raises(ValueError, match="witness_policy_unknown_witness"):
        _agent().sign(
            "deploy:release",
            {"k": "v"},
            compliance_mode=True,
            witness_policy={"required": 1, "witnesses": ["rfc3161", "rekor"]},
        )


def test_witness_policy_required_exceeds_len_rejected() -> None:
    with pytest.raises(ValueError, match="witness_policy_required_out_of_range"):
        _agent().sign(
            "deploy:release",
            {"k": "v"},
            compliance_mode=True,
            witness_policy={"required": 3, "witnesses": ["rfc3161", "opentimestamps"]},
        )


def test_witness_policy_required_zero_rejected() -> None:
    with pytest.raises(ValueError, match="witness_policy_required_out_of_range"):
        _agent().sign(
            "deploy:release",
            {"k": "v"},
            compliance_mode=True,
            witness_policy={"required": 0, "witnesses": ["rfc3161"]},
        )


def test_witness_policy_empty_witnesses_rejected() -> None:
    with pytest.raises(
        ValueError, match="witness_policy_witnesses_must_be_non_empty_list"
    ):
        _agent().sign(
            "deploy:release",
            {"k": "v"},
            compliance_mode=True,
            witness_policy={"required": 1, "witnesses": []},
        )


def test_witness_policy_duplicate_witness_rejected() -> None:
    with pytest.raises(ValueError, match="witness_policy_duplicate_witness"):
        _agent().sign(
            "deploy:release",
            {"k": "v"},
            compliance_mode=True,
            witness_policy={"required": 1, "witnesses": ["rfc3161", "rfc3161"]},
        )


def test_witness_policy_required_not_int_rejected() -> None:
    with pytest.raises(ValueError, match="witness_policy_required_must_be_int"):
        _agent().sign(
            "deploy:release",
            {"k": "v"},
            compliance_mode=True,
            witness_policy={"required": "1", "witnesses": ["rfc3161"]},
        )


def test_witness_policy_not_dict_rejected() -> None:
    with pytest.raises(ValueError, match="witness_policy_invalid"):
        _agent().sign(
            "deploy:release",
            {"k": "v"},
            compliance_mode=True,
            witness_policy=["rfc3161"],  # type: ignore[arg-type]
        )


def test_witness_policy_absent_means_no_field() -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign("deploy:release", {"k": "v"}, compliance_mode=True)
    assert "witness_policy" not in captured["body"]
