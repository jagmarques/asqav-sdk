"""countersign and sign_batch decode through the one sign decoder."""

from __future__ import annotations

from unittest.mock import patch

import asqav
from asqav.client import _agent_signing, _transport

_BASE = {
    "signature": "s",
    "signature_id": "sig_1",
    "action_id": "act_1",
    "timestamp": "2026-04-28T10:00:00Z",
    "verification_url": "https://api.asqav.com/api/v1/verify/sig_1",
}


def _agent() -> asqav.Agent:
    return asqav.Agent(
        agent_id="a", name="a", public_key="p", key_id="k", algorithm="ML-DSA-65",
        capabilities=[], created_at=0.0,
    )


def test_both_paths_fill_the_decoder_only_fields() -> None:
    row = {
        **_BASE,
        "receipt_type": "action",
        "payload_digest": "d1",
        "compliance_mode": True,
    }
    with patch.object(_transport, "_post", side_effect=[row, {"signatures": [row]}]):
        single = _agent().countersign("sig_1")
        batch = _agent().sign_batch([{"action_type": "x:y"}])[0]
    for res in (single, batch):
        assert res.receipt_type == "action"
        assert res.payload_digest == "d1"
        assert res.compliance_mode is True


def test_negctl_a_planted_decoder_is_the_one_both_paths_return() -> None:
    """NEGCTL: a sentinel decoder planted in the module is what both return."""
    row = {**_BASE}
    with (
        patch.object(_transport, "_post", side_effect=[row, {"signatures": [row]}]),
        patch.object(_agent_signing, "_signature_response", lambda data: "planted"),
    ):
        assert _agent().countersign("sig_1") == "planted"
        assert _agent().sign_batch([{"action_type": "x:y"}]) == ["planted"]
