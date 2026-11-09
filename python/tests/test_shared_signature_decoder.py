"""countersign() and sign_batch() decode through the one sign-response decoder."""

from __future__ import annotations

from unittest.mock import patch

import asqav
from asqav.client import _agent_signing, _transport
from asqav.client._models import SignatureResponse

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


def test_both_paths_apply_the_decoder_fallbacks() -> None:
    row = {**_BASE, "record_hash": "h1", "policy_decision": "deny", "compliance_mode": True}
    with patch.object(_transport, "_post", side_effect=[row, {"signatures": [row]}]):
        single = _agent().countersign("sig_1")
        batch = _agent().sign_batch([{"action_type": "x:y"}])[0]
    for res in (single, batch):
        assert res.chain_hash == "h1"
        assert res.decision == "deny"


def test_negctl_without_the_decoder_the_fallback_is_lost() -> None:
    row = {**_BASE, "record_hash": "h1"}
    raw = lambda data: SignatureResponse(  # noqa: E731
        **{k: data[k] for k in _BASE}, chain_hash=data.get("chain_hash")
    )
    with (
        patch.object(_transport, "_post", return_value=row),
        patch.object(_agent_signing, "_signature_response", raw),
    ):
        assert _agent().countersign("sig_1").chain_hash is None
