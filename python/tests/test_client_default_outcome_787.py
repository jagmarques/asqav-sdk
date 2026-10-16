"""Default outcome: SDK sends the result digest unless told otherwise (787).

`sign(result=...)` digests and sends; an explicit `result_digest` wins;
`attach_result` mints the follow-up result receipt linked by action_ref.
Network is stubbed: the body is what the cloud would receive.
"""

from __future__ import annotations

import hashlib
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import Agent, _digest_result_value


def _agent() -> Agent:
    return Agent(
        agent_id="agent_outcome",
        name="outcome-agent",
        public_key="pk_xxx",
        key_id="key_outcome",
        algorithm="ML-DSA-65",
        capabilities=["api:*"],
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


def _capture(**kwargs):
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign("api:call", {"k": "v"}, **kwargs)
    return captured["body"]


def test_result_is_digested_and_sent_by_default() -> None:
    body = _capture(result="hello tool output")
    assert body["result_digest"] == "sha256:" + hashlib.sha256(b"hello tool output").hexdigest()


def test_explicit_digest_wins_over_result() -> None:
    body = _capture(result="hello", result_digest="sha256:" + "b" * 64)
    assert body["result_digest"] == "sha256:" + "b" * 64


def test_no_result_sends_no_digest() -> None:
    body = _capture()
    assert "result_digest" not in body


def test_digest_is_key_order_stable() -> None:
    assert _digest_result_value({"b": 1, "a": 2}) == _digest_result_value({"a": 2, "b": 1})
    assert _digest_result_value({"a": 1}).startswith("sha256:")
    assert len(_digest_result_value(b"raw")) == len("sha256:") + 64


def test_digest_falls_back_for_non_jcs_values() -> None:
    class HostObject:
        def __str__(self) -> str:
            return "host-value"

    circular: dict = {}
    circular["self"] = circular
    for awkward in (HostObject(), circular, float("nan"), 2**60, {"k": HostObject()}):
        digest = _digest_result_value(awkward)  # must not throw (fail-open)
        assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64
    assert _digest_result_value(HostObject()) == _digest_result_value("host-value")


def test_attach_result_links_origin() -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().attach_result("act_origin_123", {"status": "ok", "rows": 3})
    body = captured["body"]
    assert body["receipt_type"] == "protectmcp:observation:result_bound"
    assert body["action_ref"] == "act_origin_123"
    assert body["result_digest"] == _digest_result_value({"status": "ok", "rows": 3})


def _wire_sign(action: str, **kwargs):
    """Run one Agent-level call with stubbed network; return the wire body."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        action()
    return captured["body"]


def test_span_end_binds_attributes() -> None:
    from asqav.client import span

    bodies: list = []

    def fake_post(path: str, body: dict) -> dict:
        bodies.append(body)
        return _ok_response()

    with (
        patch("asqav.client.get_agent", return_value=_agent()),
        patch("asqav.client._post", side_effect=fake_post),
    ):
        with span("api:test", {"model": "x"}) as s:
            s.set_attribute("tokens", 7)
    assert bodies, "span end must sign"
    assert bodies[-1]["result_digest"] == _digest_result_value({"model": "x", "tokens": 7})


def test_sign_output_binds_output() -> None:
    body = _wire_sign(
        lambda: _agent().sign_output("tool:search", "in-hash", {"hits": [1, 2]}),
    )
    assert body["result_digest"] == _digest_result_value({"hits": [1, 2]})


def test_budget_record_binds_payload() -> None:
    from asqav.client import BudgetTracker

    body = _wire_sign(
        lambda: BudgetTracker(_agent(), limit=10.0).record("api:x", actual_cost=0.5),
    )
    assert body["action_type"] == "budget:api:x"
    assert body["result_digest"] == _digest_result_value(
        {"cost": 0.5, "currency": "USD", "cumulative_spend": 0.5, "limit": 10.0}
    )


def test_attestation_generate_binds_document() -> None:
    import asqav.client as client_mod

    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with (
        patch.object(client_mod.Agent, "get", return_value=_agent()),
        patch("asqav.client._post", side_effect=fake_post),
    ):
        client_mod.generate_attestation("agent_outcome")
    body = captured["body"]
    assert body["action_type"] == "attestation:generate"
    assert body["result_digest"].startswith("sha256:")
