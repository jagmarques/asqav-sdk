"""Tests for asqav.sign_reasoning helper."""

from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock

import pytest

from asqav.client import SignatureResponse
from asqav.reasoning import ReasoningReceipt, _sha256, sign_reasoning


def _mock_agent() -> MagicMock:
    agent = MagicMock()
    agent.sign.return_value = SignatureResponse(
        signature="sig-1",
        signature_id="sid-1",
        action_id="aid-1",
        timestamp=1700000000.0,
        verification_url="https://asqav.com/verify/sid-1",
    )
    return agent


def test_sign_reasoning_stores_hashes_only():
    """Default mode posts only hashes, never raw prompt/trace/output."""
    agent = _mock_agent()

    prompt = "classify this ticket as urgent"
    trace = [
        {"role": "assistant", "content": "Let me think..."},
        {"role": "tool", "name": "search", "args": {"q": "urgent"}},
    ]
    output = {"label": "urgent", "confidence": 0.93}

    receipt = sign_reasoning(agent, prompt, trace, output)

    assert isinstance(receipt, ReasoningReceipt)
    assert receipt.prompt_hash == _sha256(prompt)
    assert receipt.trace_hash == _sha256(trace)
    assert receipt.output_hash == _sha256(output)
    assert receipt.raw_stored is False

    agent.sign.assert_called_once()
    kwargs = agent.sign.call_args.kwargs
    ctx = kwargs["context"]

    # Hashes are present
    assert ctx["_prompt_hash"] == receipt.prompt_hash
    assert ctx["_reasoning_trace_hash"] == receipt.trace_hash
    assert ctx["_output_hash"] == receipt.output_hash

    # Raw payloads are NOT present by default
    assert "_raw_prompt" not in ctx
    assert "_raw_reasoning_trace" not in ctx
    assert "_raw_output" not in ctx

    # No raw text anywhere in the serialized context, digest only
    serialized = json.dumps(ctx, sort_keys=True, default=str)
    assert "urgent" not in serialized
    assert "Let me think" not in serialized


def test_sign_reasoning_raw_optional():
    """store_raw=True adds raw payloads under _raw_* keys."""
    agent = _mock_agent()

    receipt = sign_reasoning(
        agent,
        prompt="hello",
        trace=["step-1", "step-2"],
        output={"final": "done"},
        store_raw=True,
    )

    assert receipt.raw_stored is True

    ctx = agent.sign.call_args.kwargs["context"]
    assert ctx["_raw_prompt"] == "hello"
    assert ctx["_raw_reasoning_trace"] == ["step-1", "step-2"]
    assert ctx["_raw_output"] == {"final": "done"}

    # Hashes are still present alongside raw
    assert ctx["_prompt_hash"] == hashlib.sha256(b"hello").hexdigest()


def test_sign_reasoning_retention_param():
    """retention_days flows to context and ReasoningReceipt when raw is on."""
    agent = _mock_agent()

    receipt = sign_reasoning(
        agent,
        prompt="p",
        trace="t",
        output="o",
        store_raw=True,
        retention_days=30,
    )

    assert receipt.retention_days == 30
    ctx = agent.sign.call_args.kwargs["context"]
    assert ctx["_raw_retention_days"] == 30


def test_sign_reasoning_retention_ignored_without_store_raw():
    """retention_days without store_raw does not leak raw payloads."""
    agent = _mock_agent()

    sign_reasoning(
        agent,
        prompt="p",
        trace="t",
        output="o",
        retention_days=7,
    )

    ctx = agent.sign.call_args.kwargs["context"]
    assert "_raw_prompt" not in ctx
    assert "_raw_retention_days" not in ctx


def test_sha256_canonicalizes_dict_key_order():
    """Equivalent dicts in different key order produce equal hashes."""
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert _sha256(a) == _sha256(b)


def test_sign_reasoning_merges_user_context():
    """Caller-supplied context is preserved alongside injected hashes."""
    agent = _mock_agent()

    sign_reasoning(
        agent,
        prompt="p",
        trace="t",
        output="o",
        context={"customer": "acme", "request_id": "r-42"},
    )

    ctx = agent.sign.call_args.kwargs["context"]
    assert ctx["customer"] == "acme"
    assert ctx["request_id"] == "r-42"
    assert "_prompt_hash" in ctx


def test_sign_reasoning_passes_trace_and_parent_ids():
    """trace_id and parent_id reach the underlying sign() call."""
    agent = _mock_agent()

    sign_reasoning(
        agent,
        prompt="p",
        trace="t",
        output="o",
        trace_id="trace-123",
        parent_id="parent-sig-9",
    )

    kwargs = agent.sign.call_args.kwargs
    assert kwargs["trace_id"] == "trace-123"
    assert kwargs["parent_id"] == "parent-sig-9"
