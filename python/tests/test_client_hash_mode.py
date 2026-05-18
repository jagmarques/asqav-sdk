"""Verify Agent.sign() request body shape across modes.

We patch the module-level ``_post`` helper in ``asqav.client`` and assert
on the body it would have sent. Avoiding ``httpx.MockTransport`` keeps
this test independent of the optional httpx dep and matches the existing
``patch("asqav.client._post")`` pattern used elsewhere in the test suite.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav import client as client_mod
from asqav.client import Agent

# A canned response that satisfies SignatureResponse decoding.
MOCK_SIGN_RESPONSE: dict = {
    "signature": "sig-bytes",
    "signature_id": "sig_001",
    "action_id": "act_001",
    "timestamp": "2026-04-28T12:00:00Z",
    "verification_url": "https://verify.asqav.com/sig_001",
    "algorithm": "ml-dsa-65",
    "chain_hash": "ch_abc",
    "rfc3161_timestamp": None,
    "scan_result": None,
    "policy_digest": "pd",
    "policy_decision": "permit",
    "authorization_ref": None,
    "bitcoin_anchor": None,
}


def _make_agent() -> Agent:
    return Agent(
        agent_id="agt_001",
        name="t",
        public_key="pk",
        key_id="kid",
        algorithm="ml-dsa-65",
        capabilities=[],
        created_at=0.0,
    )


@pytest.fixture(autouse=True)
def _isolate_module_state() -> None:
    """Reset SDK module state between tests."""
    saved = (client_mod._mode, client_mod._org_salt, client_mod._api_key)
    client_mod._api_key = "sk_test"
    yield
    client_mod._mode, client_mod._org_salt, client_mod._api_key = saved


def test_full_payload_mode_sends_context() -> None:
    client_mod._mode = "full-payload"
    client_mod._org_salt = None
    agent = _make_agent()

    with patch("asqav.client._post", return_value=MOCK_SIGN_RESPONSE) as mock_post:
        agent.sign("api:call", {"prompt": "hi", "user_id": "u_001"})

    path, body = mock_post.call_args[0]
    assert path == "/agents/agt_001/sign"
    assert "context" in body
    assert "hash" not in body
    assert body["context"] == {"prompt": "hi", "user_id": "u_001"}
    assert body["action_type"] == "api:call"


def test_hash_only_mode_sends_hash_not_context() -> None:
    client_mod._mode = "hash-only"
    client_mod._org_salt = None
    agent = _make_agent()

    with patch("asqav.client._post", return_value=MOCK_SIGN_RESPONSE) as mock_post:
        agent.sign("api:call", {"prompt": "hi", "user_id": "u_001"})

    _path, body = mock_post.call_args[0]
    assert "hash" in body, "hash-only mode must include hash"
    assert "context" not in body, "hash-only mode must NOT leak context"
    assert body["hash"].startswith("sha256:")
    assert len(body["hash"]) == len("sha256:") + 64
    assert body["hash_algo"] == "sha256"
    assert body["action_type"] == "api:call"
    # Wire payload_digest carries {hash, size}; the SDK forwards size
    # alongside the hash so the cloud can build the upstream object form.
    assert "payload_size" in body
    assert isinstance(body["payload_size"], int) and body["payload_size"] > 0


def test_hash_only_payload_size_matches_canonical_bytes_length() -> None:
    """payload_size MUST equal the byte length of the canonical fingerprint."""
    from asqav.canonicalize import canonicalize

    client_mod._mode = "hash-only"
    client_mod._org_salt = None
    agent = _make_agent()

    context = {"prompt": "hi", "user_id": "u_001"}
    expected_bytes = canonicalize({"action_type": "api:call", "context": context})

    with patch("asqav.client._post", return_value=MOCK_SIGN_RESPONSE) as mock_post:
        agent.sign("api:call", context)

    _, body = mock_post.call_args[0]
    assert body["payload_size"] == len(expected_bytes)


def test_hash_only_metadata_does_not_leak_user_fields() -> None:
    """user_id, ip, prompt, etc. in context must NOT appear in metadata."""
    client_mod._mode = "hash-only"
    client_mod._org_salt = None
    agent = _make_agent()

    sensitive_context = {
        "prompt": "secret",
        "user_id": "u_001",
        "ip": "1.2.3.4",
        "request_id": "req_xyz",
    }
    with patch("asqav.client._post", return_value=MOCK_SIGN_RESPONSE) as mock_post:
        agent.sign("api:call", sensitive_context)

    _, body = mock_post.call_args[0]
    metadata = body["metadata"]
    for leak in ("user_id", "ip", "prompt", "request_id"):
        assert leak not in metadata, f"sensitive key {leak!r} leaked into metadata"
    assert metadata["agent_id"] == "agt_001"
    assert metadata["action_type"] == "api:call"


def test_hash_only_includes_optional_model_name_when_opted_in() -> None:
    """If caller adds ``_model_name`` sentinel to context, surface in metadata."""
    client_mod._mode = "hash-only"
    client_mod._org_salt = None
    agent = _make_agent()

    with patch("asqav.client._post", return_value=MOCK_SIGN_RESPONSE) as mock_post:
        agent.sign(
            "api:call",
            {"prompt": "hi", "_model_name": "gpt-4", "_tool_name": "search"},
        )

    _, body = mock_post.call_args[0]
    assert body["metadata"]["model_name"] == "gpt-4"
    assert body["metadata"]["tool_name"] == "search"


def test_hash_only_surfaces_parent_id_from_context_sentinel() -> None:
    """``_parent_id`` in context auto-surfaces to metadata bag."""
    client_mod._mode = "hash-only"
    client_mod._org_salt = None
    agent = _make_agent()

    with patch("asqav.client._post", return_value=MOCK_SIGN_RESPONSE) as mock_post:
        agent.sign("api:call", {"prompt": "hi", "_parent_id": "act_parent_42"})

    _, body = mock_post.call_args[0]
    assert body["metadata"]["parent_id"] == "act_parent_42"


def test_hash_only_explicit_tool_name_kwarg_appears_in_metadata() -> None:
    """Passing tool_name=... to sign() surfaces it in the wire metadata bag."""
    client_mod._mode = "hash-only"
    client_mod._org_salt = None
    agent = _make_agent()

    with patch("asqav.client._post", return_value=MOCK_SIGN_RESPONSE) as mock_post:
        agent.sign("api:call", {"prompt": "hi"}, tool_name="web_search")

    _, body = mock_post.call_args[0]
    assert body["metadata"]["tool_name"] == "web_search"


def test_hash_only_explicit_model_name_kwarg_appears_in_metadata() -> None:
    """Passing model_name=... to sign() surfaces it in the wire metadata bag."""
    client_mod._mode = "hash-only"
    client_mod._org_salt = None
    agent = _make_agent()

    with patch("asqav.client._post", return_value=MOCK_SIGN_RESPONSE) as mock_post:
        agent.sign("api:call", {"prompt": "hi"}, model_name="claude-opus-4-7")

    _, body = mock_post.call_args[0]
    assert body["metadata"]["model_name"] == "claude-opus-4-7"


def test_hash_only_explicit_parent_id_kwarg_appears_in_metadata() -> None:
    """Passing parent_id=... to sign() surfaces it in the wire metadata bag."""
    client_mod._mode = "hash-only"
    client_mod._org_salt = None
    agent = _make_agent()

    with patch("asqav.client._post", return_value=MOCK_SIGN_RESPONSE) as mock_post:
        agent.sign("api:call", {"prompt": "hi"}, parent_id="act_parent_99")

    _, body = mock_post.call_args[0]
    assert body["metadata"]["parent_id"] == "act_parent_99"


def test_full_payload_keeps_telemetry_in_context_only_not_duplicated() -> None:
    """In full-payload mode the three fields ride inside context (as
    ``_tool_name`` / ``_model_name`` / ``_parent_id``) and the body must
    not also carry a top-level metadata bag duplicating them."""
    client_mod._mode = "full-payload"
    client_mod._org_salt = None
    agent = _make_agent()

    with patch("asqav.client._post", return_value=MOCK_SIGN_RESPONSE) as mock_post:
        agent.sign(
            "api:call",
            {"prompt": "hi"},
            tool_name="web_search",
            model_name="gpt-4",
            parent_id="act_parent_1",
        )

    _, body = mock_post.call_args[0]
    assert "metadata" not in body
    assert body["context"]["_tool_name"] == "web_search"
    assert body["context"]["_model_name"] == "gpt-4"
    assert body["context"]["_parent_id"] == "act_parent_1"


def test_hash_only_uses_hmac_when_org_salt_set() -> None:
    """A non-None org_salt swaps SHA-256 for HMAC-SHA-256 with the same shape."""
    agent = _make_agent()
    plain_hash = None
    salted_hash = None

    client_mod._mode = "hash-only"
    client_mod._org_salt = None
    with patch("asqav.client._post", return_value=MOCK_SIGN_RESPONSE) as mock_post:
        agent.sign("api:call", {"k": 3})
        plain_hash = mock_post.call_args[0][1]["hash"]

    client_mod._org_salt = b"\x01" * 32
    with patch("asqav.client._post", return_value=MOCK_SIGN_RESPONSE) as mock_post:
        agent.sign("api:call", {"k": 3})
        salted_hash = mock_post.call_args[0][1]["hash"]

    assert plain_hash != salted_hash
    assert plain_hash.startswith("sha256:") and salted_hash.startswith("sha256:")


def test_init_resolves_mode_from_url() -> None:
    """asqav.init() should set _mode from the api_base_url when mode=auto."""
    saved_base = client_mod._api_base
    try:
        # Cloud URL -> hash-only
        asqav.init(api_key="sk_test", base_url="https://api.asqav.com/api/v1")
        assert client_mod._mode == "hash-only"

        # Self-hosted URL -> full-payload
        asqav.init(api_key="sk_test", base_url="http://localhost:8000")
        assert client_mod._mode == "full-payload"

        # Explicit override
        asqav.init(
            api_key="sk_test",
            base_url="https://api.asqav.com/api/v1",
            mode="full-payload",
        )
        assert client_mod._mode == "full-payload"
    finally:
        client_mod._api_base = saved_base
