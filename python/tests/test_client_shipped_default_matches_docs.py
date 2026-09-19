"""Pin the SDK's shipped hash-mode default to plain sha256 (criterion 222).

The docs describe an unsalted default; this test fails if the shipped
default ever becomes a keyed salt (the A+B b"\\xAB"*32 mutation).
"""

from __future__ import annotations

import inspect

import asqav.client as client


def test_shipped_default_is_unsalted_sha256() -> None:
    print("MODULE:", client.__file__)
    assert client._org_salt is None
    assert inspect.signature(client.init).parameters["org_salt"].default is None
    client._api_key, client._mode = "sk_test", "hash-only"
    body = client._build_sign_body(
        action_type="api:call",
        context={"model": "gpt-4"},
        session_id=None,
        agent_id="agt_x",
    )
    assert body["hash_algo"] == "sha256"
    assert body["hash"].startswith("sha256:")
