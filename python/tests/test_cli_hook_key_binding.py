"""The PreToolUse gate opens only for a key published for the gated agent (741).

Each test runs the real `asqav hook pretool` in a subprocess against a local
platform fake (agent, sign and key-directory endpoints on 127.0.0.1). Nothing
here reaches api.asqav.com, so a green run says nothing about production.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import pytest

import asqav
from asqav import cli_hook
from asqav.verifier import verify_receipt as vr

ML_DSA_65 = pytest.importorskip("dilithium_py.ml_dsa").ML_DSA_65
pytest.importorskip("typer")
pytest.importorskip("httpx")

ORG_A = "11111111-1111-4111-8111-111111111111"
ORG_B = "22222222-2222-4222-8222-222222222222"
GATED_ID = "agt_gated"
EVENT = {
    "session_id": "s-741",
    "tool_use_id": "toolu_741",
    "tool_name": "Bash",
    "tool_input": {"command": "ls"},
}


@dataclass(frozen=True)
class _Party:
    agent_id: str
    org: str
    kid: str
    pk: bytes
    sk: bytes
    thumb: str

    def row(self, *, agent_id: str | None = None) -> dict[str, Any]:
        row = {
            "kid": self.kid,
            "agent_id": self.agent_id if agent_id is None else agent_id,
            "issuer_id": self.org,
            "org_id": self.org,
            "alg": "ML-DSA-65",
            "kty": "AKP",
            "public_key": base64.b64encode(self.pk).decode(),
            "status": "active",
            "revoked_at": None,
            "key_thumbprint": self.thumb,
        }
        if agent_id == "":
            del row["agent_id"]
        return row


def _party(agent_id: str, org: str, kid: str) -> _Party:
    pk, sk = ML_DSA_65.keygen()
    thumb = vr.thumbprint_for_key(alg="ML-DSA-65", public_key=pk)
    return _Party(agent_id, org, kid, pk, sk, thumb)


@pytest.fixture(scope="module")
def cast() -> dict[str, _Party]:
    return {
        "gated": _party(GATED_ID, ORG_A, "key-gated"),
        "sibling": _party("agt_sibling", ORG_A, "key-sibling"),
        "foreign": _party("agt_foreign", ORG_B, "key-foreign"),
    }


def _mint(
    body: dict[str, Any],
    *,
    signer: _Party,
    kid: str,
    issuer_id: str,
    thumb: str | None,
) -> dict[str, Any]:
    """Signer's receipt naming the gated agent and answering the POSTed body."""
    digest = body["hash"]
    payload: dict[str, Any] = {
        "type": "protectmcp:decision",
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "issuer_id": issuer_id,
        "agent_id": GATED_ID,
        "action_ref": digest,
        "hash": digest,
        "policy_digest": "sha256:" + "3" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
        "invocation_ref": body["invocation_ref"],
    }
    if thumb is not None:
        payload["key_thumbprint"] = thumb
    sig = ML_DSA_65.sign(signer.sk, vr.canonical_json(payload))
    return {
        "signature": {"alg": "ML-DSA-65", "kid": kid, "sig": base64.b64encode(sig).decode()},
        "signature_id": "sig_741",
        "action_id": "act_741",
        "timestamp": payload["issued_at"],
        "verification_url": "https://example.invalid/verify/sig_741",
        "payload": payload,
        "anchors": [],
        "decision": "allow",
        "compliance_mode": True,
    }


class _Platform:
    """Local stand-in for the agent, sign and key-directory endpoints."""

    def __init__(self, directory: list[dict[str, Any]]) -> None:
        self.directory = directory
        self.mint: Callable[[dict[str, Any]], dict[str, Any]] = lambda _b: {}
        self.sign_posts: list[dict[str, Any]] = []
        self.jwks_fetches = 0
        self.unexpected: list[str] = []
        platform = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a: Any) -> None:
                pass

            def _send(self, doc: dict[str, Any], status: int = 200) -> None:
                raw = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/.well-known/jwks.json":
                    platform.jwks_fetches += 1
                    self._send({"keys": platform.directory})
                elif self.path == f"/api/v1/agents/{GATED_ID}":
                    self._send(
                        {
                            "agent_id": GATED_ID,
                            "name": "gated",
                            "public_key": "pk",
                            "key_id": "key-gated",
                            "algorithm": "ml-dsa-65",
                            "capabilities": [],
                            "created_at": "2026-10-10T00:00:00+00:00",
                        }
                    )
                else:
                    platform.unexpected.append(f"GET {self.path}")
                    self._send({"detail": "not found"}, 404)

            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path == f"/api/v1/agents/{GATED_ID}/sign":
                    body = json.loads(raw)
                    platform.sign_posts.append(body)
                    self._send(platform.mint(body))
                else:
                    platform.unexpected.append(f"POST {self.path}")
                    self._send({"detail": "not found"}, 404)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self) -> _Platform:
        self.thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _run_hook(platform: _Platform, tmp_path) -> subprocess.CompletedProcess[str]:
    """`asqav hook pretool` as the harness runs it, on the local platform."""
    # The tree under test comes first on the path, so a stale install cannot answer.
    src = os.path.dirname(os.path.dirname(os.path.abspath(asqav.__file__)))
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "PYTHONPATH": src,
        "ASQAV_API_KEY": "sk_test_741",
        "ASQAV_AGENT_ID": GATED_ID,
        "ASQAV_API_URL": f"{platform.base}/api/v1",
        cli_hook._JWKS_CACHE_ENV: str(tmp_path / "jwks-cache.json"),
        cli_hook._DEADLINE_ENV: "60",
        "ASQAV_CREDENTIALS_PATH": str(tmp_path / "credentials"),
        "NO_PROXY": "127.0.0.1",
    }
    return subprocess.run(
        [sys.executable, "-m", "asqav.cli", "hook", "pretool"],
        input=json.dumps(EVENT),
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )


def _gate(
    cast, tmp_path, make_receipt, directory=None
) -> tuple[subprocess.CompletedProcess[str], _Platform]:
    rows = directory if directory is not None else [p.row() for p in cast.values()]
    with _Platform(rows) as platform:
        platform.mint = lambda body: make_receipt(body, cast)
        result = _run_hook(platform, tmp_path)
    assert len(platform.sign_posts) == 1, f"the gate never reached the signer: {result.stderr}"
    assert platform.unexpected == []
    return result, platform


# Genuine permits: signed by the gated agent's key, whatever name it carries.
def _genuine_with_thumbprint(body, cast):
    g = cast["gated"]
    return _mint(body, signer=g, kid=ORG_A, issuer_id=ORG_A, thumb=g.thumb)


def _genuine_org_kid_no_thumbprint(body, cast):
    g = cast["gated"]
    return _mint(body, signer=g, kid=ORG_A, issuer_id=ORG_A, thumb=None)


def _genuine_key_kid_no_thumbprint(body, cast):
    g = cast["gated"]
    return _mint(body, signer=g, kid=g.kid, issuer_id=ORG_A, thumb=None)


@pytest.mark.parametrize(
    "make_receipt",
    [
        pytest.param(_genuine_with_thumbprint, id="thumbprint"),
        pytest.param(_genuine_org_kid_no_thumbprint, id="org-kid-no-thumbprint"),
        pytest.param(_genuine_key_kid_no_thumbprint, id="own-key-kid-no-thumbprint"),
    ],
)
def test_a_permit_signed_by_the_gated_agents_own_key_exits_zero(
    cast, tmp_path, make_receipt
) -> None:
    result, _platform = _gate(cast, tmp_path, make_receipt)
    assert result.returncode == 0, result.stderr
    assert "permit sig_741" in result.stderr


# Receipts that name the gated agent but verify against a key it does not own.
def _sibling_key_by_thumbprint(body, cast):
    s = cast["sibling"]
    return _mint(body, signer=s, kid=ORG_A, issuer_id=ORG_A, thumb=s.thumb)


def _other_org_key_by_thumbprint(body, cast):
    f = cast["foreign"]
    return _mint(body, signer=f, kid=ORG_B, issuer_id=ORG_B, thumb=f.thumb)


def _sibling_key_selected_by_kid(body, cast):
    s = cast["sibling"]
    return _mint(body, signer=s, kid=s.kid, issuer_id=ORG_A, thumb=None)


def _other_org_key_selected_by_kid(body, cast):
    f = cast["foreign"]
    return _mint(body, signer=f, kid=f.kid, issuer_id=ORG_B, thumb=None)


@pytest.mark.parametrize(
    "make_receipt",
    [
        pytest.param(_sibling_key_by_thumbprint, id="same-org-sibling-key"),
        pytest.param(_other_org_key_by_thumbprint, id="other-org-key"),
        pytest.param(_sibling_key_selected_by_kid, id="sibling-key-by-kid-no-thumbprint"),
        pytest.param(_other_org_key_selected_by_kid, id="other-org-key-by-kid-no-thumbprint"),
    ],
)
def test_a_receipt_naming_the_gated_agent_but_signed_by_another_agents_key_exits_two(
    cast, tmp_path, make_receipt
) -> None:
    result, _platform = _gate(cast, tmp_path, make_receipt)
    assert result.returncode == 2, result.stderr
    assert "did not verify" in result.stderr
    assert "blocking tool call" in result.stderr
    assert "permit" not in result.stderr


def test_a_published_key_with_no_agent_id_cannot_open_the_gate(cast, tmp_path) -> None:
    """The cloud always publishes agent_id; an entry without it fails closed."""
    rows = [cast["gated"].row(agent_id=""), cast["sibling"].row(), cast["foreign"].row()]
    result, _platform = _gate(cast, tmp_path, _genuine_with_thumbprint, directory=rows)
    assert result.returncode == 2, result.stderr
    assert f"no key for the gated agent {GATED_ID}" in result.stderr


def test_a_directory_that_never_lists_the_gated_agent_blocks_after_one_refresh(
    cast, tmp_path
) -> None:
    rows = [cast["sibling"].row(), cast["foreign"].row()]
    result, platform = _gate(cast, tmp_path, _genuine_with_thumbprint, directory=rows)
    assert result.returncode == 2, result.stderr
    assert f"no key for the gated agent {GATED_ID}" in result.stderr
    assert platform.jwks_fetches == 2, "expected the cache fill and exactly one refresh"


def test_keys_of_agent_keeps_only_the_gated_agents_entries() -> None:
    mixed = {
        "keys": [
            {"kid": "a", "agent_id": "agt_1"},
            {"kid": "b", "agent_id": "agt_2"},
            {"kid": "c"},
            "not-a-dict",
            {"kid": "d", "agent_id": "agt_1"},
        ]
    }
    kept = cli_hook._keys_of_agent(mixed, "agt_1")
    assert [k["kid"] for k in kept["keys"]] == ["a", "d"]
    assert cli_hook._keys_of_agent({}, "agt_1") == {"keys": []}
