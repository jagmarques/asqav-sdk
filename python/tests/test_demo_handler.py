"""Tests for the local demo HTTP handler (`asqav demo`).

Covers `DemoHandler.do_POST`, which is uncovered by default because the
demo server is launched interactively. The strategy is to instantiate a
`ThreadingHTTPServer` on an ephemeral port and drive it with `urllib`
so the real handler runs end-to-end against the real `_sign` / `_verify`
helpers - no mocks on the demo internals.
"""

from __future__ import annotations

import json
import os
import socketserver
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.demo import DemoHandler, DemoState


@contextmanager
def _running_server() -> Iterator[tuple[str, DemoState]]:
    """Start a DemoHandler on an ephemeral port; yield the base URL + state."""
    state = DemoState()
    DemoHandler.state = state
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), DemoHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def _post(url: str, body: dict | None, *, raw: bytes | None = None) -> tuple[int, dict]:
    """Issue a POST; return (status, decoded JSON body or {} on empty)."""
    if raw is not None:
        data = raw
    else:
        data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8") or "{}"
        return exc.code, json.loads(body_text)


# === Happy path: /api/decide signs a receipt for a known scenario ===


def test_do_post_decide_signs_and_records() -> None:
    """POST /api/decide with valid fields signs and stores the receipt."""
    with _running_server() as (base, state):
        status, body = _post(
            f"{base}/api/decide",
            {
                "id": "scenario-rmrf",
                "decision": "approved",
                "reason": "user confirmed cleanup is safe",
            },
        )
        assert status == 200
        assert body["ok"] is True
        receipt = body["receipt"]
        # Receipt carries the canonical fields used by the cloud's profile.
        assert receipt["payload"]["scenario_id"] == "scenario-rmrf"
        assert receipt["payload"]["policy_decision"] == "permit"
        assert receipt["signature"].startswith("hmac-sha256:")
        # State records the approval so the UI can render it.
        assert state.approvals["scenario-rmrf"]["decision"] == "approved"


# === Round trip: a receipt produced by /api/decide verifies via /api/verify ===


def test_do_post_verify_round_trip() -> None:
    """A receipt produced by /api/decide verifies via /api/verify."""
    with _running_server() as (base, _state):
        _, decide = _post(
            f"{base}/api/decide",
            {
                "id": "scenario-wire",
                "decision": "denied",
                "reason": "amount exceeds the four-eyes policy ceiling",
            },
        )
        receipt = decide["receipt"]
        # policy_decision mirrors the controlled vocabulary (deny here).
        assert receipt["payload"]["policy_decision"] == "deny"

        status, body = _post(f"{base}/api/verify", receipt)
        assert status == 200
        assert body["valid"] is True


# === Error paths ===


def test_do_post_invalid_json_returns_400() -> None:
    """Non-JSON bodies are rejected with 400."""
    with _running_server() as (base, _state):
        status, body = _post(f"{base}/api/decide", None, raw=b"{not json")
        assert status == 400
        assert "invalid json" in body["error"]


def test_do_post_decide_rejects_bad_decision() -> None:
    """`decision` must be `approved` or `denied`."""
    with _running_server() as (base, _state):
        status, body = _post(
            f"{base}/api/decide",
            {"id": "scenario-rmrf", "decision": "maybe", "reason": "x" * 20},
        )
        assert status == 400
        assert "approved or denied" in body["error"]


def test_do_post_decide_rejects_short_reason() -> None:
    """Reasons under 10 chars are rejected so the audit trail stays meaningful."""
    with _running_server() as (base, _state):
        status, body = _post(
            f"{base}/api/decide",
            {"id": "scenario-rmrf", "decision": "approved", "reason": "short"},
        )
        assert status == 400
        assert "at least 10" in body["error"]


def test_do_post_decide_unknown_scenario_returns_404() -> None:
    """Unknown scenario ids surface as a 404."""
    with _running_server() as (base, _state):
        status, body = _post(
            f"{base}/api/decide",
            {
                "id": "scenario-ghost",
                "decision": "approved",
                "reason": "ten characters at least",
            },
        )
        assert status == 404
        assert body["error"] == "unknown scenario"


def test_do_post_unknown_path_returns_404() -> None:
    """POSTs to an unknown path return 404."""
    with _running_server() as (base, _state):
        status, body = _post(f"{base}/api/nope", {"x": 1})
        assert status == 404


def test_do_post_verify_with_garbage_returns_invalid() -> None:
    """`/api/verify` returns valid=False for a malformed receipt."""
    with _running_server() as (base, _state):
        status, body = _post(
            f"{base}/api/verify",
            {"payload": {"scenario_id": "fake"}, "signature": "not-a-real-sig"},
        )
        # The handler catches any exception and renders valid=False.
        assert status == 200
        assert body["valid"] is False


# === Edge case: empty body ===


def test_do_post_empty_body_falls_through_to_404() -> None:
    """Empty body decodes to `{}`; /api/decide path with empty body rejects on decision."""
    with _running_server() as (base, _state):
        req = urllib.request.Request(
            f"{base}/api/decide",
            data=b"",
            headers={"Content-Type": "application/json", "Content-Length": "0"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 400
