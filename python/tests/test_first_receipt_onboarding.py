"""First-receipt onboarding: verify output, API base resolution, quickstart, errors, snippet.

Every network call goes to an in-process stub or to 127.0.0.1:9 (closed port).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import asqav
from asqav import async_client, credentials
from asqav import client as client_mod
from asqav.cli import app

runner = CliRunner()
DEAD_BASE = "http://127.0.0.1:9/api/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCKED: list[str] = []
DASHBOARD_SNIPPET = (
    "import asqav\n"
    "asqav.init()\n"
    'agent = asqav.Agent.create("my-first-agent")\n'
    'receipt = agent.sign("test", {"hello": "world"})\n'
    'print("Receipt:", receipt.verification_url)\n'
)

_ERROR_BODIES: dict[str, tuple[int, bytes, str]] = {
    "502html": (
        502,
        b"<html><head><title>502 Bad Gateway</title></head>\n<body>cloudflare</body></html>",
        "text/html",
    ),
    "403": (403, b'{"detail": "API key lacks scope agents:write"}', "application/json"),
    "403code": (
        403,
        b'{"detail": "API key lacks scope agents:write", "code": "scope_missing"}',
        "application/json",
    ),
    "422": (
        422,
        b'{"detail": [{"loc": ["body", "name"], "msg": "String should have at most 64 '
        b'characters", "type": "string_too_long"}]}',
        "application/json",
    ),
}


class _Stub:
    def __init__(self) -> None:
        self.mode = "ok"
        self.detail: dict[str, Any] = {}
        self.requests: list[tuple[str, str]] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/api/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, code: int, body: Any, ctype: str = "application/json") -> None:
                raw = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _route(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                stub.requests.append((method, self.path))
                if stub.mode in _ERROR_BODIES:
                    code, raw, ctype = _ERROR_BODIES[stub.mode]
                    return self._send(code, raw, ctype)
                if stub.mode == "html200":
                    return self._send(200, b"<html><body>Sign in to the Wi-Fi</body></html>",
                                      "text/html")
                code, payload = stub.answer(method, self.path.split("?")[0], body)
                return self._send(code, payload)

            def do_GET(self) -> None:
                self._route("GET")

            def do_POST(self) -> None:
                self._route("POST")

        return Handler

    def answer(self, method: str, path: str, body: bytes) -> tuple[int, Any]:
        now = "2026-10-11T00:00:00+00:00"
        if path.endswith("/health"):
            return 200, {"status": "healthy"}
        if "/maintenance/" in path or path.endswith("/emergency-halt"):
            return 200, {"ok": True}
        if path.rstrip("/").endswith("/agents") and method == "GET":
            return 200, []
        if path.endswith("/agents/create"):
            name = json.loads(body or b"{}").get("name")
            return 200, {"agent_id": "agt_stub01", "name": name, "public_key": "AAAA",
                         "key_id": "key_stub01", "algorithm": "ml-dsa-65",
                         "capabilities": [], "created_at": now}
        if path.endswith("/sign"):
            return 200, {"signature": "c2ln", "signature_id": "sig_stub01",
                         "action_id": "act_stub01", "timestamp": now,
                         "verification_url": "http://127.0.0.1/verify/sig_stub01",
                         "algorithm": "ml-dsa-65", "compliance_mode": True,
                         "action_ref": "sha256:" + "a" * 64,
                         "previousReceiptHash": "0" * 64}
        if "/verify/" in path:
            return 200, self.verify_doc(path.rsplit("/", 1)[-1], now)
        return 404, {"detail": "Not Found"}

    def verify_doc(self, sig_id: str, now: str) -> dict[str, Any]:
        sig_ok = self.mode != "badsig"
        return {"signature_id": sig_id, "agent_id": "agt_stub01", "agent_name": "my-agent",
                "action_id": "act_stub01", "action_type": "test", "signature": "c2ln",
                "algorithm": "ml-dsa-65", "signed_at": now, "verified": False,
                "verification_url": f"http://127.0.0.1/verify/{sig_id}",
                "verification_detail": {
                    "signer_key_match": True, "signature_valid": sig_ok,
                    "algorithm_match": True, "agent_active": True,
                    "validation_label": ("anchor_pending_no_cryptographic_proof"
                                         if sig_ok else "invalid_signature"),
                    "anchor_status_ots": "pending", **self.detail}}


@pytest.fixture
def stub() -> Any:
    s = _Stub()
    yield s
    s.server.shutdown()
    s.server.server_close()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    # Default base is a closed local port and asqav.com never resolves, so no test reaches prod.
    for name in ("ASQAV_API_URL", "ASQAV_API_BASE", "ASQAV_API_KEY", "ASQAV_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ASQAV_CREDENTIALS_PATH", str(tmp_path / "credentials"))
    monkeypatch.setattr(credentials, "_DEFAULT_API_BASE", DEAD_BASE, raising=False)
    real_getaddrinfo = socket.getaddrinfo
    BLOCKED.clear()

    def guarded(host: Any, *args: Any, **kwargs: Any) -> Any:
        if "asqav.com" in str(host):
            BLOCKED.append(str(host))
            raise OSError(f"test blocked a lookup of {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", guarded)
    monkeypatch.setattr("asqav.retry.time.sleep", lambda _s: None)
    monkeypatch.setattr(client_mod.time, "sleep", lambda _s: None)
    names = ("_api_key", "_api_base", "_explicit_api_base", "_client", "_mode")
    saved = {n: getattr(client_mod, n, None) for n in names}
    client_mod._api_base = DEAD_BASE
    client_mod._explicit_api_base = None
    yield
    for name, value in saved.items():
        setattr(client_mod, name, value)


def _save_login(tmp_path: Path, base: str) -> None:
    (tmp_path / "credentials").write_text(json.dumps({"api_key": "sk_test", "api_base": base}))


# === 1. verify reports signature validity and anchor state separately ===


def test_verify_cli_fresh_receipt_is_valid_with_anchor_pending(stub: _Stub) -> None:
    client_mod._api_base = stub.base
    result = runner.invoke(app, ["verify", "sig_stub01"])
    assert result.exit_code == 0, result.output
    assert "Signature: valid" in result.output
    assert "Anchor: pending" in result.output
    assert "invalid" not in result.output


def test_verify_cli_bad_signature_reads_invalid(stub: _Stub) -> None:
    client_mod._api_base = stub.base
    stub.mode = "badsig"
    result = runner.invoke(app, ["verify", "sig_stub01"])
    assert "Signature: invalid" in result.output


REJECTIONS = [
    "signer_key_changed",
    "agent_revoked_before_issuance",
    "chain_break",
    "signature_expired",
    "agent_unknown",
    "compliance_mode_anchor_required",
]


@pytest.mark.parametrize("label", REJECTIONS)
def test_verify_cli_server_rejection_reads_invalid_and_exits_1(stub: _Stub, label: str) -> None:
    # signature_valid stays true on these; the server's label is the verdict.
    client_mod._api_base = stub.base
    stub.detail = {"validation_label": label, "anchor_valid_ots": True,
                   "anchor_status_ots": "valid"}
    result = runner.invoke(app, ["verify", "sig_stub01"])
    assert f"Signature: invalid ({label})" in result.output
    assert result.exit_code == 1


def test_verify_cli_one_invalid_anchor_reads_invalid(stub: _Stub) -> None:
    client_mod._api_base = stub.base
    stub.detail = {"validation_label": "anchor_invalid", "anchor_valid_ots": True,
                   "anchor_status_ots": "valid", "anchor_status_rfc3161": "invalid"}
    result = runner.invoke(app, ["verify", "sig_stub01"])
    assert "Anchor: invalid" in result.output
    assert "Signature: invalid (anchor_invalid)" in result.output
    assert result.exit_code == 1


def test_verify_dict_carries_signature_and_anchor(stub: _Stub) -> None:
    client_mod._api_base = stub.base
    result = asqav.verify("sig_stub01")
    assert result["verified"] is False
    assert result["signature_valid"] is True
    assert result["anchor"] == "pending"


# === 2. one API base resolution for init(), the clients and doctor ===


def test_init_uses_base_saved_by_login(tmp_path: Path) -> None:
    _save_login(tmp_path, "http://saved.test/api/v1")
    asqav.init(api_key="sk_test")
    assert client_mod._api_base == "http://saved.test/api/v1"


def test_init_honours_asqav_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASQAV_API_BASE", "http://env-base.test/api/v1")
    asqav.init(api_key="sk_test")
    assert client_mod._api_base == "http://env-base.test/api/v1"


def test_base_resolution_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert credentials.resolve_api_base() == DEAD_BASE
    _save_login(tmp_path, "http://file.test")
    assert credentials.resolve_api_base() == "http://file.test"
    monkeypatch.setenv("ASQAV_API_BASE", "http://base.test")
    assert credentials.resolve_api_base() == "http://base.test"
    monkeypatch.setenv("ASQAV_API_URL", "http://url.test")
    asqav.init(api_key="sk_test")
    assert client_mod._api_base == "http://url.test"
    asqav.init(api_key="sk_test", base_url="http://arg.test")
    assert client_mod._api_base == "http://arg.test"
    # A later init() without a base (govern() does this) keeps the explicit one.
    asqav.init(api_key="sk_test")
    assert client_mod._api_base == "http://arg.test"


def test_doctor_probes_the_saved_base(stub: _Stub, tmp_path: Path) -> None:
    _save_login(tmp_path, stub.base)
    result = runner.invoke(app, ["doctor"])
    assert f"API base: {stub.base}" in result.output
    assert result.exit_code == 0, result.output
    assert ("GET", "/api/v1/agents") in stub.requests


@pytest.mark.parametrize(
    ("command", "env", "path"),
    [
        (["migrate", "run", "v3-20"], "ASQAV_MAINTENANCE_KEY",
         "/api/v1/maintenance/run-migration-v3-20"),
        (["org", "halt", "org_1", "--yes"], "ASQAV_SESSION_TOKEN",
         "/api/v1/orgs/org_1/emergency-halt"),
    ],
)
def test_admin_commands_use_the_saved_base(stub: _Stub, tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch,
                                           command: list[str], env: str, path: str) -> None:
    _save_login(tmp_path, stub.base)
    monkeypatch.setenv(env, "secret_test_value")
    runner.invoke(app, command)
    assert ("POST", path) in stub.requests
    assert BLOCKED == []


def test_async_client_reads_config_set_by_init(stub: _Stub) -> None:
    asqav.init(api_key="sk_test", base_url=stub.base)
    assert async_client._get_config() == (stub.base, "sk_test")


# === 3. quickstart prints code that runs, and says what it does ===


def _python_blocks(output: str) -> list[str]:
    # Sections between "--- title ---" lines that compile as Python and use asqav.
    blocks, current = [], []
    for line in output.splitlines() + ["--- end ---"]:
        if line.startswith("--- "):
            text = "\n".join(current).strip()
            if "asqav" in text and _compiles(text):
                blocks.append(text)
            current = []
        else:
            current.append(line)
    return blocks


def _compiles(text: str) -> bool:
    try:
        compile(text, "<printed>", "exec")
    except SyntaxError:
        return False
    return True


def _run_printed(code: str, stub: _Stub, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("ASQAV_API_KEY", "sk_test")
    monkeypatch.setenv("ASQAV_API_URL", stub.base)
    client_mod._api_base = stub.base
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "<printed>", "exec"), {"__name__": "__printed__"})
    return out.getvalue()


def test_quickstart_printed_code_runs(stub: _Stub, monkeypatch: pytest.MonkeyPatch) -> None:
    output = runner.invoke(app, ["quickstart"]).output
    blocks = _python_blocks(output)
    assert blocks, output
    printed = "".join(_run_printed(block, stub, monkeypatch) for block in blocks)
    assert "Receipt: http://127.0.0.1/verify/sig_stub01" in printed
    assert any(path.endswith("/sign") for _, path in stub.requests)


def test_quickstart_help_says_what_it_does() -> None:
    help_text = " ".join(runner.invoke(app, ["quickstart", "--help"]).output.split())
    assert "first-receipt" in help_text
    assert "Writes no files" in help_text


def test_init_snippet_runs(stub: _Stub, monkeypatch: pytest.MonkeyPatch) -> None:
    from asqav.cli import _onboarding_snippet

    printed = _run_printed(_onboarding_snippet("litellm"), stub, monkeypatch)
    assert "Receipt:" in printed


# === 4. transport failures and non-JSON bodies raise one-line AsqavErrors ===


@pytest.fixture(params=["httpx", "urllib"])
def transport(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "urllib":
        monkeypatch.setattr(client_mod, "_HTTPX_AVAILABLE", False)
    return request.param


def _create_error(base: str) -> Exception:
    asqav.init(api_key="sk_test", base_url=base)
    with pytest.raises(asqav.AsqavError) as info:
        asqav.Agent.create("my-first-agent")
    return info.value


def test_connection_refused_is_one_line_asqav_error(transport: str) -> None:
    err = _create_error(DEAD_BASE)
    assert isinstance(err, asqav.AsqavConnectionError)
    # Same hierarchy as TypeScript: an APIError with status 0.
    assert isinstance(err, asqav.APIError) and err.status_code == 0
    assert "127.0.0.1:9" in str(err)
    assert "\n" not in str(err)


def test_verify_connection_refused_is_asqav_error(transport: str) -> None:
    with pytest.raises(asqav.AsqavConnectionError) as info:
        asqav.verify_signature("sig_x")
    assert "\n" not in str(info.value)


def test_async_connection_refused_is_asqav_error() -> None:
    asqav.init(api_key="sk_test", base_url=DEAD_BASE)
    with pytest.raises(asqav.AsqavConnectionError):
        asyncio.run(async_client._async_get("/agents"))


def test_html_error_page_is_one_line(stub: _Stub, transport: str) -> None:
    stub.mode = "502html"
    err = _create_error(stub.base)
    assert isinstance(err, asqav.APIError) and err.status_code == 502
    assert "<html" not in str(err) and "\n" not in str(err)
    assert "502" in str(err)


def test_non_json_success_body_is_asqav_error(stub: _Stub, transport: str) -> None:
    stub.mode = "html200"
    err = _create_error(stub.base)
    assert isinstance(err, asqav.AsqavResponseError)
    assert "\n" not in str(err)
    client_mod._api_base = stub.base
    with pytest.raises(asqav.AsqavResponseError):
        asqav.verify_signature("sig_x")


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("403", "API key lacks scope agents:write"),
        ("403code", "API key lacks scope agents:write (scope_missing)"),
        ("422", "name: String should have at most 64 characters"),
    ],
)
def test_server_reason_survives(stub: _Stub, transport: str, mode: str, reason: str) -> None:
    stub.mode = mode
    assert str(_create_error(stub.base)) == reason


# === 5. one canonical first-receipt snippet ===


def test_canonical_snippet_is_the_dashboard_text() -> None:
    from asqav.cli import FIRST_RECEIPT_KEY_STEP, FIRST_RECEIPT_SNIPPET

    assert FIRST_RECEIPT_SNIPPET == DASHBOARD_SNIPPET
    assert FIRST_RECEIPT_KEY_STEP == "export ASQAV_API_KEY=<paste your key here>"


@pytest.mark.parametrize(
    "doc", ["README.md", "python/README.md", "python/README-ZH.md", "python/docs/CLI.md"]
)
def test_docs_carry_the_canonical_snippet(doc: str) -> None:
    from asqav.cli import FIRST_RECEIPT_KEY_STEP, FIRST_RECEIPT_SNIPPET

    text = (REPO_ROOT / doc).read_text(encoding="utf-8")
    assert FIRST_RECEIPT_SNIPPET in text
    assert FIRST_RECEIPT_KEY_STEP in text


@pytest.mark.parametrize("command", [["quickstart"], ["init"]])
def test_cli_prints_the_canonical_snippet(command: list[str], tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from asqav.cli import FIRST_RECEIPT_SNIPPET

    monkeypatch.chdir(tmp_path)
    assert FIRST_RECEIPT_SNIPPET in runner.invoke(app, command).output
