"""Archive sync against a real local S3-compatible process (761).

moto's in-process mock proves bucket logic; this proves --s3-endpoint-url
against a real HTTP S3 endpoint. Signed full-envelope pages re-verify offline
on read back. Skips when the moto server dependencies are absent.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time

import pytest
from typer.testing import CliRunner

from asqav.cli import _verify_archive_page, app
from tests._archive_page_761 import _page_full

pytest.importorskip(
    "moto.moto_server.threaded_moto_server",
    reason="moto[server] not installed; local-S3 e2e needs it",
)
pytest.importorskip("flask", reason="moto server needs flask")

runner = CliRunner()


def _receipt(i):
    return {"signature_id": f"sig_{i}", "agent_id": "agt_arc"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _awsenviron(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def test_sync_against_local_s3_process(monkeypatch, tmp_path):
    """End to end: signed pages sync through --s3-endpoint-url, read back OK."""
    import boto3

    _awsenviron(monkeypatch)
    port = _free_port()
    endpoint = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [sys.executable, "-m", "moto.server", "-H", "127.0.0.1", "-p", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        s3 = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint)
        deadline = time.monotonic() + 60
        while True:
            try:
                s3.list_buckets()
                break
            except Exception:
                assert time.monotonic() < deadline, "local S3 process never came up"
                time.sleep(0.5)
        s3.create_bucket(Bucket="arc", ObjectLockEnabledForBucket=True)

        pages = {
            0: _page_full([_receipt(1)], 1, True),
            1: _page_full([_receipt(2)], None, False, cursor=1),
        }
        calls = []

        def _fake_get(path):
            calls.append(path)
            return pages[int(path.split("cursor=")[1].split("&")[0])]

        import asqav.cli as cli_mod

        monkeypatch.setattr(cli_mod, "_init_sdk", lambda: None)
        import asqav.client as client_mod

        monkeypatch.setattr(client_mod, "_get", _fake_get)

        out = str(tmp_path / "arc")
        result = runner.invoke(
            app,
            ["archive", "sync", "--out", out, "--s3", "s3://arc/pages",
             "--s3-endpoint-url", endpoint],
        )
        assert result.exit_code == 0, result.output

        fresh = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint)
        for cursor in (0, 1):
            raw = fresh.get_object(Bucket="arc", Key=f"pages/page-{cursor}.json")
            _verify_archive_page(json.loads(raw["Body"].read().decode()))
        head = fresh.head_object(Bucket="arc", Key="pages/page-0.json")
        assert head["ObjectLockMode"] == "COMPLIANCE"
        assert "ObjectLockRetainUntilDate" in head
        assert fresh.get_object(Bucket="arc", Key="pages/cursor")["Body"].read() == b"1"
    finally:
        server.terminate()
        server.wait(timeout=30)
