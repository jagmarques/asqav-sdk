"""Test-suite isolation for the file-backed credential layer.

Points ASQAV_CREDENTIALS_PATH at a per-test nonexistent path so no test reads a
real ~/.asqav/credentials file, which would otherwise flip the "no key" paths.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_asqav_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ASQAV_CREDENTIALS_PATH", str(tmp_path / "credentials"))


@pytest.fixture
def signing_transport():
    from asqav.client import _transport

    saved = (_transport._mode, _transport._org_salt, _transport._api_key)
    _transport._api_key = "sk_test"
    yield
    _transport._mode, _transport._org_salt, _transport._api_key = saved


_SRC = Path(__file__).resolve().parents[1] / "src"
_CHILD = """
import json, sys
from asqav import verify_receipt_offline
from asqav.verifier import verify_receipt as vr
from asqav.verifier.oracle.adapters.asqav_native import _flat_signed_fields
data = json.load(sys.stdin)
receipt = data["receipt"]
output = {
    "result": verify_receipt_offline(receipt, data["jwks"]),
    "fields": list(_flat_signed_fields(receipt)),
    "signed": list(vr._FLAT_SIGNED),
}
"""


def run_flat_receipt(fixture: Path, name: str, tamper=None, *, extra_output=None) -> dict:
    data = json.loads(fixture.read_text())
    receipt = data["receipts"][name]
    if tamper is not None:
        tamper(receipt)
    child = _CHILD
    if extra_output is not None:
        child += f"\noutput.update({extra_output})"
    done = subprocess.run(
        [sys.executable, "-c", child + "\nprint(json.dumps(output))"],
        input=json.dumps({"receipt": receipt, "jwks": data["jwks"]}),
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(_SRC)}, check=True,
    )
    return json.loads(done.stdout)
