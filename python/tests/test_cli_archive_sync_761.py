"""Archive sync command (criterion 761, slice 2: CLI to customer dir)."""

from __future__ import annotations

import hashlib
import json
import os

from typer.testing import CliRunner

from asqav._jcs import canonical_json
from asqav.cli import app

runner = CliRunner()


def _page(receipts, next_cursor, has_more):
    body = {"organization_id": "org_arc", "receipts": receipts}
    digest = "sha256:" + hashlib.sha256(canonical_json(body)).hexdigest()
    return {
        "organization_id": "org_arc",
        "receipts": receipts,
        "next_cursor": next_cursor,
        "has_more": has_more,
        "page_digest": digest,
        "page_signature": "c2ln",
        "page_signature_algorithm": "ml-dsa-65",
        "page_public_key": "a2V5",
    }


def _receipt(i):
    return {"signature_id": f"sig_{i}", "agent_id": "agt_arc"}


def test_sync_writes_pages_and_resumes(monkeypatch, tmp_path):
    """Two pages sync to files; re-running resumes from the cursor."""
    calls = []

    def _fake_get(path):
        calls.append(path)
        if "cursor=0" in path:
            return _page([_receipt(1)], 1, True)
        return _page([_receipt(2)], None, False)

    import asqav.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_init_sdk", lambda: None)
    import asqav.client as client_mod

    monkeypatch.setattr(client_mod, "_get", _fake_get)
    out = str(tmp_path / "arc")
    result = runner.invoke(app, ["archive", "sync", "--out", out, "--limit", "1"])
    assert result.exit_code == 0, result.output
    assert os.path.exists(os.path.join(out, "page-0.json"))
    assert os.path.exists(os.path.join(out, "page-1.json"))
    with open(os.path.join(out, "cursor")) as f:
        assert f.read().strip() == "1"
    # Resume: cursor file present, first fetch starts at 1.
    calls.clear()
    result = runner.invoke(app, ["archive", "sync", "--out", out, "--limit", "1"])
    assert result.exit_code == 0, result.output
    assert "cursor=1" in calls[0]


def test_sync_aborts_on_digest_mismatch(monkeypatch, tmp_path):
    """A tampered page digest stops the sync with exit 1."""
    bad = _page([_receipt(1)], None, False)
    bad["page_digest"] = "sha256:" + "0" * 64
    import asqav.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_init_sdk", lambda: None)
    import asqav.client as client_mod

    monkeypatch.setattr(client_mod, "_get", lambda path: bad)
    out = str(tmp_path / "arc")
    result = runner.invoke(app, ["archive", "sync", "--out", out])
    assert result.exit_code == 1
    assert "mismatch" in result.output
