"""Archive sync to S3 (criterion 761, slice 3: customer bucket e2e)."""

from __future__ import annotations

import datetime
import hashlib
import json
import os

import boto3
from moto import mock_aws
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


def _pageserver(monkeypatch, pages):
    """Serve canned pages by cursor; records fetched cursors."""
    calls = []

    def _fake_get(path):
        calls.append(path)
        cursor = int(path.split("cursor=")[1].split("&")[0])
        return pages[cursor]

    import asqav.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_init_sdk", lambda: None)
    import asqav.client as client_mod

    monkeypatch.setattr(client_mod, "_get", _fake_get)
    return calls


def _awsenviron(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def _bucket(name="arc"):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=name, ObjectLockEnabledForBucket=True)
    return s3


def _head(s3, key, bucket="arc"):
    return s3.head_object(Bucket=bucket, Key=key)


@mock_aws
def test_s3_sync_writes_pages_with_retention(monkeypatch, tmp_path):
    """Pages land in the bucket under COMPLIANCE retention."""
    _awsenviron(monkeypatch)
    with mock_aws():
        s3 = _bucket()
        _pageserver(
            monkeypatch,
            {0: _page([_receipt(1)], 1, True), 1: _page([_receipt(2)], None, False)},
        )
        out = str(tmp_path / "arc")
        result = runner.invoke(
            app, ["archive", "sync", "--out", out, "--s3", "s3://arc/pages"]
        )
        assert result.exit_code == 0, result.output
        for key in ("pages/page-0.json", "pages/page-1.json"):
            head = _head(s3, key)
            assert head["ObjectLockMode"] == "COMPLIANCE"
            assert head["ObjectLockRetainUntilDate"] > datetime.datetime.now(
                datetime.timezone.utc
            )
        cur = s3.get_object(Bucket="arc", Key="pages/cursor")
        assert cur["Body"].read().decode() == "1"
        assert os.path.exists(os.path.join(out, "page-0.json"))


@mock_aws
def test_s3_resume_has_no_gaps_or_duplicates(monkeypatch, tmp_path):
    """A re-run fetches only new pages; one version per stored key."""
    _awsenviron(monkeypatch)
    with mock_aws():
        s3 = _bucket()
        first = _page([_receipt(1)], 1, True)
        body = json.dumps(first, indent=2, default=str).encode()
        s3.put_object(Bucket="arc", Key="pages/page-0.json", Body=body)
        s3.put_object(Bucket="arc", Key="pages/cursor", Body=b"1")
        calls = _pageserver(monkeypatch, {1: _page([_receipt(2)], None, False)})
        out = str(tmp_path / "arc")
        result = runner.invoke(
            app, ["archive", "sync", "--out", out, "--s3", "s3://arc/pages"]
        )
        assert result.exit_code == 0, result.output
        assert "cursor=1" in calls[0]
        assert len(calls) == 1
        got = s3.list_objects_v2(Bucket="arc", Prefix="pages/page-")
        assert sorted(o["Key"] for o in got["Contents"]) == [
            "pages/page-0.json",
            "pages/page-1.json",
        ]
        vers = s3.list_object_versions(Bucket="arc", Prefix="pages/")
        assert len(vers["Versions"]) == 3  # 2 pages + cursor, no dupes


@mock_aws
def test_s3_abort_on_digest_mismatch_writes_nothing(monkeypatch, tmp_path):
    """A tampered page stops the sync before any object is stored."""
    _awsenviron(monkeypatch)
    with mock_aws():
        s3 = _bucket()
        bad = _page([_receipt(1)], None, False)
        bad["page_digest"] = "sha256:" + "0" * 64
        _pageserver(monkeypatch, {0: bad})
        out = str(tmp_path / "arc")
        result = runner.invoke(
            app, ["archive", "sync", "--out", out, "--s3", "s3://arc/pages"]
        )
        assert result.exit_code == 1
        assert "mismatch" in result.output
        assert s3.list_objects_v2(Bucket="arc").get("KeyCount", 0) == 0


@mock_aws
def test_s3_stored_pages_verify_offline(monkeypatch, tmp_path):
    """Every stored page re-derives its digest from bucket bytes alone."""
    _awsenviron(monkeypatch)
    with mock_aws():
        s3 = _bucket()
        _pageserver(
            monkeypatch,
            {0: _page([_receipt(1)], 1, True), 1: _page([_receipt(2)], None, False)},
        )
        out = str(tmp_path / "arc")
        result = runner.invoke(
            app, ["archive", "sync", "--out", out, "--s3", "s3://arc/pages"]
        )
        assert result.exit_code == 0, result.output
        for cursor in (0, 1):
            raw = s3.get_object(Bucket="arc", Key=f"pages/page-{cursor}.json")
            page = json.loads(raw["Body"].read().decode())
            expect = "sha256:" + hashlib.sha256(
                canonical_json(
                    {
                        "organization_id": page["organization_id"],
                        "receipts": page["receipts"],
                    }
                )
            ).hexdigest()
            assert expect == page["page_digest"]


def test_s3_bad_url_is_usage_error(monkeypatch, tmp_path):
    """A non-S3 URL is rejected before any fetch or write happens."""
    import asqav.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_init_sdk", lambda: None)
    out = str(tmp_path / "arc")
    result = runner.invoke(app, ["archive", "sync", "--out", out, "--s3", "https://x/y"])
    assert result.exit_code == 2
    assert "s3://bucket/prefix" in result.output
    assert not os.path.exists(out)


@mock_aws
def test_s3_missing_bucket_aborts(monkeypatch, tmp_path):
    """Syncing into a bucket that does not exist fails instead of duping."""
    _awsenviron(monkeypatch)
    with mock_aws():
        _pageserver(monkeypatch, {0: _page([_receipt(1)], None, False)})
        out = str(tmp_path / "arc")
        result = runner.invoke(
            app, ["archive", "sync", "--out", out, "--s3", "s3://ghost/pages"]
        )
        assert result.exit_code == 1
