"""The Audit Pack CLI can ask for a pack whose receipts are verifiable.

The export sent no `include_signed_bytes`, so it took the server default and every
receipt came back with its signed bytes withheld. The recipient could check the
bundle's own signature and not one receipt inside it, because the preimage was
absent. There was no flag to ask for anything else.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

import asqav.cli as cli
import asqav.client as client


@pytest.fixture
def sent() -> dict[str, Any]:
    """Capture the body the CLI puts on the wire."""
    return {}


def _export(sent: dict[str, Any], **overrides: Any) -> None:
    def fake_post(path: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        sent["path"] = path
        sent["body"] = dict(body)
        return {"receipts": [], "bundle_digest": "sha256:x"}

    kwargs: dict[str, Any] = {
        "start": "2026-09-03T00:00:00Z",
        "end": "2026-09-04T00:00:00Z",
        "organization_id": "",
        "only_compliance": True,
        "include_signed_bytes": False,
        "output_file": "/dev/null",
    }
    kwargs.update(overrides)
    with (
        mock.patch.object(client, "_post", fake_post),
        mock.patch.object(cli, "_init_sdk", lambda: None),
    ):
        cli.audit_pack_export(**kwargs)


class TestTheFlagReachesTheWire:
    def test_the_member_is_always_sent(self, sent: dict[str, Any]) -> None:
        """Its absence was the defect: the export inherited whatever the server defaulted to."""
        _export(sent)
        assert "include_signed_bytes" in sent["body"], sent["body"]

    def test_the_default_is_the_minimising_one(self, sent: dict[str, Any]) -> None:
        """Content minimisation stays the default; the caller opts in deliberately."""
        _export(sent)
        assert sent["body"]["include_signed_bytes"] is False

    def test_asking_for_signed_bytes_is_carried(self, sent: dict[str, Any]) -> None:
        """The whole point: a pack an auditor can actually verify receipts in."""
        _export(sent, include_signed_bytes=True)
        assert sent["body"]["include_signed_bytes"] is True

    def test_the_other_members_are_unchanged(self, sent: dict[str, Any]) -> None:
        """The added member must not disturb the window or the compliance filter."""
        _export(sent, include_signed_bytes=True)
        assert sent["body"]["start"] == "2026-09-03T00:00:00Z"
        assert sent["body"]["end"] == "2026-09-04T00:00:00Z"
        assert sent["body"]["only_compliance"] is True
        assert sent["path"] == "/audit-pack/export"
