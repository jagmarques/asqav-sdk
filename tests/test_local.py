"""Tests for asqav local queue-and-sync."""

from __future__ import annotations

import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from asqav.local import LocalQueue, local_sign


# ---------------------------------------------------------------------------
# LocalQueue unit tests
# ---------------------------------------------------------------------------


def test_enqueue_creates_file(tmp_path: object) -> None:
    """enqueue writes a JSON file in the queue directory."""
    queue = LocalQueue(queue_dir=str(tmp_path))
    item_id = queue.enqueue("agent_abc", "api:call", {"model": "gpt-4"})

    filepath = tmp_path / f"{item_id}.json"  # type: ignore[operator]
    assert filepath.exists()

    data = json.loads(filepath.read_text())
    assert data["agent_id"] == "agent_abc"
    assert data["action_type"] == "api:call"
    assert data["context"] == {"model": "gpt-4"}
    assert data["status"] == "pending"
    assert "queued_at" in data


def test_enqueue_returns_id(tmp_path: object) -> None:
    """enqueue returns a string ID matching the filename."""
    queue = LocalQueue(queue_dir=str(tmp_path))
    item_id = queue.enqueue("agent_1", "read:data")

    assert isinstance(item_id, str)
    assert len(item_id) > 0
    # ID should match a file in the directory
    files = list(tmp_path.iterdir())  # type: ignore[union-attr]
    assert len(files) == 1
    assert files[0].stem == item_id


def test_list_pending_sorted(tmp_path: object) -> None:
    """list_pending returns items sorted by queued_at ascending."""
    queue = LocalQueue(queue_dir=str(tmp_path))

    id1 = queue.enqueue("agent_1", "action_a")
    time.sleep(0.01)
    id2 = queue.enqueue("agent_2", "action_b")
    time.sleep(0.01)
    id3 = queue.enqueue("agent_3", "action_c")

    items = queue.list_pending()
    assert len(items) == 3
    assert items[0]["id"] == id1
    assert items[1]["id"] == id2
    assert items[2]["id"] == id3


def test_count(tmp_path: object) -> None:
    """count returns the number of pending items."""
    queue = LocalQueue(queue_dir=str(tmp_path))
    assert queue.count() == 0

    queue.enqueue("agent_1", "action_a")
    queue.enqueue("agent_2", "action_b")
    assert queue.count() == 2


def test_sync_success(tmp_path: object) -> None:
    """sync deletes all items on success and returns correct counts."""
    queue = LocalQueue(queue_dir=str(tmp_path))
    queue.enqueue("agent_1", "action_a")
    queue.enqueue("agent_2", "action_b")

    with patch("asqav.client._post", return_value={"signature_id": "sig_123"}):
        result = queue.sync()

    assert result["synced"] == 2
    assert result["failed"] == 0
    assert result["total"] == 2
    assert result["errors"] == []
    assert queue.count() == 0


def test_sync_partial_failure(tmp_path: object) -> None:
    """sync keeps failed items in queue and reports errors."""
    call_count = 0

    def mock_post(path: str, data: dict) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise Exception("API unreachable")
        return {"signature_id": f"sig_{call_count}"}

    queue = LocalQueue(queue_dir=str(tmp_path))
    queue.enqueue("agent_1", "action_a")
    queue.enqueue("agent_2", "action_b")

    with patch("asqav.client._post", side_effect=mock_post):
        result = queue.sync()

    assert result["synced"] == 1
    assert result["failed"] == 1
    assert result["total"] == 2
    assert len(result["errors"]) == 1
    assert "API unreachable" in result["errors"][0]["error"]
    # One file should remain
    assert queue.count() == 1


def test_sync_empty_queue(tmp_path: object) -> None:
    """sync with empty queue returns zeroes."""
    queue = LocalQueue(queue_dir=str(tmp_path))

    with patch("asqav.client._post") as mock_post:
        result = queue.sync()

    assert result["synced"] == 0
    assert result["failed"] == 0
    assert result["total"] == 0
    mock_post.assert_not_called()


def test_clear(tmp_path: object) -> None:
    """clear removes all items and returns count deleted."""
    queue = LocalQueue(queue_dir=str(tmp_path))
    queue.enqueue("agent_1", "action_a")
    queue.enqueue("agent_2", "action_b")
    queue.enqueue("agent_3", "action_c")

    deleted = queue.clear()
    assert deleted == 3
    assert queue.count() == 0


def test_local_sign_convenience(tmp_path: object) -> None:
    """local_sign creates a file in the specified queue directory."""
    item_id = local_sign("agent_x", "deploy:model", {"version": "2"}, queue_dir=str(tmp_path))

    assert isinstance(item_id, str)
    filepath = tmp_path / f"{item_id}.json"  # type: ignore[operator]
    assert filepath.exists()

    data = json.loads(filepath.read_text())
    assert data["agent_id"] == "agent_x"
    assert data["action_type"] == "deploy:model"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


from typer.testing import CliRunner

from asqav.cli import app

runner = CliRunner()


@patch("asqav.init")
def test_cli_sync_empty(mock_init: MagicMock, tmp_path: object) -> None:
    """asqav sync with empty queue prints 'nothing to sync'."""
    result = runner.invoke(
        app,
        ["sync"],
        env={"ASQAV_API_KEY": "sk_test", "ASQAV_QUEUE_DIR": str(tmp_path)},
    )
    assert result.exit_code == 0
    assert "nothing to sync" in result.output.lower()


def test_cli_queue_count(tmp_path: object) -> None:
    """asqav queue count shows correct count."""
    # Pre-populate queue
    queue = LocalQueue(queue_dir=str(tmp_path))
    queue.enqueue("agent_1", "action_a")
    queue.enqueue("agent_2", "action_b")

    result = runner.invoke(
        app,
        ["queue", "count"],
        env={"ASQAV_QUEUE_DIR": str(tmp_path)},
    )
    assert result.exit_code == 0
    assert "2" in result.output


def test_cli_queue_list(tmp_path: object) -> None:
    """asqav queue list shows pending items."""
    queue = LocalQueue(queue_dir=str(tmp_path))
    queue.enqueue("agent_abc", "api:call")

    result = runner.invoke(
        app,
        ["queue", "list"],
        env={"ASQAV_QUEUE_DIR": str(tmp_path)},
    )
    assert result.exit_code == 0
    assert "agent_abc" in result.output
    assert "api:call" in result.output
