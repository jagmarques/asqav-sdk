"""
asqav local mode - Queue actions offline and sync when connectivity returns.

Enables developers to use asqav in offline or air-gapped environments.
Actions are queued locally as JSON files and synced to the API later.

Usage:
    from asqav import LocalQueue, local_sign

    # Queue an action locally
    item_id = local_sign("agent_123", "api:call", {"model": "gpt-4"})

    # Sync when ready
    queue = LocalQueue()
    result = queue.sync()
    print(f"Synced {result['synced']}/{result['total']} items")
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


_DEFAULT_QUEUE_DIR = os.path.join(os.path.expanduser("~"), ".asqav", "queue")


class LocalQueue:
    """Queue for offline action signing with later sync to asqav API."""

    def __init__(self, queue_dir: str | None = None) -> None:
        resolved = (
            queue_dir
            or os.environ.get("ASQAV_QUEUE_DIR")
            or _DEFAULT_QUEUE_DIR
        )
        self.queue_dir = Path(resolved)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        agent_id: str,
        action_type: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Queue an action for later sync. Returns the queue item ID."""
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%d%H%M%S")
        short_id = uuid4().hex[:8]
        item_id = f"{timestamp}_{short_id}"

        payload = {
            "agent_id": agent_id,
            "action_type": action_type,
            "context": context or {},
            "queued_at": now.isoformat(),
            "status": "pending",
        }

        filepath = self.queue_dir / f"{item_id}.json"
        filepath.write_text(json.dumps(payload, indent=2))
        return item_id

    def list_pending(self) -> list[dict[str, Any]]:
        """List all pending queue items, sorted oldest first."""
        items: list[dict[str, Any]] = []
        for filepath in self.queue_dir.glob("*.json"):
            try:
                data = json.loads(filepath.read_text())
                data["id"] = filepath.stem
                items.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        items.sort(key=lambda x: x.get("queued_at", ""))
        return items

    def sync(
        self, on_progress: Callable[[str, bool, str | None], Any] | None = None
    ) -> dict[str, Any]:
        """Sync pending items to API. Returns summary with synced/failed counts."""
        from .client import _post

        items = self.list_pending()
        synced = 0
        failed = 0
        errors: list[dict[str, str]] = []

        for item in items:
            item_id = item["id"]
            agent_id = item["agent_id"]
            action_type = item["action_type"]
            context = item.get("context", {})

            try:
                _post(
                    f"/agents/{agent_id}/sign",
                    {"action_type": action_type, "context": context},
                )
                # Success - remove from queue
                filepath = self.queue_dir / f"{item_id}.json"
                filepath.unlink(missing_ok=True)
                synced += 1
                if on_progress:
                    on_progress(item_id, True, None)
            except Exception as exc:
                failed += 1
                error_msg = str(exc)
                errors.append({"id": item_id, "error": error_msg})
                if on_progress:
                    on_progress(item_id, False, error_msg)

        return {
            "synced": synced,
            "failed": failed,
            "total": synced + failed,
            "errors": errors,
        }

    def count(self) -> int:
        """Return number of pending items in the queue."""
        return len(list(self.queue_dir.glob("*.json")))

    def clear(self) -> int:
        """Delete all pending items. Returns number deleted."""
        files = list(self.queue_dir.glob("*.json"))
        for filepath in files:
            filepath.unlink(missing_ok=True)
        return len(files)


def local_sign(
    agent_id: str,
    action_type: str,
    context: dict[str, Any] | None = None,
    queue_dir: str | None = None,
) -> str:
    """Convenience function to queue a single action locally.

    Returns the queue item ID.
    """
    queue = LocalQueue(queue_dir=queue_dir)
    return queue.enqueue(agent_id, action_type, context)
