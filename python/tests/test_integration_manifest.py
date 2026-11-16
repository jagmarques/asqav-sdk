import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET_PREFIX = "examples/integrations/"
MANIFEST = REPO / TARGET_PREFIX / "MANIFEST.json"
EXCLUDED = (
    "n8n-nodes-asqav",
    "asqav-pydantic",
    "asqav",
    "asqav-sdk",
    "asqav-mcp",
    "asqav-claude-code",
)


def check_entries(entries: list[dict]) -> None:
    seen_repo: set[str] = set()
    seen_target: set[str] = set()
    for entry in entries:
        assert set(entry) == {"repo", "target", "status"}, (
            f"keys must be repo, target, status: {sorted(entry)}"
        )
        assert entry["status"] == "pending", f"status must be 'pending': {entry!r}"
        repo, target = entry["repo"], entry["target"]
        assert target.startswith(TARGET_PREFIX), (
            f"target must start with {TARGET_PREFIX}: {entry!r}"
        )
        assert target.endswith("/"), f"target must end with a slash: {entry!r}"
        assert repo not in EXCLUDED, f"{repo} is an excluded repo"
        assert repo.startswith("asqav-") or repo.endswith("-asqav"), (
            f"{repo} lacks the asqav marker"
        )
        assert repo not in seen_repo, f"duplicate repo {repo}"
        assert target not in seen_target, f"duplicate target {target}"
        seen_repo.add(repo)
        seen_target.add(target)


def test_manifest_entries_valid() -> None:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["entries"]
    assert check_entries(entries) is None
    assert len(entries) > 0


# NEGCTL
def test_entry_without_target_rejected() -> None:
    entries = [{"repo": "asqav-x", "status": "pending"}]
    try:
        check_entries(entries)
    except AssertionError as err:
        assert "target" in str(err)
    else:
        raise AssertionError("an entry without a target was accepted")


# NEGCTL
def test_duplicate_target_rejected() -> None:
    target = f"{TARGET_PREFIX}a/"
    entries = [
        {"repo": "asqav-a", "target": target, "status": "pending"},
        {"repo": "asqav-b", "target": target, "status": "pending"},
    ]
    try:
        check_entries(entries)
    except AssertionError as err:
        assert "duplicate" in str(err)
    else:
        raise AssertionError("two entries with one target were accepted")


# NEGCTL
def test_excluded_repo_rejected() -> None:
    entry = {
        "repo": EXCLUDED[1],
        "target": f"{TARGET_PREFIX}pydantic/",
        "status": "pending",
    }
    try:
        check_entries([entry])
    except AssertionError as err:
        assert "excluded" in str(err)
    else:
        raise AssertionError("an excluded repo was accepted")
