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


def manifest_entries() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["entries"]


def test_manifest_entries_valid() -> None:
    entries = manifest_entries()
    assert check_entries(entries) is None
    assert len(entries) > 0


# NEGCTL: an entry with no target key is refused, naming the key.
def test_entry_without_target_rejected() -> None:
    entry = dict(manifest_entries()[0])
    del entry["target"]
    try:
        check_entries([entry])
    except AssertionError as err:
        assert "target" in str(err)
    else:
        raise AssertionError("an entry without a target was accepted")


# NEGCTL: two entries naming one target folder are refused as a duplicate.
def test_duplicate_target_rejected() -> None:
    first, second = manifest_entries()[:2]
    second = dict(second, target=first["target"])
    try:
        check_entries([first, second])
    except AssertionError as err:
        assert "duplicate" in str(err)
    else:
        raise AssertionError("two entries with one target were accepted")


# NEGCTL: an excluded repo name is refused even when its entry is well formed.
def test_excluded_repo_rejected() -> None:
    entry = dict(manifest_entries()[0], repo=EXCLUDED[1])
    try:
        check_entries([entry])
    except AssertionError as err:
        assert "excluded" in str(err)
    else:
        raise AssertionError("an excluded repo was accepted")
