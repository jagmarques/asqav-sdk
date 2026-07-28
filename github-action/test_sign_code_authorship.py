"""Unit tests for the code-authorship recording action.

No network, no real Asqav API. The server call (asqav.client._post) is mocked.
These tests cover the load-bearing pieces of the action:

1. advisory change_digest computation (wire form + diff hashing),
2. the action posts the ADVISORY digest to POST /v1/code-authorship,
3. the written in-toto Statement binds the SERVER subject digest (not the
   advisory one), and digest_match reports advisory-vs-server agreement.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ACTION_DIR = Path(__file__).resolve().parent
SDK_SRC = ACTION_DIR.parent / "python" / "src"
sys.path.insert(0, str(ACTION_DIR))
sys.path.insert(0, str(SDK_SRC))

# A globally installed `asqav` (or an editable install pointing at another
# checkout) can shadow the in-repo SDK. Purge any cached import so the module
# under test resolves to this worktree's python/src.
for _name in [m for m in list(sys.modules) if m == "asqav" or m.startswith("asqav.")]:
    del sys.modules[_name]

import sign_code_authorship as mod  # noqa: E402

from asqav.code_authorship import CODE_AUTHORSHIP_PATH  # noqa: E402

SERVER_HEX = "f" * 64
ADVISORY_HEX = "1" * 64


# --- advisory change_digest computation --------------------------------------


def test_change_digest_hashes_supplied_diff_text():
    diff = "diff --git a/x b/x\n+hello\n"
    expected = "sha256:" + hashlib.sha256(diff.encode("utf-8")).hexdigest()
    assert mod.compute_change_digest("base", "head", diff_text=diff) == expected


def test_change_digest_is_sha256_wire_form():
    digest = mod.compute_change_digest("base", "head", diff_text="anything")
    assert digest.startswith("sha256:")
    hex_part = digest.split(":", 1)[1]
    assert len(hex_part) == 64
    assert all(c in "0123456789abcdef" for c in hex_part)


def test_change_digest_falls_back_to_head_sha_without_base():
    head = "a" * 40
    expected = "sha256:" + hashlib.sha256(head.encode("utf-8")).hexdigest()
    assert mod.compute_change_digest(None, head, diff_text=None) == expected


# --- server response builders -------------------------------------------------


def _server_envelope(*, server_hex: str, digest_match: bool, advisory: str) -> dict:
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {"name": "owner/repo@" + "c" * 40, "digest": {"sha256": server_hex}}
        ],
        "predicateType": "https://asqav.com/attestation/code-authorship/v1",
        "predicate": {
            "capture_layer": "github_sha_pull",
            "asset_class": "code",
            "advisory_client_digest": advisory,
            "digest_match": digest_match,
        },
    }


def _server_response(*, server_hex: str, digest_match: bool, advisory: str) -> dict:
    return {
        "envelope": _server_envelope(
            server_hex=server_hex, digest_match=digest_match, advisory=advisory
        ),
        "receipt": {"signature_id": "sig_ca", "algorithm": "ML-DSA-65"},
        "kid": "key_ca",
        "jwks_url": "https://api.asqav.com/.well-known/jwks.json",
        "server_digest": "sha256:" + server_hex,
        "digest_match": digest_match,
    }


# --- end-to-end with a mocked server -----------------------------------------


def test_sign_and_export_posts_advisory_and_binds_server_subject(monkeypatch, tmp_path):
    captured: dict = {}
    advisory = "sha256:" + ADVISORY_HEX

    def fake_post(path: str, body: dict) -> dict:
        captured["path"] = path
        captured["body"] = body
        # The server recomputes a DIFFERENT digest than the advisory one.
        return _server_response(
            server_hex=SERVER_HEX, digest_match=False, advisory=advisory
        )

    monkeypatch.setattr("asqav.client._post", fake_post)

    ctx = mod.GitContext(
        repo_ref="owner/repo",
        commit_sha="c" * 40,
        base_sha="b" * 40,
        change_ref="https://github.com/owner/repo/pull/7",
        change_digest=advisory,
    )
    intoto_path = tmp_path / "out.intoto.jsonl"

    outputs = mod.sign_and_export(
        api_key="sk_test",
        change_class="write",
        author="model:claude-opus-4-8",
        intoto_path=str(intoto_path),
        git_ctx=ctx,
    )

    # The action posts to /v1/code-authorship with the ADVISORY digest.
    assert captured["path"] == CODE_AUTHORSHIP_PATH
    body = captured["body"]
    assert body["repo"] == "owner/repo"
    assert body["commit_sha"] == "c" * 40
    assert body["base_sha"] == "b" * 40
    assert body["change_digest"] == advisory
    assert body["change_class"] == "write"
    assert body["author"] == "model:claude-opus-4-8"
    assert body["anchor"] == "https://github.com/owner/repo/pull/7"

    # The written Statement binds the SERVER subject digest, not the advisory one.
    stmt = json.loads(intoto_path.read_text())
    assert stmt["_type"] == "https://in-toto.io/Statement/v1"
    assert stmt["predicateType"] == "https://asqav.com/attestation/code-authorship/v1"
    assert stmt["subject"][0]["digest"]["sha256"] == SERVER_HEX
    assert stmt["subject"][0]["digest"]["sha256"] != ADVISORY_HEX
    assert stmt["predicate"]["capture_layer"] == "github_sha_pull"
    assert stmt["predicate"]["advisory_client_digest"] == advisory

    # digest_match reports the advisory-vs-server disagreement.
    assert outputs["digest-match"] == "false"
    assert outputs["server-digest"] == "sha256:" + SERVER_HEX
    assert outputs["capture-layer"] == "github_sha_pull"
    assert outputs["signature-id"] == "sig_ca"
    assert outputs["intoto-statement-path"] == str(intoto_path)


def test_sign_and_export_reports_digest_match_true(monkeypatch, tmp_path):
    advisory = "sha256:" + ADVISORY_HEX

    def fake_post(path: str, body: dict) -> dict:
        # Server recomputation agrees with the advisory digest.
        return _server_response(
            server_hex=ADVISORY_HEX, digest_match=True, advisory=advisory
        )

    monkeypatch.setattr("asqav.client._post", fake_post)

    ctx = mod.GitContext(
        repo_ref="owner/repo",
        commit_sha="c" * 40,
        base_sha="b" * 40,
        change_ref=None,
        change_digest=advisory,
    )
    intoto_path = tmp_path / "out.intoto.jsonl"

    outputs = mod.sign_and_export(
        api_key="sk_test",
        change_class="write",
        author=None,
        intoto_path=str(intoto_path),
        git_ctx=ctx,
    )

    assert outputs["digest-match"] == "true"
    assert outputs["server-digest"] == "sha256:" + ADVISORY_HEX
    stmt = json.loads(intoto_path.read_text())
    assert stmt["subject"][0]["digest"]["sha256"] == ADVISORY_HEX
    assert stmt["predicate"]["digest_match"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
