"""SDK half of the authoritative code-authorship path (POST /v1/code-authorship).

The client supplies an ADVISORY change digest. The server re-fetches the commit,
recomputes the canonical diff, and signs an in-toto Statement whose
``subject[0].digest.sha256`` is the SERVER digest. These tests pin: the advisory
digest computation, the wire body the submit helper posts, the authoritative
envelope parsing, and the capture-layer observation-decision rule
(``github_sha_pull`` is authoritative; ``in_process_sdk`` / ``passive_telemetry``
are observation only and never authoritative).
"""

from __future__ import annotations

import hashlib
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav.code_authorship import (
    AUTHORITATIVE_CAPTURE_LAYER,
    CODE_AUTHORSHIP_ASSET_CLASS,
    CODE_AUTHORSHIP_PATH,
    CODE_AUTHORSHIP_PREDICATE_TYPE,
    CODE_AUTHORSHIP_WRITE_SCOPE,
    INTOTO_STATEMENT_TYPE,
    OBSERVATION_ONLY_CAPTURE_LAYERS,
    CodeAuthorshipResult,
    compute_advisory_digest,
    submit_code_authorship,
    verify_code_authorship_envelope,
)


# === constants + scope ===


def test_scope_constant_matches_backend_contract() -> None:
    """The required API-key scope is code_authorship:write."""
    assert CODE_AUTHORSHIP_WRITE_SCOPE == "code_authorship:write"


def test_scope_constant_reexported_on_cli_hook() -> None:
    """cli_hook exposes the same scope as REQUIRED_SCOPE."""
    from asqav import cli_hook

    assert cli_hook.REQUIRED_SCOPE == CODE_AUTHORSHIP_WRITE_SCOPE


def test_predicate_type_and_asset_class_match_contract() -> None:
    assert CODE_AUTHORSHIP_PREDICATE_TYPE == "https://asqav.com/attestation/code-authorship/v1"
    assert CODE_AUTHORSHIP_ASSET_CLASS == "code"
    assert AUTHORITATIVE_CAPTURE_LAYER == "github_sha_pull"
    assert OBSERVATION_ONLY_CAPTURE_LAYERS == frozenset({"in_process_sdk", "passive_telemetry"})


# === advisory digest computation ===


def test_advisory_digest_hashes_supplied_diff_text() -> None:
    diff = "diff --git a/x b/x\n+hello\n"
    expected = "sha256:" + hashlib.sha256(diff.encode("utf-8")).hexdigest()
    assert compute_advisory_digest("base", "head", diff_text=diff) == expected


def test_advisory_digest_is_sha256_wire_form() -> None:
    digest = compute_advisory_digest("base", "head", diff_text="anything")
    assert digest.startswith("sha256:")
    hex_part = digest.split(":", 1)[1]
    assert len(hex_part) == 64
    assert all(c in "0123456789abcdef" for c in hex_part)


def test_advisory_digest_falls_back_to_head_sha_without_base() -> None:
    head = "a" * 40
    expected = "sha256:" + hashlib.sha256(head.encode("utf-8")).hexdigest()
    assert compute_advisory_digest(None, head) == expected


# === envelope builders ===


def _server_envelope(
    *,
    server_digest_hex: str,
    capture_layer: str,
    advisory: str = "sha256:" + "1" * 64,
    digest_match: bool = True,
    predicate_type: str = CODE_AUTHORSHIP_PREDICATE_TYPE,
    statement_type: str = INTOTO_STATEMENT_TYPE,
    include_subject_digest: bool = True,
) -> dict:
    subject: list = []
    if include_subject_digest:
        subject = [{"name": "owner/repo@" + "c" * 40, "digest": {"sha256": server_digest_hex}}]
    return {
        "_type": statement_type,
        "subject": subject,
        "predicateType": predicate_type,
        "predicate": {
            "capture_layer": capture_layer,
            "asset_class": CODE_AUTHORSHIP_ASSET_CLASS,
            "advisory_client_digest": advisory,
            "digest_match": digest_match,
        },
    }


def _server_response(
    *,
    server_digest_hex: str,
    capture_layer: str = "github_sha_pull",
    digest_match: bool = True,
) -> dict:
    return {
        "envelope": _server_envelope(server_digest_hex=server_digest_hex, capture_layer=capture_layer),
        "receipt": {"signature_id": "sig_ca", "algorithm": "ML-DSA-65"},
        "kid": "key_ca",
        "jwks_url": "https://api.asqav.com/.well-known/jwks.json",
        "server_digest": "sha256:" + server_digest_hex,
        "digest_match": digest_match,
    }


# === submit helper ===


def test_submit_posts_advisory_digest_to_code_authorship_endpoint() -> None:
    captured: dict = {}
    server_hex = "f" * 64

    def fake_post(path: str, body: dict) -> dict:
        captured["path"] = path
        captured["body"] = body
        return _server_response(server_digest_hex=server_hex)

    with patch("asqav.client._post", side_effect=fake_post):
        result = submit_code_authorship(
            repo="owner/repo",
            commit_sha="c" * 40,
            base_sha="b" * 40,
            change_digest="sha256:" + "1" * 64,
            change_class="write",
            author="human:alice@example.com",
            anchor="https://github.com/owner/repo/pull/7",
        )

    assert captured["path"] == CODE_AUTHORSHIP_PATH
    body = captured["body"]
    assert body["repo"] == "owner/repo"
    assert body["commit_sha"] == "c" * 40
    assert body["base_sha"] == "b" * 40
    # The client digest is advisory. It travels so the server can report a match.
    assert body["change_digest"] == "sha256:" + "1" * 64
    assert body["change_class"] == "write"
    assert body["author"] == "human:alice@example.com"
    assert body["anchor"] == "https://github.com/owner/repo/pull/7"

    # The authoritative subject digest is the SERVER value, not the client's.
    assert result.subject_digest == server_hex
    assert result.server_digest == "sha256:" + server_hex
    assert result.capture_layer == "github_sha_pull"
    assert result.kid == "key_ca"
    assert result.subject_matches_server is True


def test_submit_omits_absent_optional_fields() -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _server_response(server_digest_hex="e" * 64)

    with patch("asqav.client._post", side_effect=fake_post):
        submit_code_authorship(repo="owner/repo", commit_sha="c" * 40)

    body = captured["body"]
    assert body == {"repo": "owner/repo", "commit_sha": "c" * 40}


def test_result_exposes_digest_match_semantics() -> None:
    """digest_match reports advisory-vs-server agreement. The subject is the server's."""
    advisory = "sha256:" + "1" * 64
    server_hex = "2" * 64
    envelope = _server_envelope(
        server_digest_hex=server_hex,
        capture_layer="github_sha_pull",
        advisory=advisory,
        digest_match=False,
    )
    result = CodeAuthorshipResult.from_response(
        {
            "envelope": envelope,
            "receipt": {},
            "kid": "k",
            "jwks_url": "u",
            "server_digest": "sha256:" + server_hex,
            "digest_match": False,
        }
    )
    # Advisory digest disagreed, but the bound subject is still the server digest.
    assert result.digest_match is False
    assert result.advisory_client_digest == advisory
    assert result.subject_digest == server_hex
    assert result.subject_matches_server is True


# === capture-layer observation-decision rule ===


def test_github_sha_pull_envelope_is_authoritative_and_passes() -> None:
    envelope = _server_envelope(server_digest_hex="a" * 64, capture_layer="github_sha_pull")
    verification = verify_code_authorship_envelope(envelope)
    assert verification.passed
    assert verification.verdict == "PASS"
    assert verification.authoritative is True
    assert verification.observation_only is False
    assert verification.capture_layer == "github_sha_pull"
    assert verification.subject_digest == "a" * 64


@pytest.mark.parametrize("layer", sorted(OBSERVATION_ONLY_CAPTURE_LAYERS))
def test_observation_capture_layer_is_never_authoritative(layer: str) -> None:
    """in_process_sdk and passive_telemetry are observation only, never a decision."""
    envelope = _server_envelope(server_digest_hex="a" * 64, capture_layer=layer)
    verification = verify_code_authorship_envelope(envelope)
    assert verification.passed is False
    assert verification.verdict == "REJECT"
    assert verification.authoritative is False
    assert verification.observation_only is True
    assert "observation_capture_layer_not_authoritative" in verification.reasons


def test_missing_subject_digest_rejected() -> None:
    envelope = _server_envelope(
        server_digest_hex="a" * 64,
        capture_layer="github_sha_pull",
        include_subject_digest=False,
    )
    verification = verify_code_authorship_envelope(envelope)
    assert verification.passed is False
    assert "code_authorship_missing_subject_digest" in verification.reasons


def test_wrong_predicate_type_rejected() -> None:
    envelope = _server_envelope(
        server_digest_hex="a" * 64,
        capture_layer="github_sha_pull",
        predicate_type="https://example.com/wrong/v1",
    )
    verification = verify_code_authorship_envelope(envelope)
    assert verification.passed is False
    assert "code_authorship_wrong_predicate_type" in verification.reasons


def test_unknown_capture_layer_not_authoritative() -> None:
    envelope = _server_envelope(server_digest_hex="a" * 64, capture_layer="network_proxy")
    verification = verify_code_authorship_envelope(envelope)
    assert verification.passed is False
    assert verification.authoritative is False
    assert verification.observation_only is False
    assert "code_authorship_capture_layer_not_github_sha_pull" in verification.reasons


def test_non_object_envelope_rejected_cleanly() -> None:
    verification = verify_code_authorship_envelope(None)  # type: ignore[arg-type]
    assert verification.passed is False
    assert "envelope_not_an_object" in verification.reasons


# === public surface ===


def test_public_surface_reexported() -> None:
    assert asqav.CODE_AUTHORSHIP_WRITE_SCOPE == CODE_AUTHORSHIP_WRITE_SCOPE
    assert asqav.submit_code_authorship is submit_code_authorship
    assert asqav.verify_code_authorship_envelope is verify_code_authorship_envelope
    assert asqav.compute_advisory_digest is compute_advisory_digest


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
