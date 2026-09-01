"""The exit-artifact command a customer is told to run, run as written.

The exit manifest ships verify_receipt.py beside the receipts and the JWKS archived
at export, and tells the reader to run one command against them. Every other CLI test
monkeypatches sys.argv and calls main() in-process, which cannot catch a file that
stopped standing on its own or an import that only resolves inside the repo.

This runs the real command in a subprocess, in a directory holding nothing but the
three exit-artifact files, with outbound network blocked, over an anchored ML-DSA-65
receipt. Intact PASSes, tampered FAILs.

The receipt is minted here so the PASS/FAIL halves stay time-independent; this fixture
proves the CLI path, not wire-format conformance, which the corpus vectors carry.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from asqav.verifier import verify_receipt as vr

#: The string the exit manifest emits, from the asqav server's audit_pack export route
#: ("offline_verification"). Kept verbatim so a drift there shows up as a failure here.
EXIT_MANIFEST_COMMAND = (
    "python verify_receipt.py --receipt <receipt.json> --jwks <jwks.json> --offline"
)

VERIFIER_SOURCE = Path(vr.__file__)

#: Refuses every outbound connection in the child, so a verifier that reached for the
#: network instead of the archived JWKS fails loudly rather than passing quietly.
SITECUSTOMIZE = """
import socket

_BLOCKED = OSError("outbound network blocked by test_exit_artifact_cli")


def _refuse(*_a, **_k):
    raise _BLOCKED


class _NoConnect(socket.socket):
    connect = _refuse
    connect_ex = _refuse


socket.socket = _NoConnect
socket.create_connection = _refuse
"""

try:
    from dilithium_py.ml_dsa import ML_DSA_65 as _ML_DSA_65_CHECK  # noqa: F401

    _DILITHIUM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    _DILITHIUM_AVAILABLE = False


    # An anchored ML-DSA-65 receipt plus the JWKS an export would archive. The RFC3161 token
    # is real and the caller pins the TSA key (--tsa-key), since presence alone never PASSes.
def _anchored_mldsa_pair() -> tuple[dict, dict, bytes]:
    import hashlib

    from dilithium_py.ml_dsa import ML_DSA_65

    from tests.tsa_testkit import mint_ml_dsa_anchor

    pk, sk = ML_DSA_65.keygen()
    kid = "exit-artifact-key-01"
    payload = {
        "type": "protectmcp:decision",
        "issued_at": "2026-06-19T00:00:00.000000Z",
        "issuer_id": kid,
        "action_ref": "sha256:" + "a" * 64,
        "payload_digest": {"hash": "b" * 64, "size": 128},
        "policy_digest": "sha256:" + "c" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
        # Far future, so the run does not start failing on its own expiry one day.
        "expires_at": "2099-01-01T00:00:00.000000Z",
    }
    sig = ML_DSA_65.sign(sk, vr.canonical_json(payload))
    envelope = {
        "payload": payload,
        "signature": {"alg": "ML-DSA-65", "kid": kid, "sig": base64.b64encode(sig).decode()},
    }
    bound = hashlib.sha256(vr.envelope_minus_anchors_jcs(envelope)).digest()
    tok, tsa_pk = mint_ml_dsa_anchor(bound)
    envelope["anchors"] = [{"type": "rfc3161", "value": tok}]
    jwks = {
        "keys": [
            {
                "kid": kid,
                "issuer_id": kid,
                "alg": "ML-DSA-65",
                "status": "active",
                "public_key": base64.b64encode(pk).decode(),
            }
        ]
    }
    return envelope, jwks, tsa_pk


    # Write the files an exit artifact carries, and nothing else. The pinned TSA
    # key ships beside them the way the walkthrough's <asqav-tsa-chain.pem> does.
def _lay_out_exit_artifact(root: Path, receipt: dict, jwks: dict, tsa_pk: bytes) -> None:
    shutil.copy(VERIFIER_SOURCE, root / "verify_receipt.py")
    (root / "receipt.json").write_text(json.dumps(receipt))
    (root / "jwks.json").write_text(json.dumps(jwks))
    (root / "tsa-key.b64").write_text(base64.b64encode(tsa_pk).decode())
    (root / "sitecustomize.py").write_text(SITECUSTOMIZE)


def _child_env(root: Path) -> dict[str, str]:
    """PYTHONPATH holds only the artifact dir, so the child cannot reach this checkout.

    It runs the copied file as a customer would, and picks up the sitecustomize that
    refuses the network. The rest of the environment is inherited so the interpreter
    still starts on every runner.
    """
    return {**os.environ, "PYTHONPATH": str(root)}


    # Run the exit-manifest command verbatim, with the network refused.
def _run_documented_command(root: Path, *extra: str) -> subprocess.CompletedProcess:
    argv = EXIT_MANIFEST_COMMAND.replace("<receipt.json>", "receipt.json").replace(
        "<jwks.json>", "jwks.json"
    ).split()
    assert argv[0] == "python" and argv[1] == "verify_receipt.py", argv
    return subprocess.run(
        [sys.executable, *argv[1:], *extra],
        cwd=root,
        capture_output=True,
        text=True,
        env=_child_env(root),
    )


    # Without this control, a passing offline run proves nothing about the network.
def test_the_network_block_actually_blocks(tmp_path: Path) -> None:
    (tmp_path / "sitecustomize.py").write_text(SITECUSTOMIZE)
    probe = "import urllib.request; urllib.request.urlopen('https://api.asqav.com/', timeout=5)"
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=_child_env(tmp_path),
    )
    assert proc.returncode != 0, proc.stdout
    assert "outbound network blocked" in proc.stderr, proc.stderr


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_exit_artifact_command_passes_an_intact_pair(tmp_path: Path) -> None:
    receipt, jwks, tsa_pk = _anchored_mldsa_pair()
    _lay_out_exit_artifact(tmp_path, receipt, jwks, tsa_pk)
    proc = _run_documented_command(tmp_path, "--tsa-key", "tsa-key.b64")
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "=> verified" in proc.stdout, proc.stdout
    # A verified verdict with a skipped signature axis is the failure this test exists for.
    assert "[  ok] signature" in proc.stdout, proc.stdout
    assert "[  ok] anchors" in proc.stdout, proc.stdout
    assert "[  ok] expiry" in proc.stdout, proc.stdout


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_exit_artifact_command_without_tsa_key_never_passes_on_presence(tmp_path: Path) -> None:
    """The documented command alone: anchors report unverifiable, never verified.

    The token's imprint matches (the offline proof the timestamp covers these
    bytes), but without pinned TSA key material the TSA signature cannot be
    trusted, so the draft's no-PASS-on-presence rule caps the verdict at
    unverified/unverifiable. The verified path above adds --tsa-key.
    """
    receipt, jwks, tsa_pk = _anchored_mldsa_pair()
    _lay_out_exit_artifact(tmp_path, receipt, jwks, tsa_pk)
    proc = _run_documented_command(tmp_path)
    assert proc.returncode == 2, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "=> unverified (failure_class: unverifiable" in proc.stdout, proc.stdout
    assert "[skip] anchors" in proc.stdout, proc.stdout
    assert "imprint matches" in proc.stdout, proc.stdout


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_exit_artifact_command_fails_a_tampered_receipt(tmp_path: Path) -> None:
    receipt, jwks, tsa_pk = _anchored_mldsa_pair()
    receipt["payload"]["decision"] = "deny"
    _lay_out_exit_artifact(tmp_path, receipt, jwks, tsa_pk)
    proc = _run_documented_command(tmp_path)
    assert proc.returncode == 1, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "=> unverified (failure_class: invalid)" in proc.stdout, proc.stdout
    assert "[FAIL] signature" in proc.stdout, proc.stdout


    # A swapped verification key must not verify the receipt it did not sign.
@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_exit_artifact_command_fails_a_tampered_jwks(tmp_path: Path) -> None:
    receipt, jwks, tsa_pk = _anchored_mldsa_pair()
    other_jwks = _anchored_mldsa_pair()[1]
    jwks["keys"][0]["public_key"] = other_jwks["keys"][0]["public_key"]
    _lay_out_exit_artifact(tmp_path, receipt, jwks, tsa_pk)
    proc = _run_documented_command(tmp_path)
    assert proc.returncode == 1, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "=> unverified (failure_class: invalid)" in proc.stdout, proc.stdout


    # The exit artifact ships one file, so it must not import the rest of the SDK.
def test_the_verifier_file_stands_alone() -> None:
    source = VERIFIER_SOURCE.read_text()
    assert "SPDX-License-Identifier: Apache-2.0" in source
    intra_package = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and (" asqav" in line or line.startswith("from ."))
    ]
    assert intra_package == [], intra_package
