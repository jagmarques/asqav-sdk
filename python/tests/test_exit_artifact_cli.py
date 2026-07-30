"""The exit-artifact command a customer is told to run, run as written.

The exit manifest ships verify_receipt.py beside the receipts and the JWKS archived
at export, and tells the reader to run one command against them. Every other CLI test
monkeypatches sys.argv and calls main() in-process, which cannot catch a file that
stopped standing on its own or an import that only resolves inside the repo.

This runs the real command in a subprocess, in a directory holding nothing but the
three exit-artifact files, with outbound network blocked, over an anchored ML-DSA-65
receipt. Intact PASSes, tampered FAILs.

The receipt is minted here rather than taken from the corpus: the anchored ML-DSA-65
production vector (asqav-06-mldsa65-payload-prod) has a signed expires_at that lapsed
on 2026-06-20, so it can no longer produce the PASS half. This fixture proves the CLI
path, not wire-format conformance, which the corpus vectors carry.
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


def _anchored_mldsa_pair() -> tuple[dict, dict]:
    """An anchored ML-DSA-65 receipt plus the JWKS an export would archive for it."""
    from dilithium_py.ml_dsa import ML_DSA_65

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
        "anchors": [{"type": "rfc3161", "value": "dGVzdC1hbmNob3I="}],
    }
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
    return envelope, jwks


def _lay_out_exit_artifact(root: Path, receipt: dict, jwks: dict) -> None:
    """Write the three files an exit artifact carries, and nothing else."""
    shutil.copy(VERIFIER_SOURCE, root / "verify_receipt.py")
    (root / "receipt.json").write_text(json.dumps(receipt))
    (root / "jwks.json").write_text(json.dumps(jwks))
    (root / "sitecustomize.py").write_text(SITECUSTOMIZE)


def _child_env(root: Path) -> dict[str, str]:
    """PYTHONPATH holds only the artifact dir, so the child cannot reach this checkout.

    It runs the copied file as a customer would, and picks up the sitecustomize that
    refuses the network. The rest of the environment is inherited so the interpreter
    still starts on every runner.
    """
    return {**os.environ, "PYTHONPATH": str(root)}


def _run_documented_command(root: Path) -> subprocess.CompletedProcess:
    """Run the exit-manifest command verbatim, with the network refused."""
    argv = EXIT_MANIFEST_COMMAND.replace("<receipt.json>", "receipt.json").replace(
        "<jwks.json>", "jwks.json"
    ).split()
    assert argv[0] == "python" and argv[1] == "verify_receipt.py", argv
    return subprocess.run(
        [sys.executable, *argv[1:]],
        cwd=root,
        capture_output=True,
        text=True,
        env=_child_env(root),
    )


def test_the_network_block_actually_blocks(tmp_path: Path) -> None:
    """Without this control, a passing offline run proves nothing about the network."""
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
    receipt, jwks = _anchored_mldsa_pair()
    _lay_out_exit_artifact(tmp_path, receipt, jwks)
    proc = _run_documented_command(tmp_path)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "=> PASS" in proc.stdout, proc.stdout
    # A PASS with a skipped signature axis would be the failure this test exists for.
    assert "[  ok] signature" in proc.stdout, proc.stdout
    assert "[  ok] anchors" in proc.stdout, proc.stdout
    assert "[  ok] expiry" in proc.stdout, proc.stdout


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_exit_artifact_command_fails_a_tampered_receipt(tmp_path: Path) -> None:
    receipt, jwks = _anchored_mldsa_pair()
    receipt["payload"]["decision"] = "deny"
    _lay_out_exit_artifact(tmp_path, receipt, jwks)
    proc = _run_documented_command(tmp_path)
    assert proc.returncode == 1, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "=> FAIL" in proc.stdout, proc.stdout
    assert "[FAIL] signature" in proc.stdout, proc.stdout


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_exit_artifact_command_fails_a_tampered_jwks(tmp_path: Path) -> None:
    """A swapped verification key must not verify the receipt it did not sign."""
    receipt, jwks = _anchored_mldsa_pair()
    _, other_jwks = _anchored_mldsa_pair()
    jwks["keys"][0]["public_key"] = other_jwks["keys"][0]["public_key"]
    _lay_out_exit_artifact(tmp_path, receipt, jwks)
    proc = _run_documented_command(tmp_path)
    assert proc.returncode == 1, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "=> FAIL" in proc.stdout, proc.stdout


def test_the_verifier_file_stands_alone() -> None:
    """The exit artifact ships one file, so it must not import the rest of the SDK."""
    source = VERIFIER_SOURCE.read_text()
    assert "SPDX-License-Identifier: Apache-2.0" in source
    intra_package = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and (" asqav" in line or line.startswith("from ."))
    ]
    assert intra_package == [], intra_package
