"""The corpus-B regen check treats a keyed payload_digest as unrecomputable.

asqav-42 carries an HMAC digest (hmac-sha256) under holder key material no
verifier holds, so a plain sha256 recompute would report false drift. With the
keyed marker removed the vector must still drift, or the skip is vacuous.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[2] / "verifier"
VECTOR = VERIFIER / "conformance-vectors" / "asqav-42-keyed-digest-unverifiable"


def _regen():
    spec = importlib.util.spec_from_file_location("regen_cv", VERIFIER / "regen_conformance_vectors.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_keyed_digest_is_unrecomputable_not_drift() -> None:
    drifts, _skips, unrecomputable, _ok = _regen()._audit_vector(VECTOR)
    assert drifts == []
    assert any("payload_digest" in note and "keyed" in note for note in unrecomputable)


def test_same_digest_without_the_keyed_marker_still_drifts(tmp_path: Path) -> None:
    copy = tmp_path / VECTOR.name
    shutil.copytree(VECTOR, copy)
    receipt = json.loads((copy / "receipt.json").read_text())
    del receipt["payload"]["hash_algo"]
    (copy / "receipt.json").write_text(json.dumps(receipt))
    drifts, _skips, _unrecomputable, _ok = _regen()._audit_vector(copy)
    assert [d.member for d in drifts] == ["payload_digest"]


def test_regen_check_passes_on_the_published_corpus() -> None:
    proc = subprocess.run([sys.executable, str(VERIFIER / "regen_conformance_vectors.py"), "--check"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
