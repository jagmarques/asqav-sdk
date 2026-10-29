"""Conformance corpus freeze at v1, verification path A (criterion 420).

Every corpus file is pinned by SHA-256 and length in manifest.lock.json, whose
digest is pinned here and in each README; sha256sum re-derives it too. This
module also owns the README vector index format (criterion 821).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
FINGERPRINT_ROOT = _REPO_ROOT / "conformance"
VERIFIER_ROOT = _REPO_ROOT / "verifier" / "conformance-vectors"
FINGERPRINT_LOCK = FINGERPRINT_ROOT / "manifest.lock.json"
VERIFIER_LOCK = VERIFIER_ROOT / "manifest.lock.json"

# The lock digests, as in each lock's digest field and the corpus READMEs. A
# regenerated lock that forgets to update these is drift, not a refresh.
FINGERPRINT_LOCK_DIGEST = "511989671a21de1f8be82ab1c53a6e8f5c568e5c17abec25faeca0169b623877"
VERIFIER_LOCK_DIGEST = "592b79324b596277a33ca3b09ef4d3b5144f9c44da67021aaf047fc07096c656"

# The em dash as an escape, so the guard does not carry the character it bans.
EM_DASH = "\u2014"

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    _CRYPTOGRAPHY_AVAILABLE = True
except ImportError:  # pragma: no cover - installed by every CI extra
    _CRYPTOGRAPHY_AVAILABLE = False

try:
    from dilithium_py.ml_dsa import ML_DSA_65

    _DILITHIUM_AVAILABLE = True
except ImportError:  # pragma: no cover - installed by every CI extra
    _DILITHIUM_AVAILABLE = False


def _jcs_bytes(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _load_lock(path: Path) -> dict:
    assert path.exists(), f"missing lock file: {path}"
    return json.loads(path.read_text())


def _locks() -> list[tuple[Path, Path, str]]:
    return [
        (FINGERPRINT_ROOT, FINGERPRINT_LOCK, FINGERPRINT_LOCK_DIGEST),
        (VERIFIER_ROOT, VERIFIER_LOCK, VERIFIER_LOCK_DIGEST),
    ]

    # The rolling lock is the contract: a version that moves with the bytes.


@pytest.mark.parametrize("root,lock_path,pinned", _locks(), ids=["fingerprint", "verifier"])
def test_lock_declares_a_rolling_version(root, lock_path, pinned) -> None:
    lock = _load_lock(lock_path)
    assert lock["rolling"] is True, lock_path
    assert isinstance(lock["corpus_version"], int), lock_path
    assert lock["corpus_version"] >= 1, lock_path
    assert lock["digest_algorithm"] == "SHA-256", lock_path
    assert lock["corpus"], lock_path


# A version that never moved while the bytes did is the defect this pins against
@pytest.mark.parametrize("root,lock_path,pinned", _locks(), ids=["fingerprint", "verifier"])
def test_history_records_every_completed_cut(root, lock_path, pinned) -> None:
    lock = _load_lock(lock_path)
    history = lock["version_history"]
    assert len(history) == lock["corpus_version"] - 1, lock_path
    assert [e["version"] for e in history] == list(range(1, lock["corpus_version"])), lock_path
    for entry in history:
        assert len(entry["digest"]) == 64, lock_path
        assert entry["files"] >= 1, lock_path
    # The current cut's digest is the lock's own, never inside its own preimage
    assert all(e["digest"] != lock["digest"] for e in history), lock_path


@pytest.mark.parametrize("root,lock_path,pinned", _locks(), ids=["fingerprint", "verifier"])
def test_every_file_pin_rederives(root, lock_path, pinned) -> None:
    lock = _load_lock(lock_path)
    for entry in lock["files"]:
        data = (root / entry["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], (
            f"{entry['path']}: sha256 drift - regenerate with verifier/freeze_corpus_lock.py"
        )
        assert len(data) == entry["bytes"], (
            f"{entry['path']}: byte-length drift - regenerate with verifier/freeze_corpus_lock.py"
        )


@pytest.mark.parametrize("root,lock_path,pinned", _locks(), ids=["fingerprint", "verifier"])
def test_no_file_escapes_the_lock(root, lock_path, pinned) -> None:
    lock = _load_lock(lock_path)
    pinned_paths = {e["path"] for e in lock["files"]}
    on_disk = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != "manifest.lock.json"
    }
    assert on_disk == pinned_paths, (
        f"corpus drift: unpinned {sorted(on_disk - pinned_paths)}, "
        f"stale {sorted(pinned_paths - on_disk)}"
    )


@pytest.mark.parametrize("root,lock_path,pinned", _locks(), ids=["fingerprint", "verifier"])
def test_lock_digest_rederives_and_matches_the_repo_pin(root, lock_path, pinned) -> None:
    lock = _load_lock(lock_path)
    body = {k: v for k, v in lock.items() if k != "digest"}
    assert hashlib.sha256(_jcs_bytes(body)).hexdigest() == lock["digest"], lock_path
    assert lock["digest"] == pinned, (
        f"{lock_path}: digest changed; update the README and the test constant together"
    )

    # The published Ed25519 seed regenerates the ACTA vectors' public key


@pytest.mark.skipif(not _CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed")
def test_ed25519_corpus_seed_rederives_the_public_key() -> None:
    lock = _load_lock(VERIFIER_LOCK)
    acta = lock["signing"]["ed25519_acta"]
    pub = (
        Ed25519PrivateKey.from_private_bytes(bytes.fromhex(acta["seed_hex"]))
        .public_key()
        .public_bytes_raw()
        .hex()
    )
    assert pub == acta["public_key_hex"], "published Ed25519 seed does not match its key"

    # The published ML-DSA-65 seed regenerates the key and the pinned KAT signature
    # byte for byte (dilithium-py deterministic=True, the FIPS 204 pure variant)


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
@pytest.mark.parametrize(
    "lock_path", [FINGERPRINT_LOCK, VERIFIER_LOCK], ids=["fingerprint", "verifier"]
)
def test_mldsa_corpus_seed_gives_signature_byte_parity(lock_path) -> None:
    lock = _load_lock(lock_path)
    mldsa = lock["signing"]["mldsa65"]
    seed = bytes.fromhex(mldsa["seed_hex"])
    pk, sk = ML_DSA_65.key_derive(seed)
    assert pk.hex() == mldsa["public_key_hex"], "published ML-DSA seed does not match its key"
    kat = mldsa["known_answer"]
    message = bytes.fromhex(kat["message_hex"])
    signature = ML_DSA_65.sign(sk, message, deterministic=True)
    assert signature.hex() == kat["signature_hex"], (
        "ML-DSA-65 signature bytes drifted from the pinned KAT: deterministic signing "
        "must reproduce them exactly"
    )
    assert ML_DSA_65.verify(pk, message, signature) is True

    # Both locks publish the same ML-DSA-65 corpus seed


def test_the_mldsa_seed_is_one_published_value() -> None:
    fp = _load_lock(FINGERPRINT_LOCK)["signing"]["mldsa65"]
    vv = _load_lock(VERIFIER_LOCK)["signing"]["mldsa65"]
    assert fp["seed_hex"] == vv["seed_hex"]
    assert fp["public_key_hex"] == vv["public_key_hex"]
    assert fp["known_answer"] == vv["known_answer"]

    # The KAT message is the pinned SHA-256 of the named fingerprint vector


def test_kat_message_is_the_fingerprint_vector_digest() -> None:
    lock = _load_lock(FINGERPRINT_LOCK)
    message_hex = lock["signing"]["mldsa65"]["known_answer"]["message_hex"]
    vectors = json.loads((FINGERPRINT_ROOT / "vectors.json").read_text())
    kat = next(v for v in vectors["vectors"] if v["name"] == "minimal_read")
    assert message_hex == kat["sha256"]
    assert hashlib.sha256(kat["canonical"].encode("utf-8")).hexdigest() == kat["sha256"]

    # The walkthrough fixtures are byte copies of locked corpus files, so the
    # auditor-facing artifacts cannot drift from the corpus pins


def test_walkthrough_fixtures_are_locked_corpus_bytes() -> None:
    fixtures = _REPO_ROOT / "verifier" / "docs" / "fixtures"
    pairs = {
        "published-receipt.json": VERIFIER_ROOT / "asqav-06-mldsa65-payload-prod" / "receipt.json",
        "published-jwks.json": VERIFIER_ROOT / "asqav-06-mldsa65-payload-prod" / "jwks.json",
        "ed25519-receipt.json": VERIFIER_ROOT / "asqav-01-genesis-permit" / "receipt.json",
        "ed25519-jwks.json": VERIFIER_ROOT / "asqav-01-genesis-permit" / "jwks.json",
    }
    for fixture, source in pairs.items():
        assert (fixtures / fixture).exists(), f"missing walkthrough fixture: {fixture}"
        assert (fixtures / fixture).read_bytes() == source.read_bytes(), (
            f"verifier/docs/fixtures/{fixture} drifted from {source}"
        )

    # The generated vector index in the verifier corpus README (criterion 821).
    # The renderer owns the format and the lock owns the bytes.


def _renderer():
    """The index renderer, loaded from the path the corpus lock pins."""
    spec = importlib.util.spec_from_file_location(
        "render_vector_list", _REPO_ROOT / "verifier" / "render_vector_list.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vector_manifest() -> list[dict]:
    return json.loads((VERIFIER_ROOT / "manifest.json").read_text(encoding="utf-8"))


def _readme_bytes() -> bytes:
    return (VERIFIER_ROOT / "README.md").read_bytes()


def test_vector_list_lines_use_colon() -> None:
    entries = _vector_manifest()
    lines = _renderer()._render(entries)
    assert len(lines) == len(entries)
    for entry, line in zip(entries, lines, strict=True):
        assert line.startswith(f"- `{entry['dir']}`: "), line
        # One code span per line: a directory name that carries a backtick
        # would break the span, and the index must not render it.
        assert line.count("`") == 2, line
        assert EM_DASH not in line, line


def test_render_vector_list_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = _renderer()
    target = tmp_path / "README.md"
    target.write_bytes(_readme_bytes())
    monkeypatch.setattr(renderer, "README", target)
    assert renderer.main() == 0
    first = target.read_bytes()
    assert renderer.main() == 0
    assert target.read_bytes() == first
    assert first == _readme_bytes(), "the committed README is not the renderer's output"


def test_readme_has_no_em_dash() -> None:
    assert EM_DASH not in (VERIFIER_ROOT / "README.md").read_text(encoding="utf-8")


def test_lock_rejects_em_dash_readme(tmp_path: Path) -> None:
    # The negative control runs the real lock checker over a copy that carries
    # an em dash, then re-freezes that copy and checks it a second time.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copytree(_REPO_ROOT / "verifier", corpus / "verifier")
    shutil.copytree(_REPO_ROOT / "conformance", corpus / "conformance")
    readme = corpus / "verifier" / "conformance-vectors" / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + " " + EM_DASH, encoding="utf-8")
    check = corpus / "verifier" / "check_corpus_lock.sh"
    drifted = subprocess.run(["bash", str(check)], capture_output=True, text=True)
    assert drifted.returncode == 1, drifted.stdout + drifted.stderr
    assert "README.md" in drifted.stdout + drifted.stderr
    frozen = subprocess.run(
        [sys.executable, str(corpus / "verifier" / "freeze_corpus_lock.py")],
        capture_output=True,
        text=True,
    )
    assert frozen.returncode == 0, frozen.stdout + frozen.stderr
    refrozen = subprocess.run(["bash", str(check)], capture_output=True, text=True)
    assert refrozen.returncode == 0, refrozen.stdout + refrozen.stderr

