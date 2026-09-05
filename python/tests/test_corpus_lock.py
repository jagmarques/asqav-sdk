"""Conformance corpus freeze at v1, verification path A (criterion 420).

Every vector file in both corpora is pinned by SHA-256 and byte length in a
manifest.lock.json; the lock's own digest is pinned here and in each corpus
README. This module re-derives every pin with hashlib + pathlib and fails loud
on any drift. The second independent path re-derives the same pins through the
sha256sum binary: verifier/check_corpus_lock.sh (CI runs both).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
FINGERPRINT_ROOT = _REPO_ROOT / "conformance"
VERIFIER_ROOT = _REPO_ROOT / "verifier" / "conformance-vectors"
FINGERPRINT_LOCK = FINGERPRINT_ROOT / "manifest.lock.json"
VERIFIER_LOCK = VERIFIER_ROOT / "manifest.lock.json"

# The lock digests, reproduced beside each lock's own digest field and in the corpus
# READMEs. A regenerated lock that forgets to update these is drift, not a refresh.
FINGERPRINT_LOCK_DIGEST = "1ef6d34d3f9515f2442218e8bf1e1370dd3b0cf1cb1dd143e121b0ff8384732f"
VERIFIER_LOCK_DIGEST = "40e4fd334f5fe8bc835498f95a8b29659f65f14a23dc233c9e967aabe6b556d3"

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

    # The rolling lock is the contract: a version that moves with the bytes, and named


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
    # The current cut's digest is the lock's own, so it is never inside its own preimage
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
