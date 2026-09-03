# Copyright 2026 Asqav
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Freeze both conformance corpora at v1 into manifest.lock.json files.

Writes one lock per corpus: every file pinned by SHA-256 and byte length, plus
the published signing seeds and a digest of the lock itself (criterion 420).
CI re-derives every pin by two independent paths and fails on any drift:
python/tests/test_corpus_lock.py (hashlib) and verifier/check_corpus_lock.sh
(sha256sum). Re-running this script after an intentional corpus edit is the
only way the pins move; the freeze then bumps nothing but the pins.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from dilithium_py.ml_dsa import ML_DSA_65

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
LOCK_NAME = "manifest.lock.json"

#: Published Ed25519 seed the ACTA vectors sign with; mirrors gen_acta_vectors.py
ACTA_ED25519_SEED_HEX = (
    "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
)

#: Published ML-DSA-65 seed; the derivation phrase is the nothing-up-my-sleeve pin
MLDSA_SEED_PHRASE = b"asqav conformance corpus v1 ML-DSA-65 signing seed"

#: The fingerprint vector whose SHA-256 digest is the ML-DSA-65 known-answer message
KAT_VECTOR_NAME = "minimal_read"


def _jcs_bytes(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")

    # Pin every file under the corpus root except the lock being written


def _file_pins(root: Path) -> list[dict]:
    pins = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == LOCK_NAME:
            continue
        data = path.read_bytes()
        pins.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    return pins


def _lock_digest(lock: dict) -> str:
    body = {k: v for k, v in lock.items() if k != "digest"}
    return hashlib.sha256(_jcs_bytes(body)).hexdigest()


def _mldsa_signing_section(message_hex: str, signature_hex: str) -> dict:
    seed = hashlib.sha256(MLDSA_SEED_PHRASE).digest()
    pk, _sk = ML_DSA_65.key_derive(seed)
    return {
        "algorithm": "ML-DSA-65",
        "derivation": f"seed = SHA-256 of the ASCII phrase {MLDSA_SEED_PHRASE.decode()!r}",
        "seed_hex": seed.hex(),
        "public_key_hex": pk.hex(),
        "signing": "dilithium-py ML_DSA_65.sign(sk, m, deterministic=True), FIPS 204 pure variant",
        "known_answer": {
            "message_hex": message_hex,
            "signature_hex": signature_hex,
        },
    }


def _rolling_version(root: Path, files: list[dict]) -> tuple[int, list[dict]]:
    """Carry the lock's own history forward and advance the version on a real re-cut.

    The corpus was pinned as frozen at v1 and then re-cut repeatedly while the version
    stayed at 1, so the lock asserted a fixed byte set it no longer named. History holds
    the COMPLETED versions only, each with the digest it published; the current version's
    digest is the lock's own `digest` field, which keeps the record out of its own preimage.
    A cut whose file pins are unchanged keeps its version, so re-running this is idempotent.
    """
    existing = root / LOCK_NAME
    if not existing.exists():
        return 1, []
    prior = json.loads(existing.read_text())
    history = list(prior.get("version_history") or [])
    version = int(prior.get("corpus_version") or 1)
    if prior.get("files") == files:
        return version, history
    history.append(
        {
            "version": version,
            "files": len(prior.get("files") or []),
            "digest": prior.get("digest", ""),
        }
    )
    return version + 1, history


def freeze_fingerprint_corpus() -> str:
    root = _ROOT / "conformance"
    vectors = json.loads((root / "vectors.json").read_text())
    kat = next(v for v in vectors["vectors"] if v["name"] == KAT_VECTOR_NAME)
    message = bytes.fromhex(kat["sha256"])
    seed = hashlib.sha256(MLDSA_SEED_PHRASE).digest()
    _pk, sk = ML_DSA_65.key_derive(seed)
    signature = ML_DSA_65.sign(sk, message, deterministic=True)
    files = _file_pins(root)
    version, history = _rolling_version(root, files)
    lock = {
        "corpus": "asqav-fingerprint-corpus",
        "corpus_version": version,
        "rolling": True,
        "digest_algorithm": "SHA-256",
        "version_history": history,
        "files": files,
        "signing": {
            "note": (
                "The fingerprint corpus pins canonical bytes and their SHA-256; the server "
                "signs that digest with ML-DSA-65. The published seed and the deterministic "
                "signature below give signature-byte parity for the KAT message. Production "
                "signing randomises ML-DSA (FIPS 204 hedged variant), so live signature bytes "
                "are not reproducible; verification is deterministic and that is what audits."
            ),
            "mldsa65": _mldsa_signing_section(
                kat["sha256"],
                signature.hex(),
            ),
            "known_answer_message_note": (
                f"the 32-byte SHA-256 digest of the canonical bytes of vector "
                f"{KAT_VECTOR_NAME!r}: the hash is what ML-DSA-65 signs"
            ),
        },
    }
    lock["digest"] = _lock_digest(lock)
    out = root / LOCK_NAME
    out.write_text(json.dumps(lock, indent=2) + "\n")
    return lock["digest"]


def freeze_verifier_corpus() -> str:
    root = _HERE / "conformance-vectors"
    vectors = json.loads(
        (root.parent.parent / "conformance" / "vectors.json").read_text()
    )
    kat = next(v for v in vectors["vectors"] if v["name"] == KAT_VECTOR_NAME)
    message = bytes.fromhex(kat["sha256"])
    seed = hashlib.sha256(MLDSA_SEED_PHRASE).digest()
    _pk, sk = ML_DSA_65.key_derive(seed)
    signature = ML_DSA_65.sign(sk, message, deterministic=True)
    ed_pub = (
        Ed25519PrivateKey.from_private_bytes(bytes.fromhex(ACTA_ED25519_SEED_HEX))
        .public_key()
        .public_bytes_raw()
        .hex()
    )
    files = _file_pins(root)
    version, history = _rolling_version(root, files)
    lock = {
        "corpus": "asqav-verifier-conformance-vectors",
        "corpus_version": version,
        "rolling": True,
        "digest_algorithm": "SHA-256",
        "version_history": history,
        "files": files,
        "signing": {
            "note": (
                "Locally regenerated vectors sign with the published seeds below, so their "
                "signature bytes reproduce exactly (Ed25519 is deterministic per RFC 8032; "
                "ML-DSA-65 uses the dilithium-py deterministic=True pure variant). Upstream "
                "and prod-signed receipts carry public keys only; their signature bytes are "
                "pinned by the per-file SHA-256 entries above, and verification of them is "
                "deterministic."
            ),
            "ed25519_acta": {
                "algorithm": "Ed25519",
                "generator": "gen_acta_vectors.py",
                "used_by": [
                    "acta-01-genesis",
                    "acta-02-chain-link",
                    "acta-03-tamper-sig",
                    "acta-05-commitment-mode-unsupported",
                ],
                "seed_hex": ACTA_ED25519_SEED_HEX,
                "public_key_hex": ed_pub,
                "signing": "Ed25519 per RFC 8032: deterministic, signature bytes reproduce",
            },
            "mldsa65": {
                **_mldsa_signing_section(kat["sha256"], signature.hex()),
                "used_by": ["asqav-12-time-edge-expiry"],
            },
        },
    }
    lock["digest"] = _lock_digest(lock)
    out = root / LOCK_NAME
    out.write_text(json.dumps(lock, indent=2) + "\n")
    return lock["digest"]


def main() -> None:
    fp = freeze_fingerprint_corpus()
    vv = freeze_verifier_corpus()
    print(f"conformance/{LOCK_NAME} digest: {fp}")
    print(f"verifier/conformance-vectors/{LOCK_NAME} digest: {vv}")


if __name__ == "__main__":
    main()
