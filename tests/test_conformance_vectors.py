"""Verify conformance vectors are internally consistent.

Any third-party implementation should be able to reproduce the canonical bytes
and SHA-256 for each input. If this test fails, vectors.json was edited without
regenerating the derived fields.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

VECTORS_PATH = Path(__file__).parent.parent / "conformance" / "vectors.json"


def _jcs(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def test_vectors_file_exists() -> None:
    assert VECTORS_PATH.exists(), f"missing {VECTORS_PATH}"


def test_vectors_schema_metadata() -> None:
    d = json.loads(VECTORS_PATH.read_text())
    assert d["version"] == 1
    assert d["canonicalization"] == "RFC 8785 JCS"
    assert d["hash_algorithm"] == "SHA-256"
    assert isinstance(d["vectors"], list)
    assert len(d["vectors"]) >= 3


def test_each_vector_canonical_matches_input() -> None:
    d = json.loads(VECTORS_PATH.read_text())
    for v in d["vectors"]:
        expected_canon = _jcs(v["input"])
        assert v["canonical"] == expected_canon, (
            f"{v['name']}: canonical drift. Regenerate vectors.json."
        )


def test_each_vector_sha256_matches_canonical() -> None:
    d = json.loads(VECTORS_PATH.read_text())
    for v in d["vectors"]:
        expected_hash = hashlib.sha256(v["canonical"].encode("utf-8")).hexdigest()
        assert v["sha256"] == expected_hash, (
            f"{v['name']}: sha256 drift. Regenerate vectors.json."
        )
