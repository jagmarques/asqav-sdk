"""Anti-vacuous tests for asqav.commitment (criterion 281).

Prove the keyed commitment helper is deterministic for fixed inputs, changes
when any single input changes, mints 16 random opening bytes, and never
default-generates the caller-held key.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.commitment import commit, new_opening

_KEY = bytes(range(32))
_OPENING = bytes(range(16))
_LABEL = "asqav.attestation"
_VERSION = 1
_DATA = b"claim-bytes"

# Independent HMAC-SHA256 over opening || label || version(4 BE) || data.
_GOLDEN = "4895dd9c2ad9a9f35f18d7c60d6957e4a87b017dff65d1316f76d437905fff88"


def test_commit_matches_independent_hmac_golden() -> None:
    """commit() equals a hand-computed HMAC-SHA256 over the framed input."""
    got = commit(_KEY, _OPENING, _LABEL, _VERSION, _DATA)
    assert got == _GOLDEN
    framed = _OPENING + _LABEL.encode() + _VERSION.to_bytes(4, "big") + _DATA
    assert got == hmac.new(_KEY, framed, hashlib.sha256).hexdigest()


def test_commit_is_deterministic_for_fixed_inputs() -> None:
    """Same inputs always yield the same lowercase hex digest."""
    first = commit(_KEY, _OPENING, _LABEL, _VERSION, _DATA)
    second = commit(_KEY, _OPENING, _LABEL, _VERSION, _DATA)
    assert first == second
    assert len(first) == 64
    assert first == first.lower()


def test_commit_changes_when_key_changes() -> None:
    other = commit(bytes(range(1, 33)), _OPENING, _LABEL, _VERSION, _DATA)
    assert other != commit(_KEY, _OPENING, _LABEL, _VERSION, _DATA)


def test_commit_changes_when_opening_changes() -> None:
    other = commit(_KEY, new_opening(), _LABEL, _VERSION, _DATA)
    assert other != commit(_KEY, _OPENING, _LABEL, _VERSION, _DATA)


def test_commit_changes_when_label_changes() -> None:
    other = commit(_KEY, _OPENING, "asqav.other", _VERSION, _DATA)
    assert other != commit(_KEY, _OPENING, _LABEL, _VERSION, _DATA)


def test_commit_changes_when_version_changes() -> None:
    other = commit(_KEY, _OPENING, _LABEL, _VERSION + 1, _DATA)
    assert other != commit(_KEY, _OPENING, _LABEL, _VERSION, _DATA)


def test_commit_changes_when_data_changes() -> None:
    other = commit(_KEY, _OPENING, _LABEL, _VERSION, b"other-bytes")
    assert other != commit(_KEY, _OPENING, _LABEL, _VERSION, _DATA)


def test_version_is_four_big_endian_bytes_in_framing() -> None:
    """version 1 and 256 differ, proving a 4-byte big-endian field, not a byte."""
    low = commit(_KEY, _OPENING, _LABEL, 1, _DATA)
    high = commit(_KEY, _OPENING, _LABEL, 256, _DATA)
    assert low != high


def test_new_opening_is_sixteen_bytes() -> None:
    assert len(new_opening()) == 16
    assert isinstance(new_opening(), bytes)


def test_new_opening_is_random_across_calls() -> None:
    """Two openings differ (CSPRNG); a constant opening would be a bug."""
    assert new_opening() != new_opening()


def test_commit_has_no_default_key() -> None:
    """The key is caller-supplied: commit() must not default-generate it."""
    params = inspect.signature(commit).parameters
    assert params["key"].default is inspect.Parameter.empty


def test_public_surface_exposed_at_package_root() -> None:
    import asqav

    assert asqav.commit is commit
    assert asqav.new_opening is new_opening
