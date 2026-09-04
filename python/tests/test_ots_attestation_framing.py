# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""The OpenTimestamps attestation framing: varbytes payload, LEB128 height.

An attestation serialises as tag + varbytes(payload), and a Bitcoin
attestation's payload is the block height as a LEB128 varuint
(opentimestamps/core/notary.py TimeAttestation.deserialize, and
serialize.py's read_varuint/write_varuint, "unsigned little-endian base128").
The parser read the height varuint straight after the tag and grouped the
bytes big-endian: asqav-24's production proof (payload cb f6 3a, height
965451) consumed the length byte as height 3 and left the height bytes
"trailing", so no upgraded proof could ever complete offline.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from asqav.verifier import verify_receipt as v

_VECTOR = (
    Path(__file__).resolve().parents[2]
    / "verifier" / "conformance-vectors" / "asqav-24-anchor-block-hash-prod"
)
#: Block 965451's merkle root in display order, reversed to internal order —
#: the value the proof's op chain must evaluate to (ANCHOR-MATERIAL.md records
#: the two public sources for it).
_BLOCK_965451_ROOT_DISPLAY = "76e472dea0ba9cb2adafe0d47ef54b5928c8a9443201300c848c77894e54c57c"


def _proof() -> bytes:
    receipt = json.loads((_VECTOR / "receipt.json").read_text())
    ots = next(a for a in receipt["anchors"] if a.get("type") == "opentimestamps")
    return base64.b64decode(ots["value"])


def _parse(blob: bytes):
    off = len(v._OTS_MAGIC) + 1 + 1  # version byte, then the hash-op byte
    committed, rest = blob[off : off + 32], off + 32
    state = v._OtsState()
    end = v._ots_node(blob, rest, committed, 0, state)
    return end, state


def test_production_proof_parses_to_the_end_with_the_real_height() -> None:
    """asqav-24's proof: consumed exactly, one bitcoin attestation at 965451."""
    blob = _proof()
    end, state = _parse(blob)
    assert end == len(blob), "trailing bytes after the timestamp"
    assert [h for h, _ in state.attestations] == [965451]
    root = state.attestations[0][1]
    assert root == bytes.fromhex(_BLOCK_965451_ROOT_DISPLAY)[::-1]


def test_varuint_is_leb128() -> None:
    """Multi-byte varuints group little-endian: cb f6 3a is 965451, not 1243962."""
    assert v._ots_varuint(bytes.fromhex("cbf63a"), 0) == (965451, 3)
    assert v._ots_varuint(bytes([0x00]), 0) == (0, 1)


def test_attestation_payload_with_a_lying_length_byte_raises() -> None:
    """A varbytes whose length prefix disagrees with its content is a parse error."""
    # tag 0x00 + 8-byte bitcoin tag + varbytes length 5 + only 2 payload bytes
    blob = b"\x00" + bytes.fromhex("0588960d73d71901") + b"\x05" + b"\xcb\xf6"
    with pytest.raises(v._AnchorParseError):
        v._ots_item(0x00, blob, 1, b"\x00" * 32, 0, v._OtsState())


def test_attestation_payload_not_one_varuint_raises() -> None:
    """The height varuint must consume the payload exactly."""
    # tag 0x00 + bitcoin tag + varbytes(len=2, payload 00 00): two varuints' worth.
    blob = b"\x00" + bytes.fromhex("0588960d73d71901") + b"\x02\x00\x00"
    with pytest.raises(v._AnchorParseError):
        v._ots_item(0x00, blob, 1, b"\x00" * 32, 0, v._OtsState())


def test_a_multi_block_pem_yields_one_candidate_per_block() -> None:
    """A PEM chain file splits; a bundle is not one unusable entry."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
    )

    blocks = []
    for _ in range(2):
        pub = Ed25519PrivateKey.generate().public_key().public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        )
        blocks.append(pub)
    raw, pkeys = v._tsa_key_candidates([b"".join(blocks)])
    assert len(pkeys) == 2
    assert raw == []
