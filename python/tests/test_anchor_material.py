# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Per-vector anchor material completes asqav-24's anchors axis offline.

asqav-24-anchor-block-hash-prod ships two pieces of public material beside the
vector: the TSA certificates embedded in its own timestamp token
(`tsa_trust.pem`) and the header of the bitcoin block its OpenTimestamps proof
lands in (`bitcoin_headers.json`). With both, the axis PASSes; with a wrong
merkle root the OpenTimestamps entry is invalid and the axis FAILs; without
the TSA file the token is unverifiable and the axis SKIPs. The axis is the
whole point of shipping the material, so each control asserts on it.
"""

from __future__ import annotations

import json
from pathlib import Path

from asqav.verifier import verify_receipt as v

_VECTOR = (
    Path(__file__).resolve().parents[2]
    / "verifier" / "conformance-vectors" / "asqav-24-anchor-block-hash-prod"
)


def _run(*, tsa: bool = True, wrong_merkle_root: bool = False) -> dict:
    receipt = json.loads((_VECTOR / "receipt.json").read_text())
    jwks = json.loads((_VECTOR / "jwks.json").read_text())
    predecessor = json.loads((_VECTOR / "predecessor.json").read_text())
    if isinstance(predecessor.get("payload"), dict):
        predecessor = predecessor["payload"]
    tsa_keys = [(_VECTOR / "tsa_trust.pem").read_bytes()] if tsa else None
    headers = json.loads((_VECTOR / "bitcoin_headers.json").read_text())
    if wrong_merkle_root:
        headers["965451"]["merkle_root"] = "00" * 32
    return v.run_structured(
        receipt,
        jwks,
        predecessor_payload=predecessor,
        trusted_tsa_keys=tsa_keys,
        bitcoin_headers=headers,
    )


def _anchors(result: dict) -> dict:
    return next(a for a in result["axes"] if a["name"] == "anchors")


def test_shipped_material_completes_the_anchors_axis() -> None:
    anchors = _anchors(_run())
    assert anchors["result"] == "PASS", anchors["note"]
    assert "- rfc3161: verified" in anchors["note"]
    assert "- opentimestamps: verified" in anchors["note"]
    assert "lands in bitcoin block 965451" in anchors["note"]


def test_a_wrong_merkle_root_fails_the_axis() -> None:
    anchors = _anchors(_run(wrong_merkle_root=True))
    assert anchors["result"] == "FAIL", anchors["note"]
    assert "- opentimestamps: invalid" in anchors["note"]
    assert "merkle path does not land in bitcoin block 965451" in anchors["note"]


def test_without_the_tsa_file_the_axis_skips() -> None:
    anchors = _anchors(_run(tsa=False))
    assert anchors["result"] == "SKIPPED", anchors["note"]
    assert "- rfc3161" in anchors["note"] and "unverifiable" in anchors["note"]
    # The OpenTimestamps half still completes: the SKIP is the TSA's alone.
    assert "- opentimestamps: verified" in anchors["note"]
