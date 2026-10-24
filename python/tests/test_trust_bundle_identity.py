# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Versioned TSA trust bundle identity on structured results (criterion 584).

Two different pinned-trust inputs must yield different trust_bundle digests;
the EU Trusted List gap stays explicitly declared; a bundle whose recorded
digest disagrees with its contents can never verify anchors through it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from asqav.verifier import verify_receipt as v

_VECTORS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"
_BUNDLE = (
    Path(__file__).resolve().parents[2] / "verifier" / "trust" / "tsa-trust-bundle.json"
)
_VECTOR = "asqav-24-anchor-block-hash-prod"


def _vector():
    d = _VECTORS / _VECTOR
    receipt = json.loads((d / "receipt.json").read_text())
    jwks = json.loads((d / "jwks.json").read_text())
    pred_doc = json.loads((d / "predecessor.json").read_text())
    pred = pred_doc["payload"] if isinstance(pred_doc.get("payload"), dict) else pred_doc
    keys = (d / "tsa_trust.pem").read_bytes()
    return receipt, jwks, pred, keys, json.loads(_BUNDLE.read_text())


def _rebind(bundle: dict, roots: list) -> dict:
    """A second bundle over other roots with a correctly recomputed digest."""
    other = copy.deepcopy(bundle)
    other["roots"] = roots
    body = {k: val for k, val in other.items() if k != "digest"}
    other["digest"] = hashlib.sha256(v.canonical_json(body)).hexdigest()
    return other


def _anchors_axis(result: dict) -> dict:
    return next(a for a in result["axes"] if a["name"] == "anchors")


def test_two_bundles_yield_different_identity_digests() -> None:
    """2-cert roots vs cert[1]-only roots: the identity digest must differ.

    Non-vacuous by construction: without the feature both runs carry no
    trust_bundle key at all, so the subscript fails on the old tree.
    """
    receipt, jwks, pred, keys, full = _vector()
    single = _rebind(full, full["roots"][1:])
    assert len(single["roots"]) == 1 and len(full["roots"]) == 2
    via_full = v.run_structured(
        receipt, jwks, pred, trusted_tsa_keys=[keys], trust_bundle=full,
        trust_bundle_source="tsa-trust-bundle.json",
    )
    via_single = v.run_structured(
        receipt, jwks, pred, trusted_tsa_keys=[keys], trust_bundle=single,
        trust_bundle_source="tsa-trust-bundle.json",
    )
    assert via_full["trust_bundle"]["digest"] != via_single["trust_bundle"]["digest"]
    assert via_full["trust_bundle"]["version"] == full["bundle_version"]


def test_trusted_list_status_stays_declared() -> None:
    """The EU Trusted List gap is in the NOT_CHECKED declaration, not silent."""
    assert "tsa_trusted_list_status" in {e["check"] for e in v.NOT_CHECKED}
    assert "tsa_trusted_list_status" in {
        e["check"] for e in v.not_checked_declaration()
    }


def test_tampered_bundle_cannot_verify_anchors() -> None:
    """A digest mismatch makes the anchors axis unverifiable, never PASS."""
    receipt, jwks, pred, keys, full = _vector()
    tampered = copy.deepcopy(full)
    tampered["roots"][0]["subject"] = "CN=not-the-issuer"
    bad = v.run_structured(
        receipt, jwks, pred, trusted_tsa_keys=[keys], trust_bundle=tampered,
        trust_bundle_source="tsa-trust-bundle.json",
    )
    axis = _anchors_axis(bad)
    assert axis["result"] != "PASS", axis["note"]
    assert "digest mismatch" in axis["note"]
    good = v.run_structured(
        receipt, jwks, pred, trusted_tsa_keys=[keys], trust_bundle=full,
        trust_bundle_source="tsa-trust-bundle.json",
    )
    assert _anchors_axis(good)["result"] != "FAIL"
    assert "digest mismatch" not in _anchors_axis(good)["note"]
    assert good["trust_bundle"]["valid"] is True
