# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Offline revocation policy declaration (criterion 584 C2).

The revocation row must name the bundle-issue assertion and the refresh
input; a null condition would claim the gap is never narrowable.
"""

from __future__ import annotations

from asqav.verifier.verify_receipt import NOT_CHECKED

_ROW = "tsa_certificate_revocation"


def _row() -> dict:
    matches = [e for e in NOT_CHECKED if e["check"] == _ROW]
    assert len(matches) == 1, "revocation row must exist exactly once"
    return matches[0]


def test_revocation_row_states_the_bundle_issue_policy():
    row = _row()
    assert "no live CRL or OCSP lookup" in row["reason"]
    assert "bundle" in row["reason"] and "refreshed" in row["reason"]


def test_revocation_row_names_the_refresh_input():
    row = _row()
    assert row["condition"] is not None, "null condition claims never narrowable"
    assert "bundle_version" in row["condition"]
