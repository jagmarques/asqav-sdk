"""Archive page offline verification (criterion 761, integrated path).

_verify_archive_page must accept exactly what GET /audit-pack/archive
emits - the full envelope digest plus a signature that verifies against
the bundled key - and reject everything else. The subset-digest test is
the regression for the reported defect: the old CLI recomputed over
{organization_id, receipts}, so genuine server pages aborted at cursor 0.
"""

from __future__ import annotations

import base64
import hashlib

import pytest

from asqav._jcs import canonical_json
from asqav.cli import _verify_archive_page
from tests._archive_page_761 import _ed25519_keypair, _page_full


def _receipt(i):
    return {"signature_id": f"sig_{i}", "agent_id": "agt_arc"}


def test_accepts_full_envelope_with_real_signature():
    page = _page_full([_receipt(1)], 1, True)
    assert isinstance(_verify_archive_page(page), bytes)


def test_rejects_subset_digest():
    """The old defect: a digest over {organization_id, receipts} must fail."""
    page = _page_full([_receipt(1)], None, False)
    subset = {"organization_id": page["organization_id"], "receipts": page["receipts"]}
    page["page_digest"] = "sha256:" + hashlib.sha256(
        canonical_json(subset)
    ).hexdigest()
    with pytest.raises(ValueError, match="digest mismatch"):
        _verify_archive_page(page)


def test_rejects_tampered_receipts():
    page = _page_full([_receipt(1)], None, False)
    page["receipts"].append({"signature_id": "sig_x", "agent_id": "agt_arc"})
    with pytest.raises(ValueError, match="digest mismatch"):
        _verify_archive_page(page)


def test_rejects_wrong_key_signature():
    page = _page_full([_receipt(1)], None, False)
    other_public, _ = _ed25519_keypair()
    page["page_public_key"] = base64.b64encode(other_public).decode("ascii")
    with pytest.raises(ValueError, match="signature"):
        _verify_archive_page(page)


def test_rejects_missing_signature_material():
    page = _page_full([_receipt(1)], None, False)
    del page["page_signature"]
    with pytest.raises(ValueError, match="[Ss]ignature"):
        _verify_archive_page(page)
