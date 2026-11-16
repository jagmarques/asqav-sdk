"""A pack report says whether any anchor verified outside the pack signer."""

from __future__ import annotations

import base64
import copy
import hashlib

import pytest

from asqav.client._vocabulary import WITNESS_NAMESPACE
from asqav.verifier import pack_verify as pv
from asqav.verifier import verify_receipt as vr
from tests.test_anchor_crypto import _mint_ots
from tests.test_audit_pack_offline_verify import _minted, _seal, _tsa_bytes
from tests.tsa_testkit import make_timestamp_resp, make_tst_info, mint_ml_dsa_anchor

RFC3161 = next(t for t in WITNESS_NAMESPACE if t.startswith("rfc"))
OTS = next(t for t in WITNESS_NAMESPACE if t.startswith("open"))


def _resealed(minted: dict, mutate) -> dict:
    pack = copy.deepcopy(minted["pack"])
    mutate(pack)
    return {**minted, "pack": _seal(pack, minted["sk"], minted["pk"])}


def _tamper_imprint(pack: dict) -> None:
    entry = pack["receipts"][0]
    entry["signed_bytes_withheld"] = True
    digest = hashlib.sha256(vr.envelope_minus_anchors_jcs(vr.normalise_envelope(entry))).digest()
    anchor = entry["anchors"][0]
    raw = bytearray(base64.b64decode(anchor["value"]))
    at = raw.find(digest)
    assert at >= 0
    raw[at] ^= 1
    anchor["value"] = base64.b64encode(bytes(raw)).decode()


def _verify(minted: dict) -> dict:
    return pv.verify_pack(minted["pack"], jwks=minted["jwks"], tsa_keys=_tsa_bytes(minted["tsa"]))


def _replaced(anchors) -> dict:
    def replace(pack: dict) -> None:
        for entry in pack["receipts"]:
            entry["anchors"] = anchors

    return _resealed(_minted(), replace)


def test_pack_verify_anchor_fixtures() -> None:
    plain = _verify(_minted(seqs=()))
    assert (plain["status"], plain["operator_independent"]) == (pv.VERIFIED, False)
    assert plain["operator_independence_note"] == pv.NO_ANCHOR_NOTE
    valid = _verify(_minted())
    assert (valid["status"], valid["operator_independent"]) == (pv.VERIFIED, True)
    tampered = _resealed(_minted(), _tamper_imprint)
    report = pv.verify_pack(tampered["pack"], jwks=tampered["jwks"])
    assert "operator_independent" in report
    assert (report["status"], report["operator_independent"]) == (pv.INVALID, False)
    assert RFC3161 in report["operator_independence_note"]


# NEGCTL: empty, unknown, string and null anchors are not external.
@pytest.mark.parametrize(
    "anchors",
    [[{"type": RFC3161, "value": ""}], [{"type": "not-an-anchor-type", "value": "AAAA"}],
     RFC3161, None],
)
def test_pack_verify_anchors_not_counted_negctl(anchors) -> None:
    report = _verify(_replaced(anchors))
    assert "operator_independent" in report
    assert report["operator_independent"] is False
    if isinstance(anchors, str):
        assert "not a list" in report["operator_independence_note"]


# NEGCTL: a valid proof for another digest fails the pack.
def test_pack_verify_foreign_anchor_negctl() -> None:
    token, _tsa_pk = mint_ml_dsa_anchor(b"\x00" * 32)
    proof, _root = _mint_ots(b"\x11" * 32, 900000)
    for anchor in ({"type": RFC3161, "value": token}, {"type": OTS, "value": proof}):
        report = _verify(_replaced([anchor]))
        assert (report["status"], report["operator_independent"]) == (pv.INVALID, False)
        assert "different digest" in report["operator_independence_note"]


# NEGCTL: a token signed by the pack's own operator key is not external.
def test_pack_verify_operator_anchor_negctl() -> None:
    from dilithium_py.ml_dsa import ML_DSA_65

    m = _minted()

    def sign_with_operator(pack: dict) -> None:
        entry = pack["receipts"][0]
        env = vr.normalise_envelope(entry)
        tst = make_tst_info(hashlib.sha256(vr.envelope_minus_anchors_jcs(env)).digest())
        token = base64.b64encode(make_timestamp_resp(tst, ML_DSA_65.sign(m["sk"], tst))).decode()
        entry["anchors"] = [{"type": RFC3161, "value": token}]

    pack = _resealed(m, sign_with_operator)["pack"]
    assert pv.verify_pack(pack, jwks=m["jwks"], tsa_keys=[m["pk"]])["operator_independent"] is False
