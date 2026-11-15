"""Pack anchors are audited by default; one failed anchor fails the pack."""

import base64
import hashlib

from asqav.client._vocabulary import WITNESS_NAMESPACE
from asqav.verifier import pack_verify as pv
from asqav.verifier import verify_receipt as vr
from tests.test_anchor_crypto import _mint_ots
from tests.test_audit_pack_offline_verify import _minted, _seal
from tests.tsa_testkit import make_timestamp_resp, make_tst_info

RFC3161, OTS = sorted(WITNESS_NAMESPACE, reverse=True)


def _bound(pack: dict) -> bytes:
    return hashlib.sha256(vr.canonical_json(pv.signed_bundle_payload(pack))).digest()


def _anchored(*, operator: bool = False) -> dict:
    from dilithium_py.ml_dsa import ML_DSA_65
    m = _minted(seqs=())
    pk, sk = (m["pk"], m["sk"]) if operator else ML_DSA_65.keygen()
    m["pack"]["tsa_trust"] = [base64.b64encode(pk).decode()]
    _seal(m["pack"], m["sk"], m["pk"])
    tst = make_tst_info(_bound(m["pack"]))
    token = base64.b64encode(make_timestamp_resp(tst, ML_DSA_65.sign(sk, tst))).decode()
    m["pack"]["anchors"] = [{"type": RFC3161, "value": token}]
    return m


def test_pack_verify_anchor_fixtures() -> None:
    clean = _minted(seqs=())
    result = pv.verify_pack(clean["pack"], jwks=clean["jwks"])
    assert (result["status"], result["operator_independent"]) == (pv.VERIFIED, False)
    assert result["operator_independence_note"] == pv.NO_ANCHOR_NOTE
    valid = _anchored()
    result = pv.verify_pack(valid["pack"], jwks=valid["jwks"])
    assert (result["status"], result["operator_independent"]) == (pv.VERIFIED, True)
    tampered = _anchored()
    token = bytearray(base64.b64decode(tampered["pack"]["anchors"][0]["value"]))
    token[token.find(_bound(tampered["pack"]))] ^= 0xFF
    tampered["pack"]["anchors"][0]["value"] = base64.b64encode(token).decode()
    result = pv.verify_pack(tampered["pack"], jwks=tampered["jwks"])
    assert "operator_independent" in result
    assert (result["status"], result["operator_independent"]) == (pv.INVALID, False)
    assert "different digest" in result["operator_independence_note"]


def test_anchors_without_an_external_proof_do_not_count() -> None:
    cases = [
        ([{"type": RFC3161, "value": ""}], pv.INVALID),
        ([{"type": "unknown-anchor", "value": "YWJj"}], pv.VERIFIED),
        ([], pv.VERIFIED), (None, pv.INVALID), (RFC3161, pv.INVALID),
    ]
    for anchors, expected in cases:
        m = _minted(seqs=())
        m["pack"]["anchors"] = anchors
        result = pv.verify_pack(m["pack"], jwks=m["jwks"])
        assert (result["status"], result["operator_independent"]) == (expected, False), anchors
        if anchors == []:
            assert result["operator_independence_note"] == pv.NO_ANCHOR_NOTE
    operator = _anchored(operator=True)
    assert pv.verify_pack(operator["pack"], jwks=operator["jwks"])["operator_independent"] is False
    m = _anchored()
    del m["pack"]["bundle_digest"]
    result = pv.verify_pack(m["pack"], jwks=m["jwks"])
    assert (result["status"], result["operator_independent"]) == (pv.INVALID, False)
    assert "bundle_digest" in result["operator_independence_note"]


def test_inclusion_proof_for_another_leaf_fails() -> None:
    m = _minted(seqs=())
    m["pack"]["anchors"] = [{"type": OTS, "value": _mint_ots(b"\x11" * 32, 965451)[0]}]
    result = pv.verify_pack(m["pack"], jwks=m["jwks"])
    assert (result["status"], result["operator_independent"]) == (pv.INVALID, False)
    assert "different digest" in result["operator_independence_note"]
    m["pack"]["anchors"] = [{"type": OTS, "value": _mint_ots(_bound(m["pack"]), 965451)[0]}]
    headers = {"965451": {"merkle_root": "00" * 32, "time": "2026-06-01T12:00:00Z"}}
    result = pv.verify_pack(m["pack"], jwks=m["jwks"], bitcoin_headers=headers)
    assert result["status"] == pv.INVALID
    assert "merkle path" in result["operator_independence_note"]
