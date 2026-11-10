"""Pack anchors are audited by default; NEGCTL: one flipped token byte fails."""

from __future__ import annotations

import base64

import pytest

from asqav.client._vocabulary import WITNESS_NAMESPACE
from asqav.verifier import pack_verify as pv
from tests.test_anchor_crypto import _mint_ots
from tests.test_audit_pack_offline_verify import _minted, _seal, _tsa_bytes
from tests.tsa_testkit import make_timestamp_resp, make_tst_info

pytest.importorskip("dilithium_py")

RFC3161 = next(t for t in WITNESS_NAMESPACE if t.startswith("rfc"))
OPENTIMESTAMPS = next(t for t in WITNESS_NAMESPACE if t.startswith("open"))


def _token(bound: bytes, sk) -> str:
    from dilithium_py.ml_dsa import ML_DSA_65

    tst = make_tst_info(bound)
    return base64.b64encode(make_timestamp_resp(tst, ML_DSA_65.sign(sk, tst))).decode()


def _anchored() -> dict:
    from dilithium_py.ml_dsa import ML_DSA_65

    pk, sk = ML_DSA_65.keygen()
    m = _minted(seqs=(1,))
    m["pack"]["tsa_trust"] = [base64.b64encode(pk).decode()]
    _seal(m["pack"], m["sk"], m["pk"])
    m["pack"]["anchors"] = [{"type": RFC3161, "value": _token(pv._pack_bound(m["pack"]), sk)}]
    return m


def test_pack_verify_anchor_fixtures() -> None:
    clean = _minted(seqs=(1,))
    result = pv.verify_pack(clean["pack"], jwks=clean["jwks"], tsa_keys=_tsa_bytes(clean["tsa"]))
    assert (result["status"], result["operator_independent"]) == (pv.VERIFIED, False)
    assert "signer holds the key" in result["operator_independence_note"]
    valid = _anchored()
    result = pv.verify_pack(valid["pack"], jwks=valid["jwks"], tsa_keys=_tsa_bytes(valid["tsa"]))
    assert (result["status"], result["operator_independent"]) == (pv.VERIFIED, True)
    tampered = _anchored()
    token = bytearray(base64.b64decode(tampered["pack"]["anchors"][0]["value"]))
    at = bytes(token).find(pv._pack_bound(tampered["pack"]))
    token[at] ^= 0xFF
    tampered["pack"]["anchors"][0]["value"] = base64.b64encode(bytes(token)).decode()
    result = pv.verify_pack(tampered["pack"], jwks=tampered["jwks"], tsa_keys=_tsa_bytes(tampered["tsa"]))
    assert (result["status"], result["operator_independent"]) == (pv.INVALID, False)
    assert any("rfc3161" in r and "different digest" in r for r in result["anchors"]["reasons"])


def test_empty_or_operator_signed_anchors_are_not_external() -> None:
    for anchors in ([{"type": RFC3161, "value": ""}], [{"type": "no-such", "value": "YWJjZA=="}],
                    RFC3161, None):
        m = _minted(seqs=())
        m["pack"]["anchors"] = anchors
        result = pv.verify_pack(m["pack"], jwks=m["jwks"])
        assert result["operator_independent"] is False
    m = _minted(seqs=())
    m["pack"]["tsa_trust"] = [base64.b64encode(m["pk"]).decode()]
    _seal(m["pack"], m["sk"], m["pk"])
    m["pack"]["anchors"] = [{"type": RFC3161, "value": _token(pv._pack_bound(m["pack"]), m["sk"])}]
    result = pv.verify_pack(m["pack"], jwks=m["jwks"])
    assert result["operator_independent"] is False


def test_unverified_inclusion_proofs_fail() -> None:
    for digest, root_ok in ((b"\x11" * 32, True), (None, False)):
        m = _minted(seqs=())
        value, root = _mint_ots(pv._pack_bound(m["pack"]) if digest is None else digest, 965451)
        m["pack"]["anchors"] = [{"type": OPENTIMESTAMPS, "value": value}]
        merkle_root = root[::-1].hex() if root_ok else "00" * 32
        headers = {"965451": {"merkle_root": merkle_root, "time": "2026-06-01T12:00:00Z"}}
        result = pv.verify_pack(m["pack"], jwks=m["jwks"], bitcoin_headers=headers)
        assert (result["status"], result["operator_independent"]) == (pv.INVALID, False)
