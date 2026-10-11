"""`asqav audit-pack verify --offline` over a real pack (one flat, two chained receipts with
pending OTS anchors) and a minted RFC3161 pack, with the network refused."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from asqav.verifier import pack_verify as pv
from asqav.verifier import verify_receipt as vr
from tests.test_exit_artifact_cli import SITECUSTOMIZE

pytest.importorskip("dilithium_py")

REAL_PACK = Path(__file__).parent / "fixtures" / "audit_pack_offline_2026_10_11.json"
FLAT, FIRST, SECOND = 0, 1, 2
DEAD_BASE = "http://127.0.0.1:9"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*_a, **_k):
        raise OSError("outbound network blocked by test_audit_pack_offline_verify")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setenv("ASQAV_API_URL", DEAD_BASE)
    monkeypatch.setenv("ASQAV_API_BASE", DEAD_BASE)
    monkeypatch.delenv("ASQAV_API_KEY", raising=False)


def _real_pack() -> dict:
    return vr._load(str(REAL_PACK))


def _run_cli(tmp_path: Path, pack: dict, *extra: str) -> subprocess.CompletedProcess:
    (tmp_path / "sitecustomize.py").write_text(SITECUSTOMIZE)
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(pack))
    env = {k: v for k, v in os.environ.items() if k != "ASQAV_API_KEY"}
    env.update(PYTHONPATH=str(tmp_path), ASQAV_API_URL=DEAD_BASE, ASQAV_API_BASE=DEAD_BASE)
    code = "import sys; from asqav.cli import app; sys.argv[0] = 'asqav'; app()"
    argv = ["audit-pack", "verify", "--offline", str(path), *extra]
    return subprocess.run(
        [sys.executable, "-c", code, *argv], cwd=tmp_path, capture_output=True, text=True, env=env
    )


def _summary(*, total: int, verified: int, not_offline: int) -> dict:
    return {
        "total": total, "verified": verified, "not_verifiable_offline": not_offline, "invalid": 0,
    }


def _json_report(proc: subprocess.CompletedProcess) -> dict:
    assert "outbound network blocked" not in proc.stderr, proc.stderr
    return json.loads(proc.stdout)


def test_real_pack_verifies_offline_with_sockets_refused(tmp_path: Path) -> None:
    proc = _run_cli(tmp_path, _real_pack(), "--output", "json")
    report = _json_report(proc)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert report["bundle"]["status"] == pv.VERIFIED, report["bundle"]
    assert report["summary"] == _summary(total=3, verified=0, not_offline=3)
    flat, first, second = report["receipts"]
    assert "non-compliance-mode receipt" in flat["reasons"][0]
    for entry in (first, second):
        assert entry["signature"] == "signature valid"
        assert entry["reasons"] and all(r.startswith("anchors:") for r in entry["reasons"])
        assert "pending" in entry["reasons"][0]
    assert first["chain"].startswith("first receipt on chain")
    assert second["chain"] == "chain link rederives from predecessor"


def test_real_pack_text_report_names_each_receipt(tmp_path: Path) -> None:
    proc = _run_cli(tmp_path, _real_pack())
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "[  ok] bundle   bundle signature valid" in proc.stdout
    assert "0/3 verified, 3 not verifiable offline, 0 invalid" in proc.stdout
    assert "Error" not in proc.stdout


def test_tampered_receipt_is_invalid(tmp_path: Path) -> None:
    pack = _real_pack()
    pack["receipts"][SECOND]["payload"]["decision"] = "deny"
    report = _json_report(proc := _run_cli(tmp_path, pack, "--output", "json"))
    assert proc.returncode == 1
    second = report["receipts"][SECOND]
    assert second["status"] == pv.INVALID
    assert any(r.startswith("signature:") for r in second["reasons"]), second["reasons"]
    assert report["bundle"]["status"] == pv.INVALID


def test_tampered_chain_link_fails_the_untouched_successor() -> None:
    pack = _real_pack()
    pack["receipts"][FIRST]["payload"]["nonce"] = "0" * 24
    report = pv.verify_pack(pack)
    assert report["exit_code"] == 1
    second = report["receipts"][SECOND]
    assert second["signature"] == "signature valid"
    assert second["status"] == pv.INVALID
    assert any(r.startswith("chain: chain break") for r in second["reasons"]), second["reasons"]


def test_tampered_bundle_signature_is_invalid(tmp_path: Path) -> None:
    pack = _real_pack()
    sig = pack["bundle_signature"]
    pack["bundle_signature"] = ("B" if sig[0] != "B" else "C") + sig[1:]
    report = _json_report(proc := _run_cli(tmp_path, pack, "--output", "json"))
    assert proc.returncode == 1
    assert report["bundle"]["status"] == pv.INVALID
    assert report["bundle"]["reason"] == "bundle signature: signature mismatch"
    assert report["summary"]["invalid"] == 0


def test_edited_pack_member_breaks_the_bundle_digest() -> None:
    pack = _real_pack()
    pack["regime_mapping"] = {"eu_ai_act": []}
    report = pv.verify_pack(pack)
    assert report["bundle"]["reason"] == "bundle_digest does not match the pack contents"
    assert report["exit_code"] == 1


def test_bundle_key_outside_the_key_set_is_not_verified() -> None:
    pack = _real_pack()
    other = json.loads(json.dumps(pack["jwks"]))
    other["keys"] = []
    report = pv.verify_pack(pack, jwks=other)
    assert report["bundle"]["status"] == pv.NOT_OFFLINE
    assert report["summary"]["verified"] == 0


def test_tampered_flat_receipt_is_invalid() -> None:
    pack = _real_pack()
    flat = pack["receipts"][FLAT]
    msg = base64.b64decode(flat["message_b64"]).replace(b"permit", b"deny")
    flat["message_b64"] = base64.b64encode(msg).decode()
    result = pv.verify_pack(pack)["receipts"][FLAT]
    assert result["status"] == pv.INVALID, result


def test_lapsed_expiry_is_a_note_not_a_failure(monkeypatch) -> None:
    class _Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(days=3650)

    monkeypatch.setattr(vr, "datetime", _Later)
    report = pv.verify_pack(_real_pack())
    assert report["summary"]["invalid"] == 0
    for entry in report["receipts"][FIRST:]:
        assert entry["status"] == pv.NOT_OFFLINE
        assert not any(r.startswith("expiry") for r in entry["reasons"])
        assert "signed before it, so not a failure" in entry["notes"][0]


def test_withheld_signed_bytes_are_not_verifiable_offline() -> None:
    pack = _real_pack()
    pack["receipts"][SECOND]["signed_bytes_withheld"] = True
    result = pv.verify_pack(pack)["receipts"][SECOND]
    assert result["status"] == pv.NOT_OFFLINE
    assert "--include-signed-bytes" in result["reasons"][0]


# A two-receipt pack with RFC3161 anchors; `tsa_trust` carries the TSA keys when asked.
def _minted_pack(*, carry_tsa: bool, expires_at: str = "2099-01-01T00:00:00.000000Z") -> tuple:
    from dilithium_py.ml_dsa import ML_DSA_65

    from tests.tsa_testkit import mint_ml_dsa_anchor

    pk, sk = ML_DSA_65.keygen()
    org, kid = "11111111-2222-4333-8444-555555555555", "minted-key-01"
    receipts, tsa_keys, prev = [], [], "0" * 64
    for seq in (1, 2):
        payload = {
            "type": "protectmcp:decision", "v": 1, "seq": seq, "agent_id": "agt_minted",
            "issued_at": f"2026-06-19T00:00:0{seq}.000000Z", "issuer_id": org, "org_id": org,
            "action_ref": "sha256:" + "a" * 64, "payload_digest": {"hash": "b" * 64, "size": 1},
            "policy_digest": "sha256:" + "c" * 64, "previousReceiptHash": prev,
            "decision": "allow", "expires_at": expires_at,
        }
        sig = base64.b64encode(ML_DSA_65.sign(sk, vr.canonical_json(payload))).decode()
        env = {"payload": payload, "signature": {"alg": "ML-DSA-65", "kid": org, "sig": sig}}
        bound = hashlib.sha256(vr.envelope_minus_anchors_jcs(env)).digest()
        token, tsa_pk = mint_ml_dsa_anchor(bound)
        receipts.append({"signature_id": f"sig_{seq}", "agent_id": "agt_minted", **env,
                         "anchors": [{"type": "rfc3161", "value": token}]})
        tsa_keys.append(base64.b64encode(tsa_pk).decode())
        prev = hashlib.sha256(vr.canonical_json(payload)).hexdigest()
    key = {"kid": kid, "issuer_id": org, "org_id": org, "agent_id": "agt_minted",
           "alg": "ML-DSA-65", "status": "active", "public_key": base64.b64encode(pk).decode()}
    pack = {
        "organization_id": org, "start": "2026-06-19T00:00:00", "end": "2026-06-20T00:00:00",
        "agent_id": None, "only_compliance": True, "algorithm_registry_version": "2026-05-04.v1",
        "receipt_count": 2, "receipts": receipts, "revocation_manifest": [],
        "regime_mapping": {}, "jwks": {"keys": [key]},
    }
    if carry_tsa:
        pack["tsa_trust"] = tsa_keys
    msg = vr.canonical_json(pv.signed_bundle_payload(pack))
    pack.update(bundle_digest="sha256:" + hashlib.sha256(msg).hexdigest(),
                bundle_signature=base64.b64encode(ML_DSA_65.sign(sk, msg)).decode(),
                bundle_signature_algorithm="ML-DSA-65", bundle_public_key=key["public_key"])
    return pack, tsa_keys


def test_pack_carrying_tsa_trust_reaches_verified(tmp_path: Path) -> None:
    pack, _ = _minted_pack(carry_tsa=True)
    report = _json_report(proc := _run_cli(tmp_path, pack, "--output", "json"))
    assert proc.returncode == 0, proc.stdout
    assert report["status"] == pv.VERIFIED
    assert report["summary"] == _summary(total=2, verified=2, not_offline=0)
    assert report["receipts"][1]["chain"] == "chain link rederives from predecessor"


def test_pack_missing_tsa_material_is_not_verifiable_offline(tmp_path: Path) -> None:
    pack, tsa_keys = _minted_pack(carry_tsa=False)
    report = _json_report(proc := _run_cli(tmp_path, pack, "--output", "json"))
    assert proc.returncode == 2, proc.stdout
    assert report["summary"]["invalid"] == 0 and report["bundle"]["status"] == pv.VERIFIED
    for entry in report["receipts"]:
        assert entry["status"] == pv.NOT_OFFLINE
        assert entry["reasons"][0].startswith("anchors:") and "TSA" in entry["reasons"][0]
    for i, key in enumerate(tsa_keys):
        (tmp_path / f"tsa{i}.b64").write_text(key)
    pinned = _run_cli(tmp_path, pack, "--tsa-key", "tsa0.b64", "--tsa-key", "tsa1.b64")
    assert pinned.returncode == 0, pinned.stdout


def test_lapsed_expiry_signed_before_it_still_verifies() -> None:
    pack, _ = _minted_pack(carry_tsa=True, expires_at="2026-06-20T00:00:00.000000Z")
    report = pv.verify_pack(pack)
    assert report["exit_code"] == 0, report["receipts"]
    assert all("not a failure" in r["notes"][0] for r in report["receipts"])


def test_tampered_tsa_trust_breaks_the_bundle() -> None:
    pack, _ = _minted_pack(carry_tsa=True)
    pack["tsa_trust"] = copy.deepcopy(pack["tsa_trust"])[::-1]
    assert pv.verify_pack(pack)["bundle"]["status"] == pv.INVALID
