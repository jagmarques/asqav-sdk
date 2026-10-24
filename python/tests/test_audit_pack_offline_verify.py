"""`asqav audit-pack verify --offline` over a real pack (one flat, two chained
receipts, pending OTS anchors) and a minted RFC3161 pack, network refused."""

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
    path_entries = os.pathsep.join([str(tmp_path), str(Path(pv.__file__).parents[2])])
    env.update(PYTHONPATH=path_entries, ASQAV_API_URL=DEAD_BASE, ASQAV_API_BASE=DEAD_BASE)
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


def _write_jwks(tmp_path: Path, jwks: dict) -> str:
    (tmp_path / "holder-jwks.json").write_text(json.dumps(jwks))
    return str(tmp_path / "holder-jwks.json")


def test_real_pack_verifies_offline_with_sockets_refused(tmp_path: Path) -> None:
    jwks = _write_jwks(tmp_path, _real_pack()["jwks"])
    proc = _run_cli(tmp_path, _real_pack(), "--jwks", jwks, "--output", "json")
    report = _json_report(proc)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert report["status"] == pv.NOT_OFFLINE and report["trust"]["anchored"] is True
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


def test_without_jwks_the_pack_only_vouches_for_itself(tmp_path: Path) -> None:
    proc = _run_cli(tmp_path, _real_pack())
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "[  ok] bundle   bundle signature valid" in proc.stdout
    assert "bundle key: sha256:b4777311" in proc.stdout
    assert "self-asserted by the pack" in proc.stdout
    assert "=> self-consistent, not anchored to a trusted key: 0/3 verified" in proc.stdout
    assert pv.verify_pack(_real_pack())["status"] == pv.SELF_CONSISTENT


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


def test_bundle_key_outside_the_key_set_is_invalid() -> None:
    pack = _real_pack()
    report = pv.verify_pack(pack, jwks={"keys": []})
    assert report["bundle"] == {
        "status": pv.INVALID, "reason": "bundle key is not in the key set", "key_thumbprint": None,
    }
    assert report["exit_code"] == 1


def test_stripped_bundle_signature_is_invalid() -> None:
    pack = _real_pack()
    del pack["bundle_signature"]
    report = pv.verify_pack(pack, jwks=_real_pack()["jwks"])
    assert report["bundle"]["reason"] == "pack carries no bundle signature"
    assert report["exit_code"] == 1


def test_unsupported_bundle_algorithm_is_invalid() -> None:
    pack = _real_pack()
    pack["bundle_signature_algorithm"] = "none"
    report = pv.verify_pack(pack)
    assert report["bundle"]["reason"] == "unsupported bundle signature algorithm 'none'"
    assert report["exit_code"] == 1


def test_bundle_algorithm_must_match_the_key() -> None:
    pack = _real_pack()
    pack["bundle_signature_algorithm"] = "Ed25519"
    report = pv.verify_pack(pack)
    assert "differs from its key's alg ML-DSA-65" in report["bundle"]["reason"]
    assert report["exit_code"] == 1


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
    report = pv.verify_pack(_real_pack(), jwks=_real_pack()["jwks"])
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


ORG, AGENT, OTHER_ORG = "11111111-2222-4333-8444-555555555555", "agt_minted", "org-elsewhere"


def _sign_entry(sk, payload: dict, seq: int) -> tuple[dict, str]:
    from dilithium_py.ml_dsa import ML_DSA_65

    from tests.tsa_testkit import mint_ml_dsa_anchor

    sig = base64.b64encode(ML_DSA_65.sign(sk, vr.canonical_json(payload))).decode()
    env = {"payload": payload, "signature": {"alg": "ML-DSA-65", "kid": ORG, "sig": sig}}
    token, tsa_pk = mint_ml_dsa_anchor(hashlib.sha256(vr.envelope_minus_anchors_jcs(env)).digest())
    entry = {"signature_id": f"sig_{seq}_{payload['decision']}", "agent_id": AGENT, **env,
             "anchors": [{"type": "rfc3161", "value": token}]}
    return entry, base64.b64encode(tsa_pk).decode()


def _payload(seq: int, prev: str, expires_at: str) -> dict:
    return {
        "type": "protectmcp:decision", "v": 1, "mode": "payload", "seq": seq, "agent_id": AGENT,
        "issued_at": f"2026-06-19T00:00:0{seq}.000000Z", "issuer_id": ORG, "org_id": ORG,
        "action_ref": "sha256:" + "a" * 64, "payload_digest": {"hash": "b" * 64, "size": 1},
        "policy_digest": "sha256:" + "c" * 64, "previousReceiptHash": prev,
        "decision": "allow", "expires_at": expires_at,
    }


def _seal(pack: dict, sk, pk: bytes) -> dict:
    from dilithium_py.ml_dsa import ML_DSA_65

    pack["receipt_count"] = len(pack["receipts"])
    msg = vr.canonical_json(pv.signed_bundle_payload(pack))
    pack.update(bundle_digest="sha256:" + hashlib.sha256(msg).hexdigest(),
                bundle_signature=base64.b64encode(ML_DSA_65.sign(sk, msg)).decode(),
                bundle_signature_algorithm="ML-DSA-65",
                bundle_public_key=base64.b64encode(pk).decode())
    return pack


def _minted(seqs=(1, 2, 3), *, carry_tsa=False, expires_at="2099-01-01T00:00:00.000000Z") -> dict:
    """An RFC3161-anchored chain, holder and TSA keys, sealed by the key."""
    from dilithium_py.ml_dsa import ML_DSA_65

    pk, sk = ML_DSA_65.keygen()
    receipts, tsa_keys, prev = [], [], "0" * 64
    for seq in seqs:
        payload = _payload(seq, prev, expires_at)
        entry, tsa = _sign_entry(sk, payload, seq)
        receipts.append(entry)
        tsa_keys.append(tsa)
        prev = hashlib.sha256(vr.canonical_json(payload)).hexdigest()
    key = {"kid": "minted-key-01", "issuer_id": ORG, "org_id": ORG, "agent_id": AGENT,
           "alg": "ML-DSA-65", "status": "active", "public_key": base64.b64encode(pk).decode()}
    pack = {
        "organization_id": ORG, "start": "2026-06-19T00:00:00", "end": "2026-06-20T00:00:00",
        "agent_id": None, "only_compliance": True, "algorithm_registry_version": "2026-05-04.v1",
        "receipts": receipts, "revocation_manifest": [], "regime_mapping": {},
        "jwks": {"keys": [key]},
    }
    if carry_tsa:
        pack["tsa_trust"] = tsa_keys
    return {"pack": _seal(pack, sk, pk), "jwks": {"keys": [dict(key)]}, "tsa": tsa_keys,
            "sk": sk, "pk": pk}


def _tsa_files(tmp_path: Path, tsa_keys: list) -> list:
    args = []
    for i, key in enumerate(tsa_keys):
        (tmp_path / f"tsa{i}.b64").write_text(key)
        args += ["--tsa-key", str(tmp_path / f"tsa{i}.b64")]
    return args


def _tsa_bytes(tsa_keys: list) -> list:
    return [vr.tsa_key_material(k.encode()) for k in tsa_keys]


def test_supplied_keys_and_tsa_keys_reach_verified(tmp_path: Path) -> None:
    m = _minted()
    args = ["--jwks", _write_jwks(tmp_path, m["jwks"]), *_tsa_files(tmp_path, m["tsa"])]
    report = _json_report(proc := _run_cli(tmp_path, m["pack"], *args, "--output", "json"))
    assert proc.returncode == 0, proc.stdout
    assert report["status"] == pv.VERIFIED
    assert report["summary"] == _summary(total=3, verified=3, not_offline=0)
    assert report["receipts"][2]["chain"] == "chain link rederives from predecessor"


def test_tsa_trust_carried_in_the_pack_is_never_a_root(tmp_path: Path) -> None:
    m = _minted(carry_tsa=True)
    args = ["--jwks", _write_jwks(tmp_path, m["jwks"]), "--output", "json"]
    report = _json_report(proc := _run_cli(tmp_path, m["pack"], *args))
    assert proc.returncode == 2, proc.stdout
    assert report["summary"]["invalid"] == 0 and report["bundle"]["status"] == pv.VERIFIED
    assert report["trust"]["tsa_carried_not_trusted"] == 3
    for entry in report["receipts"]:
        assert entry["status"] == pv.NOT_OFFLINE
        assert entry["reasons"][0].startswith("anchors:") and "TSA" in entry["reasons"][0]


def test_revoked_key_cannot_forge_a_verified_pack_with_its_own_tsa(tmp_path: Path) -> None:
    m = _minted(carry_tsa=True)
    m["jwks"]["keys"][0].update(status="revoked", revoked_at="2026-09-01T00:00:00Z")
    args = ["--jwks", _write_jwks(tmp_path, m["jwks"]), "--output", "json"]
    report = _json_report(proc := _run_cli(tmp_path, m["pack"], *args))
    assert proc.returncode == 2, proc.stdout
    assert report["bundle"]["status"] == pv.NOT_OFFLINE
    assert "nothing proves the bundle was signed before it" in report["bundle"]["reason"]
    assert report["summary"]["verified"] == 0


@pytest.mark.parametrize(
    "status,member", [("revoked", "revoked_at"), ("suspended", "suspended_at")]
)
def test_key_revoked_after_the_export_is_not_verifiable_not_invalid(status, member) -> None:
    m = _minted()
    m["jwks"]["keys"][0].update({"status": status, member: "2027-01-01T00:00:00Z"})
    report = pv.verify_pack(m["pack"], jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["bundle"]["status"] == pv.NOT_OFFLINE, report["bundle"]
    assert report["bundle"]["reason"] == (
        f"bundle key {status} at 2027-01-01T00:00:00Z; "
        "nothing proves the bundle was signed before it"
    )
    assert report["summary"]["invalid"] == 0 and report["exit_code"] == 2


def test_trusted_timestamp_after_revocation_proves_the_bundle_invalid() -> None:
    m = _minted()
    m["pack"]["start"] = "2026-04-01T00:00:00"
    _seal(m["pack"], m["sk"], m["pk"])
    m["jwks"]["keys"][0].update(status="revoked", revoked_at="2026-05-01T00:00:00Z")
    report = pv.verify_pack(m["pack"], jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["bundle"]["status"] == pv.INVALID
    reason = report["bundle"]["reason"]
    assert "a receipt in the bundle has a trusted timestamp after that" in reason
    assert report["exit_code"] == 1


def test_signed_window_starting_after_revocation_is_invalid() -> None:
    m = _minted()
    m["jwks"]["keys"][0].update(status="revoked", revoked_at="2026-06-18T00:00:00Z")
    report = pv.verify_pack(m["pack"], jwks=m["jwks"])
    assert report["bundle"]["reason"] == (
        "bundle key revoked at 2026-06-18T00:00:00Z, before the signed window start"
    )
    assert report["exit_code"] == 1


def test_duplicate_signature_id_in_a_signed_pack_is_invalid() -> None:
    """Two entries claiming one id fork the entry set; not a verified pack."""
    from dilithium_py.ml_dsa import ML_DSA_65

    m = _minted()
    m["pack"]["receipts"][1]["signature_id"] = m["pack"]["receipts"][0]["signature_id"]
    msg = vr.canonical_json(pv.signed_bundle_payload(m["pack"]))
    m["pack"].update(
        bundle_digest="sha256:" + hashlib.sha256(msg).hexdigest(),
        bundle_signature=base64.b64encode(ML_DSA_65.sign(m["sk"], msg)).decode(),
    )
    report = pv.verify_pack(m["pack"], jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["status"] == pv.INVALID
    reasons = [reason for entry in report["receipts"] for reason in entry["reasons"]]
    assert any("duplicate entry" in reason for reason in reasons)


def test_boolean_receipt_count_is_invalid() -> None:
    """true == 1 in Python, so the type check, not the value, closes this."""
    from dilithium_py.ml_dsa import ML_DSA_65

    m = _minted()
    m["pack"]["receipt_count"] = True
    msg = vr.canonical_json(pv.signed_bundle_payload(m["pack"]))
    m["pack"].update(
        bundle_digest="sha256:" + hashlib.sha256(msg).hexdigest(),
        bundle_signature=base64.b64encode(ML_DSA_65.sign(m["sk"], msg)).decode(),
    )
    report = pv.verify_pack(m["pack"], jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["bundle"]["status"] == pv.INVALID
    assert "receipt_count" in report["bundle"]["reason"]


def test_dangling_mapping_reference_in_a_signed_pack_is_invalid() -> None:
    """Mapping edited, bundle re-signed: only the reference is wrong."""
    m = _minted()
    m["pack"]["regime_mapping"] = {"dora": ["sig_ghost"]}
    _seal(m["pack"], m["sk"], m["pk"])
    report = pv.verify_pack(m["pack"], jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["bundle"]["status"] == pv.INVALID
    assert "sig_ghost" in report["bundle"]["reason"]


def _resign(pack: dict, sk) -> None:
    from dilithium_py.ml_dsa import ML_DSA_65

    msg = vr.canonical_json(pv.signed_bundle_payload(pack))
    pack.update(
        bundle_digest="sha256:" + hashlib.sha256(msg).hexdigest(),
        bundle_signature=base64.b64encode(ML_DSA_65.sign(sk, msg)).decode(),
    )


@pytest.mark.parametrize(
    "case",
    json.loads(
        (Path(__file__).parents[2] / "conformance" / "pack-parity-cases.json").read_text()
    )["cases"],
    ids=lambda case: case["name"],
)
def test_shared_parity_vector(case) -> None:
    """The same vector file drives the Python and TypeScript pack verifiers."""
    m = _minted()
    pack = m["pack"]
    if case["mutation"] == "duplicate_signature_id":
        pack["receipts"][1]["signature_id"] = pack["receipts"][0]["signature_id"]
    elif case["mutation"] == "receipt_count_boolean":
        pack["receipt_count"] = True
    elif case["mutation"] == "dangling_mapping_reference":
        pack["regime_mapping"] = {"dora": ["sig_ghost"]}
    else:
        pytest.fail(f"unknown mutation {case['mutation']}")
    _resign(pack, m["sk"])
    report = pv.verify_pack(pack, jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["status"] == case["expected"]


def test_compromised_bundle_key_is_invalid() -> None:
    m = _minted()
    m["jwks"]["keys"][0].update(
        status="revoked", revocation_reason="compromised", revoked_at="2027-01-01T00:00:00Z"
    )
    report = pv.verify_pack(m["pack"], jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["bundle"]["status"] == pv.INVALID, report["bundle"]
    assert report["bundle"]["reason"] == "bundle signing key compromised"
    assert report["exit_code"] == 1


def test_unknown_bundle_key_status_is_not_verifiable() -> None:
    m = _minted()
    m["jwks"]["keys"][0]["status"] = "unknown"
    report = pv.verify_pack(m["pack"], jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["bundle"]["status"] == pv.NOT_OFFLINE, report["bundle"]
    assert report["bundle"]["reason"] == "bundle signing key status unknown"
    assert report["exit_code"] == 2


def test_docs_say_what_tsa_key_takes_and_that_todays_packs_cannot_exit_0() -> None:
    doc = (Path(__file__).parents[2] / "docs" / "offline-verification.md").read_text()
    body = doc.split("## Whole Audit Pack offline")[1].split("## TypeScript")[0]
    section = " ".join(body.split())
    assert "the TSA's own signing certificate or public key, not a root CA" in section
    assert "A pack exported today cannot reach exit code `0`" in section


def test_revoked_bundle_key_in_the_pack_snapshot_is_never_verified() -> None:
    m = _minted()
    m["pack"]["jwks"]["keys"][0]["status"] = "revoked"
    _seal(m["pack"], m["sk"], m["pk"])
    report = pv.verify_pack(m["pack"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["bundle"]["status"] == pv.NOT_OFFLINE
    assert report["summary"]["verified"] == 0 and report["exit_code"] == 2


def test_unknown_bundle_key_is_invalid() -> None:
    from dilithium_py.ml_dsa import ML_DSA_65

    m = _minted()
    _seal(m["pack"], *reversed(ML_DSA_65.keygen()))
    report = pv.verify_pack(m["pack"], jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["bundle"]["reason"] == "bundle key is not in the key set"
    assert report["summary"]["verified"] == 0 and report["exit_code"] == 1


def test_bundle_key_of_another_org_is_invalid() -> None:
    m = _minted()
    m["pack"]["organization_id"] = OTHER_ORG
    _seal(m["pack"], m["sk"], m["pk"])
    report = pv.verify_pack(m["pack"], jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["bundle"]["reason"] == "bundle signing key belongs to another organization"
    assert report["exit_code"] == 1


def test_receipt_missing_from_the_middle_of_a_chain_is_invalid() -> None:
    m = _minted()
    del m["pack"]["receipts"][1]
    _seal(m["pack"], m["sk"], m["pk"])
    report = pv.verify_pack(m["pack"], jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    third = report["receipts"][1]
    assert third["status"] == pv.INVALID
    assert third["reasons"][0] == "chain: gap, the predecessor of seq 3 is missing from the pack"
    assert report["exit_code"] == 1


def test_forked_chain_is_invalid() -> None:
    m = _minted()
    forked = copy.deepcopy(m["pack"]["receipts"][1]["payload"])
    forked["decision"] = "deny"
    m["pack"]["receipts"].insert(1, _sign_entry(m["sk"], forked, 2)[0])
    _seal(m["pack"], m["sk"], m["pk"])
    report = pv.verify_pack(m["pack"], jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    for entry in report["receipts"][1:3]:
        assert entry["status"] == pv.INVALID
        assert "chain: fork, seq 2 appears twice for agent agt_minted" in entry["reasons"]
    assert report["exit_code"] == 1


def test_duplicate_entry_is_invalid() -> None:
    m = _minted()
    m["pack"]["receipts"].append(copy.deepcopy(m["pack"]["receipts"][2]))
    _seal(m["pack"], m["sk"], m["pk"])
    report = pv.verify_pack(m["pack"], jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["receipts"][3]["status"] == pv.INVALID
    assert "duplicate entry: signature_id sig_3_allow" in report["receipts"][3]["reasons"]
    assert report["exit_code"] == 1


def test_receipts_are_not_verified_when_the_bundle_is_not() -> None:
    m = _minted()
    m["pack"]["bundle_signature"] = ""
    report = pv.verify_pack(m["pack"], jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["summary"]["verified"] == 0 and report["exit_code"] == 1
    assert report["receipts"][0]["reasons"] == [pv._UNVOUCHED_REASON]


def test_lapsed_expiry_signed_before_it_still_verifies() -> None:
    m = _minted(expires_at="2026-06-20T00:00:00.000000Z")
    report = pv.verify_pack(m["pack"], jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["exit_code"] == 0, report["receipts"]
    assert all("not a failure" in r["notes"][0] for r in report["receipts"])


def test_tampered_tsa_trust_breaks_the_bundle() -> None:
    m = _minted(carry_tsa=True)
    m["pack"]["tsa_trust"] = m["pack"]["tsa_trust"][::-1]
    assert pv.verify_pack(m["pack"], jwks=m["jwks"])["bundle"]["status"] == pv.INVALID


def test_jwks_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("[]")
    proc = _run_cli(tmp_path, _real_pack(), "--jwks", str(tmp_path / "bad.json"))
    assert proc.returncode == 1 and "Error reading input" in proc.stdout


def test_pack_dict_holding_an_integer_past_2_53_is_not_verifiable_offline() -> None:
    # Raw-text ingest refuses such a pack; the dict API must too.
    m = _minted()
    tsa = _tsa_bytes(m["tsa"])
    assert pv.verify_pack(m["pack"], jwks=m["jwks"], tsa_keys=tsa)["status"] == pv.VERIFIED
    pack = copy.deepcopy(m["pack"])
    pack["note"] = 2**53 + 1  # outside the bundle digest, so only the range check catches it
    report = pv.verify_pack(pack, jwks=m["jwks"], tsa_keys=tsa)
    assert report["status"] == pv.NOT_OFFLINE and report["exit_code"] == 2
    assert "+/-2**53" in report["bundle"]["reason"]
    entry = {**m["pack"]["receipts"][0], "note": -(2**53) - 1}
    assert pv.verify_entry(entry, m["jwks"], tsa_keys=tsa)["status"] == pv.NOT_OFFLINE


def test_dangling_mapping_reference_invalidates_the_pack() -> None:
    m = _minted()
    pack = copy.deepcopy(m["pack"])
    pack["regime_mapping"] = {"dora": ["sig_ghost"]}
    report = pv.verify_pack(pack, jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["status"] == pv.INVALID and report["exit_code"] == 1
    assert "sig_ghost" in report["bundle"]["reason"]


def test_receipt_count_mismatch_invalidates_the_pack() -> None:
    m = _minted()
    pack = copy.deepcopy(m["pack"])
    pack["receipt_count"] = len(pack["receipts"]) + 1
    report = pv.verify_pack(pack, jwks=m["jwks"], tsa_keys=_tsa_bytes(m["tsa"]))
    assert report["status"] == pv.INVALID and report["exit_code"] == 1
    assert "receipt_count" in report["bundle"]["reason"]
