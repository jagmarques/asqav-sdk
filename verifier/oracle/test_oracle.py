"""Oracle multi-format verification tests.

Each adapter must PASS a valid receipt and reject a tampered one (signature flip
and chain break). The conformance corpus runs end-to-end and every vector must
match its manifest outcome. Ed25519 is verified via the cryptography library; if
that library is absent the signature axis SKIPs and the signature-bearing
assertions are skipped rather than failing for the wrong reason.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_VERIFIER = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _VERIFIER)

from oracle import ADAPTERS, crypto, verify  # noqa: E402
from oracle.adapters.aerf import AerfAdapter  # noqa: E402
from oracle.adapters.agentreceipts import AgentReceiptsAdapter  # noqa: E402
from oracle.adapters.asqav_native import AsqavNativeAdapter  # noqa: E402
from oracle.runner import run_corpus  # noqa: E402

_CORPUS = Path(_VERIFIER) / "conformance-vectors"

_ED25519_AVAILABLE = crypto.verify_ed25519(b"\x00" * 32, b"m", b"\x00" * 64)[0] != crypto.SKIPPED

requires_ed25519 = pytest.mark.skipif(
    not _ED25519_AVAILABLE, reason="cryptography not installed; Ed25519 verify SKIPs"
)


def _load(vec: str, name: str) -> dict:
    return json.loads((_CORPUS / vec / name).read_text())


def _provider(vec: str, fmt: str):
    fname = "jwks.json" if fmt == "asqav-native" else "keys.json"
    return _load(vec, fname)


def test_adapters_registered() -> None:
    names = [a.name for a in ADAPTERS]
    assert names == ["asqav-native", "aerf", "acta", "agentreceipts"]


def test_detection_picks_the_right_adapter() -> None:
    aerf = _load("aerf-01-genesis", "receipt.json")
    native = _load("asqav-01-genesis-permit", "receipt.json")
    assert AerfAdapter().detect(aerf) is True
    assert AerfAdapter().detect(native) is False
    assert AsqavNativeAdapter().detect(native) is True
    assert AsqavNativeAdapter().detect(aerf) is False


@requires_ed25519
def test_aerf_valid_receipt_passes() -> None:
    doc = _load("aerf-01-genesis", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("aerf-01-genesis", "aerf"))
    assert res.fmt == "aerf"
    assert res.verdict == "PASS"
    assert res.axis("signature").result == crypto.PASS


@requires_ed25519
def test_aerf_chain_link_passes() -> None:
    doc = _load("aerf-02-chain-link", "receipt.json")
    pred = _load("aerf-02-chain-link", "predecessor.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("aerf-02-chain-link", "aerf"), predecessor=pred)
    assert res.verdict == "PASS"
    assert res.axis("chain").result == crypto.PASS


@requires_ed25519
def test_aerf_tampered_evidence_fails_signature() -> None:
    doc = _load("aerf-03-tamper-evidence", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("aerf-03-tamper-evidence", "aerf"))
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


@requires_ed25519
def test_aerf_tampered_chain_fails_chain() -> None:
    doc = _load("aerf-04-tamper-chain", "receipt.json")
    pred = _load("aerf-04-tamper-chain", "predecessor.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("aerf-04-tamper-chain", "aerf"), predecessor=pred)
    assert res.verdict == "FAIL"
    assert res.axis("chain").result == crypto.FAIL


@requires_ed25519
def test_aerf_impact_without_parent_sig_fails() -> None:
    """An impact receipt missing its required parent counter-signature FAILs that axis."""
    doc = _load("aerf-up-05-impact-no-parent-sig", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("aerf-up-05-impact-no-parent-sig", "aerf"))
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.PASS
    assert res.axis("parent_signature").result == crypto.FAIL


@requires_ed25519
def test_aerf_pdp_split_context_fails() -> None:
    """A PDP verdict signed for one context cannot bind a receipt claiming another."""
    doc = _load("aerf-up-08-pdp-binding-split-context", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("aerf-up-08-pdp-binding-split-context", "aerf"))
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.PASS
    assert res.axis("pdp_signature").result == crypto.FAIL


@requires_ed25519
def test_aerf_required_layer_without_key_is_incomplete_never_pass() -> None:
    """A required counter-sign axis whose key is not supplied SKIPs and downgrades to INCOMPLETE, never PASS."""
    doc = _load("aerf-up-07-pdp-binding-valid", "receipt.json")
    keys = dict(_provider("aerf-up-07-pdp-binding-valid", "aerf"))
    keys.pop("dfe937603b2f3312", None)  # drop the PDP key so the axis cannot be checked
    res = verify(doc, ADAPTERS, key_provider=keys)
    assert res.axis("pdp_signature").result == crypto.SKIPPED
    assert res.verdict == "INCOMPLETE"


@requires_ed25519
def test_asqav_native_valid_receipt_passes() -> None:
    doc = _load("asqav-01-genesis-permit", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("asqav-01-genesis-permit", "asqav-native"))
    assert res.fmt == "asqav-native"
    assert res.verdict == "PASS"
    assert res.axis("signature").result == crypto.PASS


@requires_ed25519
def test_asqav_native_chain_link_passes() -> None:
    doc = _load("asqav-03-chain-link", "receipt.json")
    pred = _load("asqav-03-chain-link", "predecessor.json")
    res = verify(
        doc, ADAPTERS, key_provider=_provider("asqav-03-chain-link", "asqav-native"), predecessor=pred
    )
    assert res.verdict == "PASS"
    assert res.axis("chain").result == crypto.PASS


@requires_ed25519
def test_asqav_native_tampered_signature_fails() -> None:
    doc = _load("asqav-04-tamper-sig", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("asqav-04-tamper-sig", "asqav-native"))
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


def test_asqav_native_chain_break_fails_without_crypto() -> None:
    """A broken chain link FAILs the chain axis independent of the signature dep."""
    doc = _load("asqav-01-genesis-permit", "receipt.json")
    successor = json.loads(json.dumps(doc))
    successor["payload"]["previousReceiptHash"] = "f" * 64
    pred = _load("asqav-01-genesis-permit", "receipt.json")
    res = verify(successor, ADAPTERS, predecessor=pred)
    assert res.axis("chain").result == crypto.FAIL


def test_unknown_format_fails_closed() -> None:
    res = verify({"hello": "world"}, ADAPTERS)
    assert res.fmt == "unknown"
    assert res.verdict == "FAIL"


def test_signature_skips_downgrade_to_incomplete() -> None:
    """No key provider means the signature cannot be checked; never a PASS."""
    doc = _load("aerf-01-genesis", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider={})
    assert res.axis("signature").result == crypto.SKIPPED
    assert res.verdict == "INCOMPLETE"


@requires_ed25519
def test_corpus_runs_and_every_vector_matches() -> None:
    results = run_corpus(_CORPUS)
    assert len(results) == 37
    failures = [r for r in results if not r.ok]
    assert not failures, f"corpus mismatches: {[(r.dir, r.actual_verdict) for r in failures]}"


# --- ACTA adapter + cross-format confusion ---

from oracle.adapters.acta import ActaAdapter  # noqa: E402


def _acta_keys(vec: str):
    return _load(vec, "acta-keys.json")


@requires_ed25519
def test_acta_valid_genesis_passes() -> None:
    doc = _load("acta-01-genesis", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_acta_keys("acta-01-genesis"))
    assert res.fmt == "acta"
    assert res.verdict == "PASS"
    assert res.axis("signature").result == crypto.PASS


@requires_ed25519
def test_acta_chain_link_passes() -> None:
    doc = _load("acta-02-chain-link", "receipt.json")
    pred = _load("acta-02-chain-link", "predecessor.json")
    res = verify(doc, ADAPTERS, key_provider=_acta_keys("acta-02-chain-link"), predecessor=pred)
    assert res.verdict == "PASS"
    assert res.axis("chain").result == crypto.PASS


@requires_ed25519
def test_acta_tampered_signature_fails() -> None:
    doc = _load("acta-03-tamper-sig", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_acta_keys("acta-03-tamper-sig"))
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


@requires_ed25519
def test_acta_commitment_mode_fails_not_false_passes() -> None:
    """A commitment-mode receipt (signs SHA-256(JCS)) must FAIL the baseline verifier."""
    doc = _load("acta-05-commitment-mode-unsupported", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_acta_keys("acta-05-commitment-mode-unsupported"))
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


def test_acta_and_asqav_native_are_mutually_exclusive() -> None:
    """Asqav-native profiles ACTA, so detection must never let both claim one receipt."""
    acta = _load("acta-01-genesis", "receipt.json")
    native = _load("asqav-01-genesis-permit", "receipt.json")
    assert ActaAdapter().detect(acta) is True
    assert AsqavNativeAdapter().detect(acta) is False
    assert AsqavNativeAdapter().detect(native) is True
    assert ActaAdapter().detect(native) is False


@requires_ed25519
def test_cross_format_confusion_asqav_with_hex_sig_does_not_verify_as_native() -> None:
    """An Asqav-shaped receipt with a hex sig routes to ACTA and FAILs - never a false native PASS."""
    native = _load("asqav-01-genesis-permit", "receipt.json")
    forged = json.loads(json.dumps(native))
    forged.pop("anchors", None)
    forged["signature"]["sig"] = "ab" * 64  # 128-char lowercase hex (ACTA-shaped)
    res = verify(forged, ADAPTERS, key_provider=_acta_keys("acta-01-genesis"))
    assert res.fmt == "acta"
    assert res.verdict != "PASS"
    # The unmodified native receipt still routes to asqav-native.
    assert verify(native, ADAPTERS, key_provider=_provider("asqav-01-genesis-permit", "asqav-native")).fmt == "asqav-native"


@requires_ed25519
def test_acta_cross_format_predecessor_fails_chain() -> None:
    """An ACTA successor with an asqav-native predecessor is not a valid chain link."""
    succ = _load("acta-02-chain-link", "receipt.json")
    foreign_pred = _load("asqav-01-genesis-permit", "receipt.json")
    res = verify(succ, ADAPTERS, key_provider=_acta_keys("acta-02-chain-link"), predecessor=foreign_pred)
    assert res.axis("chain").result == crypto.FAIL


@requires_ed25519
def test_acta_genesis_spoof_breaks_signature() -> None:
    """Deleting previousReceiptHash to fake genesis breaks the signature (it is signed)."""
    succ = _load("acta-02-chain-link", "receipt.json")
    spoof = json.loads(json.dumps(succ))
    del spoof["payload"]["previousReceiptHash"]
    res = verify(spoof, ADAPTERS, key_provider=_acta_keys("acta-02-chain-link"))
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


# --- adversarial negatives on the trust path ---

import unicodedata  # noqa: E402

import verify_receipt as _vr  # noqa: E402


@requires_ed25519
def test_wrong_key_resolution_fails_signature() -> None:
    """A receipt verified against a different valid Ed25519 key FAILs the signature axis."""
    doc = _load("aerf-01-genesis", "receipt.json")
    kid = doc["key_id"]
    other_raw = _vr._b64decode(_acta_keys("acta-01-genesis")["keys"][0]["x"])
    swapped = dict(_provider("aerf-01-genesis", "aerf"))
    swapped[kid] = other_raw.hex()
    res = verify(doc, ADAPTERS, key_provider=swapped)
    assert res.fmt == "aerf"
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


@requires_ed25519
def test_algorithm_confusion_does_not_false_pass() -> None:
    """An Ed25519 receipt relabelled to an alg the verifier does not check never PASSes."""
    doc = _load("acta-01-genesis", "receipt.json")
    forged = json.loads(json.dumps(doc))
    forged["signature"]["alg"] = "ES256"
    res = verify(forged, ADAPTERS, key_provider=_acta_keys("acta-01-genesis"))
    assert res.fmt == "acta"
    assert res.verdict != "PASS"
    assert res.axis("structure").result == crypto.FAIL
    assert res.axis("signature").result in (crypto.FAIL, crypto.SKIPPED)


@requires_ed25519
def test_chain_reorder_same_format_fails_chain() -> None:
    """A same-format predecessor that is not the true prior link FAILs the chain axis."""
    succ = _load("acta-02-chain-link", "receipt.json")
    wrong_pred = _load("acta-03-tamper-sig", "receipt.json")
    assert ActaAdapter().detect(wrong_pred) is True
    res = verify(
        succ, ADAPTERS, key_provider=_acta_keys("acta-02-chain-link"), predecessor=wrong_pred
    )
    assert res.axis("chain").result == crypto.FAIL


@requires_ed25519
def test_nfc_canonicalization_edge() -> None:
    """AERF signs over NFC-normalising JCS, so NFC and NFD forms verify identically."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    nfc = unicodedata.normalize("NFC", "café-π")
    nfd = unicodedata.normalize("NFD", "café-π")
    assert nfc.encode() != nfd.encode()

    sk = Ed25519PrivateKey.generate()
    pk_raw = sk.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    kid = "oracle-nfc-test-key"
    ad = AerfAdapter()
    rec = _load("aerf-01-genesis", "receipt.json")
    rec["agent"] = nfc
    rec["key_id"] = kid
    rec.pop("signature", None)
    rec["signature"] = sk.sign(ad.signing_input(rec)).hex()
    provider = {kid: pk_raw.hex()}

    nfc_res = verify(rec, ADAPTERS, key_provider=provider)
    assert nfc_res.axis("signature").result == crypto.PASS

    nfd_doc = json.loads(json.dumps(rec))
    nfd_doc["agent"] = nfd
    nfd_res = verify(nfd_doc, ADAPTERS, key_provider=provider)
    assert nfd_res.axis("signature").result == crypto.PASS

    tampered = json.loads(json.dumps(rec))
    tampered["agent"] = "different-agent"
    assert verify(tampered, ADAPTERS, key_provider=provider).axis("signature").result == crypto.FAIL


def test_four_way_format_exclusion() -> None:
    """Each adapter detects only its own receipt, giving a clean 4x4 detection diagonal."""
    native = _load("asqav-01-genesis-permit", "receipt.json")
    aerf = _load("aerf-01-genesis", "receipt.json")
    acta = _load("acta-01-genesis", "receipt.json")
    ar = _load("agentreceipts-01-didkey-genesis", "receipt.json")
    a, e, c, g = AsqavNativeAdapter(), AerfAdapter(), ActaAdapter(), AgentReceiptsAdapter()
    assert (a.detect(native), e.detect(native), c.detect(native), g.detect(native)) == (True, False, False, False)
    assert (a.detect(aerf), e.detect(aerf), c.detect(aerf), g.detect(aerf)) == (False, True, False, False)
    assert (a.detect(acta), e.detect(acta), c.detect(acta), g.detect(acta)) == (False, False, True, False)
    assert (a.detect(ar), e.detect(ar), c.detect(ar), g.detect(ar)) == (False, False, False, True)


# --- agent-receipts adapter (W3C-VC AgentReceipt) + upstream interop ---

from oracle.canonical import jcs_rfc8785  # noqa: E402
from oracle.did import b58btc_decode, resolve_ed25519_key  # noqa: E402

_INTEROP = _CORPUS / "agentreceipts-upstream-interop"


def _ar(vec: str, name: str = "receipt.json") -> dict:
    return _load(vec, name)


def _ar_provider(vec: str):
    path = _CORPUS / vec / "did_map.json"
    return json.loads(path.read_text()) if path.exists() else None


@requires_ed25519
def test_agentreceipts_didkey_genesis_passes() -> None:
    """A did:key genesis verifies: the key resolves inline and the signature checks."""
    doc = _ar("agentreceipts-01-didkey-genesis")
    res = verify(doc, ADAPTERS)
    assert res.fmt == "agentreceipts"
    assert res.verdict == "PASS"
    assert res.axis("signature").result == crypto.PASS


@requires_ed25519
def test_agentreceipts_chain_link_passes() -> None:
    doc = _ar("agentreceipts-02-didkey-chain-link")
    pred = _ar("agentreceipts-02-didkey-chain-link", "predecessor.json")
    res = verify(doc, ADAPTERS, predecessor=pred)
    assert res.verdict == "PASS"
    assert res.axis("chain").result == crypto.PASS


@requires_ed25519
def test_agentreceipts_tampered_payload_fails_signature() -> None:
    doc = _ar("agentreceipts-03-tamper-payload")
    res = verify(doc, ADAPTERS)
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


@requires_ed25519
def test_agentreceipts_tampered_proofvalue_fails_signature() -> None:
    doc = _ar("agentreceipts-04-tamper-proofvalue")
    res = verify(doc, ADAPTERS)
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


@requires_ed25519
def test_agentreceipts_issuer_must_control_signing_key_no_impersonation() -> None:
    """An attacker's valid signature under a key the issuer does not control must not verify."""
    from base64 import urlsafe_b64encode

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    vm = "did:web:attacker.example#key-1"
    forged = json.loads(json.dumps(_ar("agentreceipts-01-didkey-genesis")))
    forged["issuer"] = {"id": "did:web:victim.example"}
    forged["proof"]["verificationMethod"] = vm
    sig = sk.sign(AgentReceiptsAdapter().signing_input(forged))
    forged["proof"]["proofValue"] = "u" + urlsafe_b64encode(sig).decode().rstrip("=")
    res = verify(forged, ADAPTERS, key_provider={vm: pk.hex()})
    assert res.axis("signature").result == crypto.PASS  # attacker's sig is cryptographically valid
    assert res.axis("structure").result == crypto.FAIL  # but signing-key DID != issuer
    assert res.verdict == "FAIL"


def test_agentreceipts_genesis_explicit_null_is_genesis_missing_field_is_malformed() -> None:
    """Genesis carries previous_receipt_hash present and null; omitting it is malformed."""
    genesis = _ar("agentreceipts-01-didkey-genesis")
    ad = AgentReceiptsAdapter()
    assert ad.chain_step(genesis).is_genesis is True
    assert ad.schema(genesis)[0] == crypto.PASS

    missing = _ar("agentreceipts-05-genesis-missing-prev-hash")
    assert ad.schema(missing)[0] == crypto.FAIL
    assert verify(missing, ADAPTERS).verdict == "FAIL"


@requires_ed25519
def test_agentreceipts_wrong_did_fails_signature() -> None:
    """A verificationMethod resolved to a different key cannot verify the signature."""
    doc = _ar("agentreceipts-06-wrong-key")
    res = verify(doc, ADAPTERS, key_provider=_ar_provider("agentreceipts-06-wrong-key"))
    assert res.fmt == "agentreceipts"
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


@requires_ed25519
def test_agentreceipts_upstream_valid_resigned_passes() -> None:
    """The upstream malformed-corpus base receipt, re-signed clean with the upstream key."""
    doc = _ar("agentreceipts-up-00-valid-resigned")
    res = verify(doc, ADAPTERS, key_provider=_ar_provider("agentreceipts-up-00-valid-resigned"))
    assert res.verdict == "PASS"
    assert res.axis("signature").result == crypto.PASS


def test_jcs_rfc8785_byte_matches_upstream_canonicalization_vectors() -> None:
    """Gold interop: our jcs_rfc8785 output must byte-equal every upstream canonical form."""
    path = _INTEROP / "canonicalization_vectors.json"
    if not path.exists():
        pytest.skip("upstream canonicalization vectors not vendored")
    data = json.loads(path.read_text())
    mismatches = []
    for v in data["canonicalization_vectors"]:
        got = jcs_rfc8785(v["input"]).decode("utf-8")
        if got != v["canonical"]:
            mismatches.append((v["name"], v["canonical"], got))
    assert not mismatches, f"JCS interop mismatches vs upstream: {mismatches}"


def test_didkey_resolver_decodes_upstream_did_key_vectors() -> None:
    """The shared DID resolver decodes each upstream did:key to its expected raw key."""
    path = _INTEROP / "did_key_vectors.json"
    if not path.exists():
        pytest.skip("upstream did:key vectors not vendored")
    data = json.loads(path.read_text())
    for v in data["vectors"]:
        did = v["did"]
        raw, _note = resolve_ed25519_key(did + "#" + did.split(":")[-1])
        assert raw is not None, f"{v['name']} did not resolve"
        assert raw.hex() == v["public_key_hex"], f"{v['name']} key mismatch"


def test_b58btc_decode_known_value() -> None:
    """base58btc decodes a known multikey frame: 0xed01 prefix + 32 key bytes."""
    # did:key vector-1 identifier without the multibase 'z' prefix.
    decoded = b58btc_decode("6MktwupdmLXVVqTzCw4i46r4uGyosGXRnR3XjN4Zq7oMMsw")
    assert decoded[:2] == b"\xed\x01"
    assert decoded[2:].hex() == "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"


# --- Asqav-native hash mode + real-prod ground truth ---

from oracle.canonical import asqav_jcs  # noqa: E402

_PROD_HASH = _CORPUS / "asqav-05-hash-mode-prod"


def test_asqav_jcs_byte_matches_verify_receipt_canonical_json() -> None:
    """Cloud-parity anchor: the native canonicaliser equals the cloud signer's bytes."""
    samples = [
        {"b": 2, "a": 1, "z": {"y": 3, "x": [1, 2]}},
        {"mode": "hash", "v": 1, "metadata": {}, "policy_digest": None},
        _ar("agentreceipts-01-didkey-genesis"),
    ]
    for obj in samples:
        assert asqav_jcs(obj) == _vr.canonical_json(obj)


def test_hash_mode_receipt_routes_to_asqav_native() -> None:
    """A real prod hash-mode receipt detects as asqav-native and no other adapter claims it."""
    doc = _load("asqav-05-hash-mode-prod", "receipt.json")
    a, e, c, g = AsqavNativeAdapter(), AerfAdapter(), ActaAdapter(), AgentReceiptsAdapter()
    assert (a.detect(doc), e.detect(doc), c.detect(doc), g.detect(doc)) == (True, False, False, False)
    assert verify(doc, ADAPTERS, key_provider=_provider("asqav-05-hash-mode-prod", "asqav-native")).fmt == "asqav-native"


def test_hash_mode_signing_input_byte_matches_prod_message() -> None:
    """The reconstructed hash-mode signing input is byte-identical to the prod-signed bytes."""
    doc = _load("asqav-05-hash-mode-prod", "receipt.json")
    ground_truth = (_PROD_HASH / "signed_message.bytes").read_bytes()
    rebuilt = AsqavNativeAdapter().signing_input(doc)
    assert rebuilt == ground_truth
    import hashlib

    assert hashlib.sha256(rebuilt).hexdigest() == (
        "f1011f5e493bf8f3f6fa7baa660aab6be2e0bbf925d852672e67155f08b4f0b3"
    )


def test_hash_mode_signature_skips_without_dilithium_never_false_pass() -> None:
    """Without dilithium-py the ML-DSA-65 axis SKIPs; the verdict is INCOMPLETE, never PASS."""
    doc = _load("asqav-05-hash-mode-prod", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("asqav-05-hash-mode-prod", "asqav-native"))
    assert res.fmt == "asqav-native"
    assert res.axis("structure").result == crypto.PASS
    assert res.axis("signature").result in (crypto.PASS, crypto.SKIPPED)
    assert res.verdict in ("PASS", "INCOMPLETE")
    assert res.verdict != "FAIL"


def test_hash_mode_malformed_signature_fails_does_not_crash() -> None:
    """A malformed hash-mode signature (non-string) verifies to a non-PASS, never raises."""
    doc = _load("asqav-05-hash-mode-prod", "receipt.json")
    for bad in ({"not": "a string"}, 12345, ["x"]):
        forged = json.loads(json.dumps(doc))
        forged.pop("signature_b64", None)
        forged["signature"] = bad
        res = verify(forged, ADAPTERS, key_provider=_provider("asqav-05-hash-mode-prod", "asqav-native"))
        assert res.verdict != "PASS"
