"""Oracle multi-format verification tests.

Each adapter must PASS a valid receipt and reject a tampered one (signature flip
and chain break). The conformance corpus runs end-to-end and every vector must
match its manifest outcome. Ed25519 is verified via the cryptography library; if
that library is absent the signature axis SKIPs and the signature-bearing
assertions are skipped rather than failing for the wrong reason.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from asqav.verifier.oracle import ADAPTERS, crypto, verify
from asqav.verifier.oracle.adapters.aerf import AerfAdapter
from asqav.verifier.oracle.adapters.agentreceipts import AgentReceiptsAdapter
from asqav.verifier.oracle.adapters.asqav_native import AsqavNativeAdapter
from asqav.verifier.oracle.runner import main as runner_main
from asqav.verifier.oracle.runner import run_corpus

# The corpus stays at the repo root (verifier/conformance-vectors) so the TS parity gate
# and the published governance URL keep one source of truth; walk up from tests to it.
_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"

_ED25519_AVAILABLE = crypto.verify_ed25519(b"\x00" * 32, b"m", b"\x00" * 64)[0] != crypto.SKIPPED

requires_ed25519 = pytest.mark.skipif(
    not _ED25519_AVAILABLE, reason="cryptography not installed; Ed25519 verify SKIPs"
)


def _load(vec: str, name: str) -> dict:
    return json.loads((_CORPUS / vec / name).read_text())


def _provider(vec: str, fmt: str):
    fname = "jwks.json" if fmt == "asqav-native" else "keys.json"
    return _load(vec, fname)


def test_verifier_ships_inside_the_asqav_package() -> None:
    """The verifier runtime lives under src/asqav so packages=[src/asqav] ships it in both sdist and wheel.

    A standalone force-include of ``../verifier`` could only reach the wheel and broke
    the sdist (the regression this layout fixes). Pin that the runtime modules sit on
    disk inside the package tree and that no cross-root force-include points back out.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    verifier_pkg = repo_root / "python" / "src" / "asqav" / "verifier"
    for rel in (
        "__init__.py",
        "verify_receipt.py",
        "oracle/__init__.py",
        "oracle/core.py",
        "oracle/crypto.py",
        "oracle/canonical.py",
        "oracle/runner.py",
        "oracle/adapters/__init__.py",
    ):
        assert (verifier_pkg / rel).exists(), f"missing packaged runtime module: {rel}"

    pyproject = repo_root / "python" / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject not present (running from an installed package)")
    text = pyproject.read_text()
    assert '"src/asqav"' in text  # the package selector that carries the verifier
    assert "../verifier" not in text  # no cross-root force-include that an sdist cannot reach


def test_adapters_registered() -> None:
    names = [a.name for a in ADAPTERS]
    assert names == ["asqav-native", "aerf", "acta", "agentreceipts", "authproof", "pipelock-evidence-v2"]


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
def test_runner_main_reports_all_green(capsys) -> None:
    """The CLI entrypoint runs the corpus, tolerates the optional-dep vector, exits 0."""
    rc = runner_main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "vectors matched expected outcome" in out
    assert "[FAIL]" not in out


@requires_ed25519
def test_corpus_runs_and_every_vector_matches() -> None:
    results = run_corpus(_CORPUS)
    assert len(results) == 46
    # The optional-dep ML-DSA vector returns the stronger PASS when dilithium-py is present.
    def _tol(r):
        return r.ok or (
            r.expected_outcome == "INCOMPLETE"
            and r.actual_verdict == "PASS"
            and r.reason_code == "signature_skipped_no_dilithium"
        )

    failures = [(r.dir, r.actual_verdict) for r in results if not _tol(r)]
    assert not failures, f"corpus mismatches: {failures}"


# --- ACTA adapter + cross-format confusion ---

from asqav.verifier.oracle.adapters.acta import ActaAdapter  # noqa: E402


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


@requires_ed25519
def test_acta_upstream_scopeblind_receipt_verifies() -> None:
    """A REAL ScopeBlind/APS attestation verifies: genuine inbound interop.

    The receipt under acta-up-01 is lifted verbatim from the
    ScopeBlind/agent-governance-testvectors corpus (a2a-trust-header/happy-path.json):
    its Ed25519 signature was produced by the upstream reference issuer over
    JCS(payload), not by us. Its payload carries `issuer` (not the Asqav-profile
    `issuer_id`) and omits `type`, so this asserts our adapter accepts a conformant
    third-party producer rather than only Asqav-shaped receipts.
    """
    doc = _load("acta-up-01-a2a-trusted-attestation", "receipt.json")
    assert "issuer_id" not in doc["payload"] and "type" not in doc["payload"]
    res = verify(doc, ADAPTERS, key_provider=_acta_keys("acta-up-01-a2a-trusted-attestation"))
    assert res.fmt == "acta"
    assert res.verdict == "PASS"
    assert res.axis("signature").result == crypto.PASS


@requires_ed25519
def test_acta_upstream_tampered_receipt_fails() -> None:
    """The real upstream receipt with one signature nibble flipped MUST fail."""
    doc = _load("acta-up-03-a2a-tampered-sig", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_acta_keys("acta-up-03-a2a-tampered-sig"))
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


def test_acta_malformed_signature_fails_does_not_crash() -> None:
    """A malformed ACTA sig fails closed, never raises - at detection and in extract_signature.

    Through the public verify(), a non-hex sig is declined at detection so the verdict
    is FAIL with no adapter matched. The extract_signature guard is the defense in depth:
    called directly with a non-hex or non-string sig it returns b'' rather than raising,
    matching the other adapters.
    """
    doc = _load("acta-01-genesis", "receipt.json")
    for bad in ("zzzz-not-hex", {"k": "v"}, 42, None):
        forged = json.loads(json.dumps(doc))
        forged["signature"] = {"sig": bad, "alg": "EdDSA", "kid": "k1"}
        res = verify(forged, ADAPTERS, key_provider=_acta_keys("acta-01-genesis"))
        assert res.verdict != "PASS"
        assert ActaAdapter().extract_signature(forged).sig == b""


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

from asqav.verifier import verify_receipt as _vr  # noqa: E402


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

from asqav.verifier.oracle.canonical import jcs_rfc8785  # noqa: E402
from asqav.verifier.oracle.did import b58btc_decode, resolve_ed25519_key  # noqa: E402

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

from asqav.verifier.oracle.canonical import asqav_jcs  # noqa: E402

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


def test_compliance_envelope_malformed_signature_fails_does_not_crash() -> None:
    """A malformed Asqav compliance-envelope signature decodes to b'' and FAILs, never raises."""
    doc = _load("asqav-01-genesis-permit", "receipt.json")
    for bad in ({"k": "v"}, 999, None, "not base64 @@@"):
        forged = json.loads(json.dumps(doc))
        forged["signature"]["sig"] = bad
        res = verify(forged, ADAPTERS, key_provider=_provider("asqav-01-genesis-permit", "asqav-native"))
        assert res.verdict != "PASS"


def test_aerf_malformed_signature_fails_does_not_crash() -> None:
    """A malformed AERF signature (non-hex or non-string) decodes to b'' and FAILs, never raises."""
    doc = _load("aerf-01-genesis", "receipt.json")
    for bad in ("zzzz-not-hex", {"k": "v"}, 42, None):
        forged = json.loads(json.dumps(doc))
        forged["signature"] = bad
        res = verify(forged, ADAPTERS, key_provider=_provider("aerf-01-genesis", "aerf"))
        assert res.verdict != "PASS"


def test_aerf_malformed_parent_pdp_signature_fails_does_not_crash() -> None:
    """Malformed AERF parent/pdp counter-signatures decode to b'' and FAIL, never raise."""
    vec = "aerf-up-06-impact-with-parent-sig"
    doc = _load(vec, "receipt.json")
    for field in ("parent_signature", "pdp_signature"):
        for bad in ("zz", "abc", "not-hex"):
            forged = json.loads(json.dumps(doc))
            forged[field] = bad
            res = verify(forged, ADAPTERS, key_provider=_provider(vec, "aerf"))
            assert res.verdict != "PASS"


# --- Authproof adapter (ES256 over insertion-order JSON.stringify) ---

from asqav.verifier.oracle.adapters.authproof import AuthproofAdapter  # noqa: E402


def _authproof_receipt() -> dict:
    return json.loads((_CORPUS / "authproof-01-genesis-real-sdk" / "receipt.json").read_text())


@requires_ed25519
def test_authproof_real_sdk_receipt_verifies() -> None:
    """A receipt minted by the real Authproof JS SDK verifies (cross-implementation interop)."""
    res = verify(_authproof_receipt(), ADAPTERS)
    assert res.fmt == "authproof"
    assert res.verdict == "PASS"
    assert res.axis("signature").result == crypto.PASS


def test_authproof_detection_is_exclusive() -> None:
    """The Authproof fingerprint matches only its own receipt, not the other formats."""
    doc = _authproof_receipt()
    assert [a.name for a in ADAPTERS if a.detect(doc)] == ["authproof"]
    other = _load("aerf-01-genesis", "receipt.json")
    assert AuthproofAdapter().detect(other) is False


@requires_ed25519
def test_authproof_tamper_and_forge_fail_closed() -> None:
    """Tampering a signed field, forging the signature, or swapping the key all FAIL."""
    base = _authproof_receipt()
    tampered = json.loads(json.dumps(base))
    tampered["scope"] = "Send all the emails."
    assert verify(tampered, ADAPTERS).verdict == "FAIL"
    forged = json.loads(json.dumps(base))
    forged["signature"] = "25" + base["signature"][2:]
    assert verify(forged, ADAPTERS).verdict == "FAIL"
    wrong_key = json.loads(json.dumps(base))
    wrong_key["signerPublicKey"]["x"] = "AAAAXZILz_GGg8wwWuea3gOYkTvvYwzM42bgY5k_zgM"
    assert verify(wrong_key, ADAPTERS).verdict == "FAIL"


def test_authproof_malformed_signature_fails_does_not_crash() -> None:
    """A malformed Authproof signature decodes to b'' and FAILs, never raises."""
    base = _authproof_receipt()
    for bad in ("zz-not-hex", {"k": "v"}, 42, None, "abc"):
        forged = json.loads(json.dumps(base))
        forged["signature"] = bad
        assert verify(forged, ADAPTERS).verdict != "PASS"


def test_es256_malformed_key_and_sig_fail_closed() -> None:
    """ES256 verify returns FAIL (never raises) on a bad point or a wrong-length signature."""
    assert crypto.verify_es256(b"\x00" * 10, b"m", b"\x00" * 64)[0] == crypto.FAIL
    pk = AuthproofAdapter().resolve_key(_authproof_receipt(), None)[0]
    assert crypto.verify_es256(pk, b"m", b"\x00" * 10)[0] == crypto.FAIL


def test_non_dict_receipt_fails_closed_never_crashes() -> None:
    """A non-object receipt (array/string/scalar) matches no format and FAILs, never crashes."""
    from asqav.verifier.oracle.core import detect

    for bad in ([1, 2, 3], "a-string", 42, None, True):
        assert detect(bad, ADAPTERS) is None
        res = verify(bad, ADAPTERS)
        assert res.verdict != "PASS"


# --- Pipelock EvidenceReceipt v2 adapter ---
#
# Fixture: Go reference implementation output from luckyPipewrench/pipelock-verify-python
# commit 9eaff72a87b3b412945fac6de07739bc2bef2116 (tests/conformance/valid-evidence-proxy-decision.json).
# Signer public key: RFC 8032 section 7.1 test-1 vector key
# d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a (no secrets).

from asqav.verifier.oracle.adapters.pipelock import PipelockEvidenceAdapter  # noqa: E402

#: RFC 8032 test-1 public key (hex) - the Go reference uses this for all conformance fixtures.
_PIPELOCK_PUB_HEX = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
_PIPELOCK_KEY_PROVIDER = {"receipt-signing-test": _PIPELOCK_PUB_HEX}


def _pipelock(vec: str, name: str = "receipt.json") -> dict:
    return _load(vec, name)


@requires_ed25519
def test_pipelock_evidence_v2_valid_passes() -> None:
    """A real Go-reference proxy_decision EvidenceReceipt v2 verifies (inbound interop).

    The fixture is lifted verbatim from luckyPipewrench/pipelock-verify-python
    tests/conformance/valid-evidence-proxy-decision.json (commit 9eaff72). It was
    signed by the Go implementation over JCS(receipt with signature zeroed) using
    the RFC 8032 section 7.1 test-1 private key. This confirms our JCS preimage
    computation is byte-identical to the Go canonicaliser for this receipt shape.
    """
    doc = _pipelock("pipelock-ev2-01-proxy-decision")
    res = verify(doc, ADAPTERS, key_provider=_PIPELOCK_KEY_PROVIDER)
    assert res.fmt == "pipelock-evidence-v2"
    assert res.verdict == "PASS"
    assert res.axis("structure").result == crypto.PASS
    assert res.axis("signature").result == crypto.PASS


@requires_ed25519
def test_pipelock_evidence_v2_tampered_payload_fails() -> None:
    """A Go-signed receipt with payload.verdict changed from 'allow' to 'block' must FAIL.

    The original signature covers JCS(receipt-with-zeroed-sig) where verdict='allow'.
    Changing the payload shifts the preimage, so the Ed25519 signature no longer verifies.
    This is the non-vacuous tamper test: the signature axis itself must FAIL, not SKIPPED.
    """
    doc = _pipelock("pipelock-ev2-02-tamper-payload")
    res = verify(doc, ADAPTERS, key_provider=_PIPELOCK_KEY_PROVIDER)
    assert res.fmt == "pipelock-evidence-v2"
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


def test_pipelock_detection_is_exclusive() -> None:
    """The Pipelock fingerprint matches only its own receipts, not any other format."""
    pipelock_doc = _pipelock("pipelock-ev2-01-proxy-decision")
    aerf_doc = _load("aerf-01-genesis", "receipt.json")
    acta_doc = _load("acta-01-genesis", "receipt.json")
    native_doc = _load("asqav-01-genesis-permit", "receipt.json")
    authproof_doc = _load("authproof-01-genesis-real-sdk", "receipt.json")
    pl = PipelockEvidenceAdapter()
    assert pl.detect(pipelock_doc) is True
    assert pl.detect(aerf_doc) is False
    assert pl.detect(acta_doc) is False
    assert pl.detect(native_doc) is False
    assert pl.detect(authproof_doc) is False
    # The pipelock receipt is detected by exactly one adapter.
    assert [a.name for a in ADAPTERS if a.detect(pipelock_doc)] == ["pipelock-evidence-v2"]


@requires_ed25519
def test_pipelock_wrong_key_fails_signature() -> None:
    """A valid Pipelock receipt verified against a different Ed25519 key FAILs the signature axis."""
    doc = _pipelock("pipelock-ev2-01-proxy-decision")
    wrong_provider = {"receipt-signing-test": "00" * 32}
    res = verify(doc, ADAPTERS, key_provider=wrong_provider)
    assert res.fmt == "pipelock-evidence-v2"
    assert res.verdict == "FAIL"
    assert res.axis("signature").result == crypto.FAIL


def test_pipelock_missing_key_skips_never_passes() -> None:
    """No key for signer_key_id: signature axis SKIPs; verdict is INCOMPLETE, never PASS."""
    doc = _pipelock("pipelock-ev2-01-proxy-decision")
    res = verify(doc, ADAPTERS, key_provider={})
    assert res.fmt == "pipelock-evidence-v2"
    assert res.axis("signature").result == crypto.SKIPPED
    assert res.verdict == "INCOMPLETE"


def test_pipelock_schema_rejects_bad_algorithm() -> None:
    """A Pipelock receipt carrying an unsupported algorithm token FAILs the structure axis."""
    doc = _pipelock("pipelock-ev2-01-proxy-decision")
    forged = json.loads(json.dumps(doc))
    forged["signature"]["algorithm"] = "ecdsa-p256"
    res = verify(forged, ADAPTERS, key_provider=_PIPELOCK_KEY_PROVIDER)
    assert res.axis("structure").result == crypto.FAIL
    assert res.verdict == "FAIL"


def test_pipelock_chain_genesis_sentinel_accepted() -> None:
    """Both 'genesis' and 'sha256:0' chain_prev_hash values are accepted as genesis."""
    doc = _pipelock("pipelock-ev2-01-proxy-decision")
    ad = PipelockEvidenceAdapter()
    # The fixture uses sha256:0.
    assert ad.chain_step(doc).is_genesis is True
    # genesis string is also a valid sentinel.
    doc2 = json.loads(json.dumps(doc))
    doc2["chain_prev_hash"] = "genesis"
    assert ad.chain_step(doc2).is_genesis is True
    # A real chain link (sha256:<hex>) is not genesis.
    doc3 = json.loads(json.dumps(doc))
    doc3["chain_prev_hash"] = "sha256:" + "a" * 64
    assert ad.chain_step(doc3).is_genesis is False
