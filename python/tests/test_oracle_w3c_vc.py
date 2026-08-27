"""W3C VC 2.0 eddsa-jcs-2022 adapter tests.

Covers the DataIntegrityProof path: did:web resolution from an injected DID
document (offline, fail-closed), did:key self-resolution, the hashData signing
construction, tamper rejection, the @context prefix transform, and the validity
window. Corpus vectors w3c-vc-01..08 run through the shared runner; the tests
here pin the axis-level behaviour behind them.
"""
from __future__ import annotations

import json
from base64 import urlsafe_b64encode
from pathlib import Path

import pytest

from asqav.verifier.oracle import ADAPTERS, crypto, verify
from asqav.verifier.oracle.adapters.agentreceipts import AgentReceiptsAdapter
from asqav.verifier.oracle.adapters.w3c_vc import W3cVcAdapter
from asqav.verifier.oracle.did import resolve_ed25519_key

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"

_ED25519_AVAILABLE = crypto.verify_ed25519(b"\x00" * 32, b"m", b"\x00" * 64)[0] != crypto.SKIPPED

requires_ed25519 = pytest.mark.skipif(
    not _ED25519_AVAILABLE, reason="cryptography not installed; Ed25519 verify SKIPs"
)


def _load(vec: str, name: str) -> dict:
    return json.loads((_CORPUS / vec / name).read_text())


def _provider(vec: str):
    path = _CORPUS / vec / "did_map.json"
    return json.loads(path.read_text()) if path.exists() else None


    # A did:web credential verifies against the injected DID document, offline.
@requires_ed25519
def test_w3c_vc_didweb_happy_path_verifies() -> None:
    doc = _load("w3c-vc-01-didweb-happy-path", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("w3c-vc-01-didweb-happy-path"))
    assert res.fmt == "w3c-vc"
    assert res.verdict == "verified"
    assert res.failure_class is None
    assert res.axis("signature").result == crypto.PASS
    assert res.axis("expiry").result == crypto.PASS


    # A did:key credential self-resolves; no injected map is consulted.
@requires_ed25519
def test_w3c_vc_didkey_self_resolves() -> None:
    doc = _load("w3c-vc-08-didkey-happy-path", "receipt.json")
    res = verify(doc, ADAPTERS)
    assert res.verdict == "verified"
    assert res.axis("signature").result == crypto.PASS


@requires_ed25519
def test_w3c_vc_tampered_subject_fails_signature() -> None:
    doc = _load("w3c-vc-02-tamper-subject", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("w3c-vc-02-tamper-subject"))
    assert res.verdict == "unverified"
    assert res.failure_class == "invalid"
    assert res.axis("signature").result == crypto.FAIL


@requires_ed25519
def test_w3c_vc_tampered_proofvalue_fails_signature() -> None:
    doc = _load("w3c-vc-03-tamper-proofvalue", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("w3c-vc-03-tamper-proofvalue"))
    assert res.verdict == "unverified"
    assert res.failure_class == "invalid"
    assert res.axis("signature").result == crypto.FAIL


    # A DID document publishing the wrong key fails the signature, never a false pass.
@requires_ed25519
def test_w3c_vc_wrong_published_key_fails() -> None:
    doc = _load("w3c-vc-04-wrong-key-injected", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("w3c-vc-04-wrong-key-injected"))
    assert res.verdict == "unverified"
    assert res.failure_class == "invalid"
    assert res.axis("signature").result == crypto.FAIL


    # No injected DID document -> the oracle never fetches; fail closed, not verified.
@requires_ed25519
def test_w3c_vc_no_did_document_fails_closed() -> None:
    doc = _load("w3c-vc-05-no-did-document", "receipt.json")
    res = verify(doc, ADAPTERS)
    assert res.axis("signature").result == crypto.SKIPPED
    assert res.verdict == "unverified"
    assert res.failure_class == "unverifiable"


    # A lapsed validUntil FAILs the expiry axis alone; the verdict stays verified (426).
@requires_ed25519
def test_w3c_vc_expired_keeps_verified_verdict() -> None:
    doc = _load("w3c-vc-06-expired", "receipt.json")
    res = verify(doc, ADAPTERS, key_provider=_provider("w3c-vc-06-expired"))
    assert res.verdict == "verified"
    expiry = res.axis("expiry")
    assert expiry.result == crypto.FAIL
    assert expiry.note.startswith("expired at ")


    # Proof sets are not supported: a list of proofs FAILs the structure axis.
def test_w3c_vc_proof_set_is_rejected() -> None:
    doc = _load("w3c-vc-01-didweb-happy-path", "receipt.json")
    doc["proof"] = [doc["proof"]]
    ad = W3cVcAdapter()
    assert ad.detect(doc) is True
    result, note = ad.schema(doc)
    assert result == "FAIL" and "proof sets are not supported" in note
    res = verify(doc, ADAPTERS, key_provider=_provider("w3c-vc-01-didweb-happy-path"))
    assert res.verdict == "unverified"


    # A sibling cryptosuite routes to the adapter and FAILs as an algorithm mismatch.
def test_w3c_vc_other_cryptosuite_is_algorithm_mismatch() -> None:
    doc = _load("w3c-vc-01-didweb-happy-path", "receipt.json")
    doc["proof"]["cryptosuite"] = "eddsa-rdfc-2022"
    ad = W3cVcAdapter()
    assert ad.detect(doc) is True
    result, note = ad.schema(doc)
    assert result == "FAIL"
    assert note.startswith("unsupported signature algorithm")
    res = verify(doc, ADAPTERS, key_provider=_provider("w3c-vc-01-didweb-happy-path"))
    assert res.verdict == "unverified"
    assert res.failure_class == "invalid"


    # proofPurpose must be assertionMethod for a credential assertion (fail closed).
def test_w3c_vc_proofpurpose_must_be_assertion_method() -> None:
    doc = _load("w3c-vc-01-didweb-happy-path", "receipt.json")
    doc["proof"]["proofPurpose"] = "authentication"
    result, note = W3cVcAdapter().schema(doc)
    assert result == "FAIL" and "proofPurpose" in note


    # An attacker's valid signature under a key the issuer does not control must not verify.
@requires_ed25519
def test_w3c_vc_issuer_must_control_signing_key_no_impersonation() -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    attacker_did = "did:key:z" + _b58_encode(b"\xed\x01" + pk)
    doc = _load("w3c-vc-01-didweb-happy-path", "receipt.json")
    doc["issuer"] = "did:web:victim.example"
    doc["proof"]["verificationMethod"] = attacker_did + "#key-1"
    doc["proof"].pop("proofValue", None)
    ad = W3cVcAdapter()
    doc["proof"]["proofValue"] = "z" + _b58_encode(sk.sign(ad.signing_input(doc)))
    res = verify(doc, ADAPTERS)
    assert res.axis("signature").result == crypto.PASS  # attacker's sig is cryptographically valid
    assert res.axis("structure").result == crypto.FAIL  # but signing-key DID != issuer
    assert res.verdict == "unverified"
    assert res.failure_class == "invalid"


_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


    # base58btc encoder for test fixtures (mirrors the decoder pinned in test_oracle).
def _b58_encode(data: bytes) -> str:
    num = int.from_bytes(data, "big")
    out = []
    while num:
        num, rem = divmod(num, 58)
        out.append(_B58_ALPHABET[rem])
    pad = len(data) - len(data.lstrip(b"\x00"))
    return "1" * pad + "".join(reversed(out))


    # A proof @context that prefixes the document @context is signed and verifies.
@requires_ed25519
def test_w3c_vc_proof_context_prefix_transform() -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    # Re-issue under a fresh did:key issuer (the corpus private key is not recoverable).
    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    issuer = "did:key:z" + _b58_encode(b"\xed\x01" + pk)
    doc = _load("w3c-vc-01-didweb-happy-path", "receipt.json")
    doc["issuer"] = issuer
    doc["proof"]["verificationMethod"] = issuer + "#key-1"
    doc["proof"]["@context"] = ["https://www.w3.org/ns/credentials/v2"]
    doc["proof"].pop("proofValue", None)
    ad = W3cVcAdapter()
    doc["proof"]["proofValue"] = "z" + _b58_encode(sk.sign(ad.signing_input(doc)))
    res = verify(doc, ADAPTERS)
    assert res.axis("structure").result == crypto.PASS
    assert res.verdict == "verified"

    # A proof @context the document does not start with is a proven mismatch: invalid.
    bad = json.loads(json.dumps(doc))
    bad["proof"]["@context"] = ["https://example.org/other-context"]
    result, note = ad.schema(bad)
    assert result == "FAIL"
    assert note.startswith("proof @context is not a prefix")
    res = verify(bad, ADAPTERS)
    assert res.verdict == "unverified"
    assert res.failure_class == "invalid"


    # A malformed proofValue fails closed through verify() and never crashes.
def test_w3c_vc_malformed_proofvalue_fails_closed() -> None:
    doc = _load("w3c-vc-01-didweb-happy-path", "receipt.json")
    for bad in ("u-not-base58btc", "", {"k": "v"}, None):
        forged = json.loads(json.dumps(doc))
        forged["proof"]["proofValue"] = bad
        res = verify(forged, ADAPTERS, key_provider=_provider("w3c-vc-01-didweb-happy-path"))
        assert res.verdict == "unverified"
        assert W3cVcAdapter().extract_signature(forged).sig == b""


    # An unreadable validFrom FAILs the expiry axis but never folds the verdict (426).
def test_w3c_vc_unreadable_valid_from_reports_on_expiry_axis() -> None:
    doc = _load("w3c-vc-01-didweb-happy-path", "receipt.json")
    doc["validFrom"] = "not-a-date"
    axes = W3cVcAdapter().extra_axes(doc, None)
    assert axes == [("expiry", "FAIL", "unreadable expires_at (validFrom 'not-a-date')")]


# --- DID-document resolution (the offline did:web path) ---


def _did_doc_with(*methods: dict, assertion_refs: list | None = None) -> dict:
    return {
        "id": "did:web:example.com",
        "verificationMethod": list(methods),
        "assertionMethod": assertion_refs or [],
    }


def test_did_document_fragment_match_resolves() -> None:
    raw = bytes(range(32))
    doc = _did_doc_with(
        {"id": "did:web:example.com#key-1", "publicKeyMultibase": "z" + _b58_encode(b"\xed\x01" + raw)}
    )
    key, note = resolve_ed25519_key("did:web:example.com#key-1", {"did:web:example.com": doc})
    assert key == raw and "injected DID document" in note


def test_did_document_missing_fragment_fails_closed() -> None:
    doc = _did_doc_with({"id": "did:web:example.com#other", "publicKeyBase58": _b58_encode(bytes(range(32)))})
    key, note = resolve_ed25519_key("did:web:example.com#key-1", {"did:web:example.com": doc})
    assert key is None and "no verificationMethod" in note


def test_did_document_prefers_assertion_method_reference() -> None:
    key_a = bytes([1] * 32)
    key_b = bytes([2] * 32)
    doc = _did_doc_with(
        {"id": "did:web:example.com#key-a", "publicKeyBase58": _b58_encode(key_a)},
        {"id": "did:web:example.com#key-b", "publicKeyBase58": _b58_encode(key_b)},
        assertion_refs=["did:web:example.com#key-b"],
    )
    key, _note = resolve_ed25519_key("did:web:example.com", {"did:web:example.com": doc})
    assert key == key_b


def test_did_document_public_key_jwk_resolves() -> None:
    raw = bytes(range(32, 64))
    x = urlsafe_b64encode(raw).decode().rstrip("=")
    doc = _did_doc_with({"id": "did:web:example.com#jwk", "publicKeyJwk": {"kty": "OKP", "crv": "Ed25519", "x": x}})
    key, _note = resolve_ed25519_key("did:web:example.com#jwk", {"did:web:example.com": doc})
    assert key == raw


def test_did_document_non_ed25519_multikey_fails_closed() -> None:
    # 0xe7 0x01 is the secp256k1 multicodec prefix; not an Ed25519 key
    doc = _did_doc_with({"id": "did:web:example.com#k", "publicKeyMultibase": "z" + _b58_encode(b"\xe7\x01" + bytes(32))})
    key, note = resolve_ed25519_key("did:web:example.com#k", {"did:web:example.com": doc})
    assert key is None and "no Ed25519 verificationMethod" in note


    # The raw-key injection shape stays fully backwards compatible.
def test_did_map_raw_hex_still_resolves() -> None:
    raw = bytes(range(32))
    key, note = resolve_ed25519_key("did:agent:x#k1", {"did:agent:x": raw.hex()})
    assert key == raw and "injected map" in note


# --- detection exclusion against the sibling VC format ---


def test_w3c_vc_and_agentreceipts_are_mutually_exclusive() -> None:
    vc = _load("w3c-vc-01-didweb-happy-path", "receipt.json")
    ar = _load("agentreceipts-01-didkey-genesis", "receipt.json")
    w, g = W3cVcAdapter(), AgentReceiptsAdapter()
    assert (w.detect(vc), g.detect(vc)) == (True, False)
    assert (w.detect(ar), g.detect(ar)) == (False, True)
