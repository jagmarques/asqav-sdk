"""Export-only VC shim tests - data round-trip + the honesty guard."""
from __future__ import annotations

import base64
import json
from pathlib import Path

from asqav.verifier.oracle import ADAPTERS, verify
from asqav.verifier.oracle.adapters.agentreceipts import AgentReceiptsAdapter
from asqav.verifier.vc_export import (
    ASQAV_PROOF_TYPE,
    STANDARD_VC_PROOF_TYPES,
    VC_CREDENTIAL_TYPE,
    to_vc_envelope,
)

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"


def _load(vec: str, name: str = "receipt.json") -> dict:
    return json.loads((_CORPUS / vec / name).read_text())


# --- (a) data round-trip ---


    # An Asqav-native receipt maps agent/action/policy + issuer faithfully.
def test_native_receipt_maps_key_fields_into_credential_subject() -> None:
    receipt = _load("asqav-01-genesis-permit")
    vc = to_vc_envelope(receipt)

    assert vc["type"] == VC_CREDENTIAL_TYPE
    assert vc["@context"][0] == "https://www.w3.org/ns/credentials/v2"
    assert vc["issuer"]["id"] == receipt["payload"]["issuer_id"]
    sub = vc["credentialSubject"]
    assert sub["agent"]["id"] == receipt["payload"]["agent_id"]
    assert sub["action"]["type"] == receipt["payload"]["type"]
    assert sub["action"]["reference"] == receipt["payload"]["action_ref"]
    assert sub["action"]["toolName"] == receipt["payload"]["tool_name"]
    assert sub["policy"]["decision"] == receipt["payload"]["decision"]
    assert vc["validFrom"] == receipt["payload"]["issued_at"]


    # A W3C-VC AgentReceipt maps action/principal/chain + issuer into subject.
def test_agentreceipts_receipt_maps_subject_and_issuer() -> None:
    receipt = _load("agentreceipts-01-didkey-genesis")
    vc = to_vc_envelope(receipt)

    assert vc["issuer"]["id"] == receipt["issuer"]["id"]
    sub = vc["credentialSubject"]
    assert sub["action"] == receipt["credentialSubject"]["action"]
    assert sub["principal"] == receipt["credentialSubject"]["principal"]
    assert sub["outcome"] == receipt["credentialSubject"]["outcome"]
    assert sub["chain"] == receipt["credentialSubject"]["chain"]


    # The export never mutates the source receipt.
def test_export_is_pure_does_not_mutate_input() -> None:
    receipt = _load("asqav-01-genesis-permit")
    before = json.dumps(receipt, sort_keys=True)
    to_vc_envelope(receipt)
    assert json.dumps(receipt, sort_keys=True) == before


    # A flat hash-mode /sign receipt also maps (agent/org/action/policy).
def test_hash_mode_receipt_exports() -> None:
    receipt = _load("asqav-05-hash-mode-prod")
    vc = to_vc_envelope(receipt)
    sub = vc["credentialSubject"]
    assert sub["agent"]["id"] == receipt["agent_id"]
    assert sub["organization"]["id"] == receipt["org_id"]
    assert sub["action"]["contentHash"] == receipt["hash"]
    assert vc["proof"]["asqavAlgorithm"] == "ML-DSA-65"


# --- (c) signature preserved verbatim ---


    # The original Asqav signature bytes appear unchanged in the proof.
def test_native_signature_preserved_verbatim() -> None:
    receipt = _load("asqav-01-genesis-permit")
    vc = to_vc_envelope(receipt)
    assert vc["proof"]["signatureValue"] == receipt["signature"]["sig"]
    assert vc["proof"]["verificationKeyId"] == receipt["signature"]["kid"]
    assert vc["proof"]["asqavAlgorithm"] == receipt["signature"]["alg"]


def test_hash_mode_signature_preserved_verbatim() -> None:
    receipt = _load("asqav-05-hash-mode-prod")
    vc = to_vc_envelope(receipt)
    assert vc["proof"]["signatureValue"] == receipt["signature_b64"]


def test_agentreceipts_signature_preserved_verbatim() -> None:
    receipt = _load("agentreceipts-01-didkey-genesis")
    vc = to_vc_envelope(receipt)
    assert vc["proof"]["signatureValue"] == receipt["proof"]["proofValue"]


    # The proof's signingInput pointer is the exact bytes the oracle re-derives.
def test_signing_input_pointer_matches_oracle_bytes() -> None:
    receipt = _load("asqav-01-genesis-permit")
    vc = to_vc_envelope(receipt)
    pointer = base64.b64decode(vc["proof"]["signingInputBase64"])
    # The asqav-native adapter signs the payload canonical bytes directly.
    from asqav.verifier.oracle.adapters.asqav_native import AsqavNativeAdapter

    assert pointer == AsqavNativeAdapter().signing_input(receipt)


# --- (b) THE HONESTY GUARD ---


    # proof.type is the Asqav non-standard type, NOT any registered VC suite.
def test_proof_type_is_asqav_native_never_a_standard_vc_suite() -> None:
    vecs = ("asqav-01-genesis-permit", "agentreceipts-01-didkey-genesis", "asqav-05-hash-mode-prod")
    for vec in vecs:
        vc = to_vc_envelope(_load(vec))
        assert vc["proof"]["type"] == ASQAV_PROOF_TYPE
        assert vc["proof"]["type"] not in STANDARD_VC_PROOF_TYPES
        assert vc["proofIsAsqavNative"] is True
        # The credential type must not borrow the standard-suite AgentReceipt.
        assert "AgentReceipt" not in vc["type"]


def test_exported_envelope_does_not_false_pass_the_standard_vc_verifier() -> None:
    """Feeding the export to the oracle does NOT yield a standard-VC PASS."""
    for vec in ("asqav-01-genesis-permit", "agentreceipts-01-didkey-genesis"):
        vc = to_vc_envelope(_load(vec))
        assert AgentReceiptsAdapter().detect(vc) is False
        res = verify(vc, ADAPTERS)
        assert res.verdict != "PASS"
        assert res.fmt != "agentreceipts"


def test_relabelling_export_as_standard_suite_still_cannot_verify() -> None:
    """Swapping in the standard token still cannot verify the bytes as a VC."""
    receipt = _load("agentreceipts-01-didkey-genesis")
    vc = to_vc_envelope(receipt)
    forged = json.loads(json.dumps(vc))
    forged["type"] = ["VerifiableCredential", "AgentReceipt"]
    forged["version"] = "0.1.0"
    forged["id"] = receipt["id"]
    forged["credentialSubject"] = {"chain": {"previous_receipt_hash": None}}
    forged["proof"] = {
        "type": "Ed25519Signature2020",
        "created": "2026-05-04T12:00:00Z",
        "verificationMethod": receipt["proof"]["verificationMethod"],
        "proofPurpose": "assertionMethod",
        "proofValue": vc["proof"]["signatureValue"],
    }
    res = verify(forged, ADAPTERS)
    assert res.verdict != "PASS"


    # A third-party ACTA receipt maps payload.issued_at into the VC validFrom.
def test_upstream_acta_receipt_carries_validfrom_from_issued_at() -> None:
    receipt = _load("acta-up-01-a2a-trusted-attestation")
    vc = to_vc_envelope(receipt)
    assert vc["validFrom"] == receipt["payload"]["issued_at"]
    assert vc["issuanceDate"] == vc["validFrom"]
