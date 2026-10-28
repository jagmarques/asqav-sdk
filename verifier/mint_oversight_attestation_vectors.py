"""Mint the oversight-attestation mirrors as the next unissued asqav-N ids."""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python" / "src"))

ASQAV_JCS = importlib.import_module("asqav.verifier.oracle.canonical").asqav_jcs
STATE = importlib.import_module("asqav.verifier.oversight_attestation")

CORPUS = HERE / "conformance-vectors"
MANIFEST = CORPUS / "manifest.json"
SEED_PHRASE = b"asqav conformance corpus v1 oversight-attestation signing seed"
ISSUER = "00000000000000000098"
KEY_ID = "asqav-oversight-attestation-vector-key"
TYPE = "protectmcp:lifecycle:oversight_ruling"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
POLICY_SHA256 = (
    "9b71d224bd62f3785d96d46ad4a5eae02a2dff72519673ca7a7bfbc2890caada"
    "e02a2dff72519673ca"
)

#: Shared signed_envelope body (platform vectors 048/049/050 differ only in
#: reviewer.attestation); every value is copied verbatim from the platform.
PAYLOAD = {
    "action_ref": f"sha256:{EMPTY_SHA256}",
    "action_type": "review:conduct",
    "agreement_version": "art26-assignment-v3",
    "decision": "observation",
    "evidence_manifest": [f"sha256:{POLICY_SHA256}"],
    "issued_at": "2026-06-10T09:00:00+00:00",
    "issuer_id": ISSUER,
    "judged_action_refs": [f"sha256:{EMPTY_SHA256}"],
    "payload_digest": {"hash": EMPTY_SHA256, "size": 0},
    "policy_digest": f"sha256:{POLICY_SHA256}",
    "previousReceiptHash": "0" * 64,
    "ruling": "confirm",
    "tool_name": "review:conduct",
    "type": TYPE,
}
REVIEWER = {
    "authority": "oversight-charter-2026-s4",
    "principal": "reviewer_opaque_001",
    "role": "oversight-judge",
}

#: (slug, attestation, claim, platform vector, outcome note)
CASES = (
    (
        "oversight-attestation-missing",
        None,
        STATE.MISSING,
        "vector-048",
        (
            "the reviewer carries no attestation, so the review claim is "
            "missing/unverified."
        ),
    ),
    (
        "oversight-attestation-asserted",
        {"method": "manual_assertion"},
        STATE.ASSERTED,
        "vector-049",
        (
            "manual_assertion names no signature backing, so the review claim is "
            "asserted/unverified."
        ),
    ),
    (
        "oversight-attestation-signature-verified",
        {"method": "sso", "evidence": f"sha256:{POLICY_SHA256}"},
        STATE.SIGNATURE_VERIFIED,
        "vector-050",
        (
            "sso names signature backing, so the review claim is "
            "signature-verified once the receipt signature verifies."
        ),
    ),
)
SLUGS = tuple(case[0] for case in CASES)

#: The shipped verifier reports the registered type as unimplemented.
OUTCOME = "unverified"
FAILURE_CLASS = "unverifiable"
REASON = "registered_but_unimplemented"


def vector_dirs() -> list[str]:
    """The three corpus directories, allocated once and re-read thereafter."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    issued = [
        int(match.group(1))
        for entry in manifest
        for match in [re.match(r"asqav-(\d+)-", entry["dir"])]
        if match
    ]
    allocated = {
        match.group(2): int(match.group(1))
        for path in CORPUS.iterdir()
        if path.is_dir()
        for match in [re.fullmatch(r"asqav-(\d+)-(.+)", path.name)]
        if match and match.group(2) in SLUGS
    }
    if allocated:
        if set(allocated) != set(SLUGS):
            raise SystemExit(f"partial oversight-attestation allocation: {allocated}")
        return [f"asqav-{allocated[slug]}-{slug}" for slug in SLUGS]
    start = max(issued, default=0) + 1
    return [f"asqav-{start + index}-{slug}" for index, slug in enumerate(SLUGS)]


def _notes(platform: str, detail: str) -> str:
    return (
        f"Mirrors platform conformance/v3-ietf/{platform} with its signed_envelope "
        f"copied verbatim as the payload and a fresh deterministic test-key "
        f"signature: {detail} The SDK verifier does not implement the "
        f"oversight_ruling type, so structure SKIPs and the expected verdict is "
        f"unverified/unverifiable."
    )


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _payload(attestation: dict | None) -> dict:
    reviewer = ({**REVIEWER} if attestation is None
                else {"attestation": attestation, **REVIEWER})
    return {**PAYLOAD, "reviewer": reviewer}


def _key_material():
    from dilithium_py.ml_dsa import ML_DSA_65

    return ML_DSA_65, ML_DSA_65.key_derive(hashlib.sha256(SEED_PHRASE).digest())


def _jwks(public_key: bytes) -> dict:
    return {
        "keys": [
            {
                "kid": KEY_ID,
                "issuer_id": ISSUER,
                "alg": "ML-DSA-65",
                "kty": "AKP",
                "pub": _b64u(public_key),
                "public_key": base64.b64encode(public_key).decode("ascii"),
                "status": "active",
            }
        ]
    }


def _manifest_entry(dirname: str, notes: str) -> dict:
    return {
        "dir": dirname,
        "format": "asqav-native",
        "outcome": OUTCOME,
        "failure_class": FAILURE_CLASS,
        "reason_code": REASON,
        "notes": notes,
    }


def _append_ids(entries: dict[str, dict]) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_dir = {entry["dir"]: entry for entry in manifest}
    for dirname, entry in entries.items():
        by_dir[dirname] = entry
    corpus_order = [entry["dir"] for entry in manifest]
    tail = [dirname for dirname in entries if dirname not in corpus_order]
    rebuilt = [by_dir[dirname] for dirname in corpus_order] + [
        by_dir[dirname] for dirname in tail
    ]
    MANIFEST.write_text(json.dumps(rebuilt, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ML_DSA_65, (public_key, secret_key) = _key_material()
    jwks = _jwks(public_key)
    dirs = vector_dirs()
    entries = {}
    for dirname, (slug, attestation, claim, platform, detail) in zip(dirs, CASES):
        assert dirname.endswith(slug), dirname
        payload = _payload(attestation)
        signature = ML_DSA_65.sign(secret_key, ASQAV_JCS(payload), deterministic=True)
        receipt = {
            "payload": payload,
            "signature": {
                "alg": "ML-DSA-65",
                "kid": ISSUER,
                "sig": _b64u(signature),
            },
            "anchors": [],
        }
        notes = _notes(platform, detail)
        directory = CORPUS / dirname
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        (directory / "jwks.json").write_text(
            json.dumps(jwks, indent=2) + "\n", encoding="utf-8"
        )
        expected_path = directory / "expected.json"
        expected = {
            "format": "asqav-native",
            "outcome": OUTCOME,
            "failure_class": FAILURE_CLASS,
            "reason_code": REASON,
            "notes": notes,
            "oversight_attestation": claim,
        }
        if expected_path.exists():
            previous = json.loads(expected_path.read_text(encoding="utf-8"))
            if "axis_scope" in previous:
                expected["axis_scope"] = previous["axis_scope"]
        expected_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        entries[dirname] = _manifest_entry(dirname, notes)
        print(f"wrote {dirname}")
    _append_ids(entries)


if __name__ == "__main__":
    main()
