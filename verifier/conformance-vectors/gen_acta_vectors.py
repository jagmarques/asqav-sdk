"""Generate the ACTA conformance vectors for the oracle corpus.

Deterministic: a fixed Ed25519 test seed so the vectors are reproducible and
reviewable. Produces genesis (verified), chain-link (verified), tamper-sig
(unverified), a commitment-mode receipt (unverified under the baseline
verifier - the honest outcome for an unsupported optional mode), and the two
chain-form vectors for ACTA -03 §6.7: a prefixed link that verifies and a
prefixed link whose hex is wrong. Signs exactly what the ACTA adapter
verifies: Ed25519 over jcs(payload), hex-encoded.

Owns the numbered acta-0N entries of manifest.json and rewrites them in place
at their existing position; the acta-up-* entries (minted by
gen_acta_upstream_vectors.py) and every other family pass through untouched.

Run from the repo root: python verifier/conformance-vectors/gen_acta_vectors.py
Re-freeze the corpus lock afterwards: python verifier/freeze_corpus_lock.py
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

_HERE = Path(__file__).resolve().parent
_SEED = bytes.fromhex("00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff")
_KID = "sb:issuer:acta-oracle-vec-key"

_sk = Ed25519PrivateKey.from_private_bytes(_SEED)
_pk_raw = _sk.public_key().public_bytes_raw()

#: The directories this generator owns, in manifest order. Everything else in
#: manifest.json passes through untouched.
_OWNED = [
    "acta-01-genesis",
    "acta-02-chain-link",
    "acta-03-tamper-sig",
    "acta-05-commitment-mode-unsupported",
    "acta-06-chain-link-03-prefixed",
    "acta-07-chain-link-03-wrong-digest",
]


def _jcs(obj: object) -> bytes:
    """Canonical JSON bytes: sorted keys, tight separators, UTF-8."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _keyset() -> dict:
    return {"keys": [{"kty": "OKP", "crv": "Ed25519", "kid": _KID, "x": _b64url(_pk_raw)}]}


def _sign(payload: dict) -> dict:
    sig = _sk.sign(_jcs(payload))
    return {"payload": payload, "signature": {"alg": "EdDSA", "kid": _KID, "sig": sig.hex()}}


def _genesis_payload() -> dict:
    return {
        "type": "protectmcp:decision",
        "issued_at": "2026-05-04T12:00:00+00:00",
        "issuer_id": _KID,
        "agent_id": "agt_acta_001",
        "action_ref": "sha256:" + "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "decision": "allow",
    }


def _chain_hash(receipt: dict) -> str:
    """The link preimage: SHA-256 hex of the full signed predecessor."""
    return hashlib.sha256(_jcs(receipt)).hexdigest()


def _expected(outcome: str, reason_code: str, notes: str, failure_class: str | None = None) -> dict:
    entry = {"format": "acta", "outcome": outcome, "reason_code": reason_code, "notes": notes}
    if failure_class is not None:
        entry["failure_class"] = failure_class
    return entry


def _write(name: str, files: dict) -> None:
    d = _HERE / name
    d.mkdir(exist_ok=True)
    for fname, obj in files.items():
        (d / fname).write_text(json.dumps(obj, indent=2) + "\n")
    print(f"wrote {name}")


def _manifest_entry(dir_name: str, outcome: str, reason_code: str, notes: str,
                    failure_class: str | None = None) -> dict:
    entry = {"dir": dir_name, "format": "acta", "outcome": outcome}
    if failure_class is not None:
        entry["failure_class"] = failure_class
    entry["reason_code"] = reason_code
    entry["notes"] = notes
    return entry


def main() -> None:
    keyset = _keyset()
    genesis = _sign(_genesis_payload())

    _write(
        "acta-01-genesis",
        {
            "receipt.json": genesis,
            "acta-keys.json": keyset,
            "expected.json": _expected(
                "verified", "", "Valid ACTA genesis; Ed25519 over JCS(payload) verifies."
            ),
        },
    )

    succ_payload = _genesis_payload()
    succ_payload["agent_id"] = "agt_acta_002"
    succ_payload["previousReceiptHash"] = _chain_hash(genesis)
    successor = _sign(succ_payload)
    _write(
        "acta-02-chain-link",
        {
            "receipt.json": successor,
            "predecessor.json": genesis,
            "acta-keys.json": keyset,
            "expected.json": _expected(
                "verified",
                "",
                "ACTA successor links via SHA-256 of the JCS of the full predecessor receipt.",
            ),
        },
    )

    tampered = _sign(_genesis_payload())
    bad = bytearray(bytes.fromhex(tampered["signature"]["sig"]))
    bad[0] ^= 0x01
    tampered["signature"]["sig"] = bad.hex()
    _write(
        "acta-03-tamper-sig",
        {
            "receipt.json": tampered,
            "acta-keys.json": keyset,
            "expected.json": _expected(
                "unverified", "sig_mismatch", "Flipped signature byte; Ed25519 verify fails.",
                "invalid",
            ),
        },
    )

    # Optional commitment mode signs SHA-256(JCS); the baseline verifier MUST fail it, never pass.
    commit_payload = _genesis_payload()
    commit_payload["agent_id"] = "agt_acta_commit"
    digest = hashlib.sha256(_jcs(commit_payload)).digest()
    commit = {
        "payload": commit_payload,
        "signature": {"alg": "EdDSA", "kid": _KID, "sig": _sk.sign(digest).hex()},
    }
    _write(
        "acta-05-commitment-mode-unsupported",
        {
            "receipt.json": commit,
            "acta-keys.json": keyset,
            "expected.json": _expected(
                "unverified",
                "sig_mismatch",
                "Commitment-mode receipt (signs SHA-256(JCS)) fails the baseline JCS verifier - honest, never a false pass.",
                "invalid",
            ),
        },
    )

    # ACTA -03 §6.7: the chain link carries "sha256:" + the hex of the full signed
    # predecessor. The preimage is unchanged from -02; only the form differs.
    succ06_payload = _genesis_payload()
    succ06_payload["agent_id"] = "agt_acta_006"
    succ06_payload["previousReceiptHash"] = "sha256:" + _chain_hash(genesis)
    _write(
        "acta-06-chain-link-03-prefixed",
        {
            "receipt.json": _sign(succ06_payload),
            "predecessor.json": genesis,
            "acta-keys.json": keyset,
            "expected.json": _expected(
                "verified",
                "",
                "ACTA -03 §6.7 chain form: previousReceiptHash carries the 'sha256:' "
                "prefix over SHA-256 of the JCS of the full signed predecessor. Both "
                "receipts are Ed25519-signed by the corpus seed key; the adapter reads "
                "the form from the carried value and verifies.",
            ),
        },
    )

    wrong = list("sha256:" + _chain_hash(genesis))
    wrong[7] = "0" if wrong[7] != "0" else "1"
    succ07_payload = _genesis_payload()
    succ07_payload["agent_id"] = "agt_acta_006"
    succ07_payload["previousReceiptHash"] = "".join(wrong)
    _write(
        "acta-07-chain-link-03-wrong-digest",
        {
            "receipt.json": _sign(succ07_payload),
            "predecessor.json": genesis,
            "acta-keys.json": keyset,
            "expected.json": _expected(
                "unverified",
                "chain",
                "The -03 prefixed form with one wrong hex nibble, re-signed so the "
                "signature PASSes and the chain axis is what FAILs. A proven break, so "
                "the class is invalid; the reason code is the corpus's 'chain' token "
                "(aerf-04's), the acta family having no older broken-link vector.",
                "invalid",
            ),
        },
    )

    manifest_path = _HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    owned = [
        _manifest_entry(
            "acta-01-genesis", "verified", "",
            "Valid ACTA genesis; Ed25519 over JCS(payload) verifies.",
        ),
        _manifest_entry(
            "acta-02-chain-link", "verified", "",
            "ACTA successor links via SHA-256 of the JCS of the full predecessor receipt.",
        ),
        _manifest_entry(
            "acta-03-tamper-sig", "unverified", "sig_mismatch",
            "Flipped signature byte; Ed25519 verify fails.", "invalid",
        ),
        _manifest_entry(
            "acta-05-commitment-mode-unsupported", "unverified", "sig_mismatch",
            "Commitment-mode receipt (signs SHA-256(JCS)) fails the baseline JCS verifier - honest, never a false pass.",
            "invalid",
        ),
        _manifest_entry(
            "acta-06-chain-link-03-prefixed", "verified", "",
            "ACTA -03 §6.7 chain form: previousReceiptHash carries the 'sha256:' prefix "
            "over SHA-256 of the JCS of the full signed predecessor; verifies in both "
            "languages.",
        ),
        _manifest_entry(
            "acta-07-chain-link-03-wrong-digest", "unverified", "chain",
            "The -03 prefixed form with one wrong hex nibble, re-signed so the signature "
            "PASSes and the chain axis FAILs. Proven break: invalid.",
            "invalid",
        ),
    ]
    # Replace the owned entries in place, keeping every other family's position.
    first = min(i for i, m in enumerate(manifest) if m["dir"] in _OWNED)
    manifest = [m for m in manifest if m["dir"] not in _OWNED]
    manifest[first:first] = owned
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote ACTA vectors; manifest now {len(manifest)} entries")


if __name__ == "__main__":
    main()
