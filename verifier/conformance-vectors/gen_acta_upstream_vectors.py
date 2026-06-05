"""Regenerate the upstream-derived ACTA vectors from the ScopeBlind corpus.

Source: github.com/ScopeBlind/agent-governance-testvectors (Apache-2.0), the
shared conformance corpus for draft-farley-acta-signed-receipts. The receipts
ingested here are REAL third-party artifacts: their Ed25519 signatures are
produced by the ScopeBlind/APS reference issuer over the JCS of the payload, the
same primitive our ACTA adapter checks. We verify a producer we did not write.

The ScopeBlind passport receipts carry the signer key inline as ``signature.pubkey``
(hex). The upstream signing key is the deterministic conformance key whose 32-byte
public half is ``4cb5abf6ad79fbf5abbccafcc269d85cd2651ed4b885b5869f241aedf0a5ba29``
(the ``aps_issuer`` in a2a-trust-header/_keys.json), so our ``acta-keys.json`` JWK
Set is that public half re-expressed as a base64url OKP/Ed25519 JWK keyed by the
receipt's ``kid``.

Set ``ACTA_UPSTREAM_SRC`` to a checkout of that repo, then run from verifier/:

    ACTA_UPSTREAM_SRC=/path/to/agent-governance-testvectors \
        python conformance-vectors/gen_acta_upstream_vectors.py

The pinned commit is recorded in conformance-vectors/UPSTREAM.md.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_SRC = "/tmp/agtv"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _keyset(att: dict) -> dict:
    sig = att["signature"]
    raw = bytes.fromhex(sig["pubkey"])
    return {"keys": [{"kty": "OKP", "crv": "Ed25519", "kid": sig["kid"], "x": _b64url(raw)}]}


def _passport(att: dict) -> dict:
    """Lift a ScopeBlind attestation to the passport envelope our adapter detects.

    The upstream ``signature`` object already carries ``alg/kid/sig``; the extra
    ``pubkey``/``canonicalization`` members are kept verbatim so the vector stays
    byte-faithful to the upstream artifact.
    """
    s = att["signature"]
    sig = {"alg": s["alg"], "kid": s["kid"], "sig": s["sig"], "pubkey": s["pubkey"]}
    if "canonicalization" in s:
        sig["canonicalization"] = s["canonicalization"]
    return {"payload": att["payload"], "signature": sig}


def _write(name: str, files: dict) -> None:
    d = _HERE / name
    d.mkdir(exist_ok=True)
    for fname, obj in files.items():
        (d / fname).write_text(json.dumps(obj, indent=2) + "\n")


def main() -> None:
    src = Path(os.environ.get("ACTA_UPSTREAM_SRC", _DEFAULT_SRC))
    a2a = src / "a2a-trust-header"

    happy = json.loads((a2a / "happy-path.json").read_text())
    att0 = happy["header_value"]["attestations"][0]
    _write(
        "acta-up-01-a2a-trusted-attestation",
        {
            "receipt.json": _passport(att0),
            "acta-keys.json": _keyset(att0),
            "expected.json": {"verdict": "PASS"},
        },
    )

    ascending = json.loads((a2a / "trust-level-ascending.json").read_text())
    att_end = ascending["header_value"]["attestations"][2]
    _write(
        "acta-up-02-a2a-trajectory-endpoint",
        {
            "receipt.json": _passport(att_end),
            "acta-keys.json": _keyset(att_end),
            "expected.json": {"verdict": "PASS"},
        },
    )

    # Real receipt mutated by flipping the first signature nibble: a third-party
    # artifact whose Ed25519 verify MUST fail, proving the oracle rejects tampering.
    tampered = _passport(att0)
    sig_hex = tampered["signature"]["sig"]
    flip = "1" if sig_hex[0] == "0" else "0"
    tampered["signature"]["sig"] = flip + sig_hex[1:]
    _write(
        "acta-up-03-a2a-tampered-sig",
        {
            "receipt.json": tampered,
            "acta-keys.json": _keyset(att0),
            "expected.json": {"verdict": "FAIL"},
        },
    )

    print("wrote acta-up-01/02/03 from", src)


if __name__ == "__main__":
    main()
