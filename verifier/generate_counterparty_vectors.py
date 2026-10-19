# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Deterministically generate signed binding fixtures; --check never writes."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "verifier" / "conformance-vectors"
sys.path.insert(0, str(ROOT / "python" / "tests"))
from tsa_testkit import make_timestamp_resp, make_tst_info  # noqa: E402

SCOPE = "envelope_minus_anchors"
CASES = (
    ("asqav-31-counterparty-scope-match", "matches", None),
    ("asqav-32-counterparty-anchors-included", "mismatch", "invalid"),
    ("asqav-33-counterparty-scope-absent", "legacy_scope", "unverifiable"),
    ("asqav-34-counterparty-scope-unknown", "unrecognised_scope", "unverifiable"),
)

#: Adversarial families whose binding deliberately disagrees with the emitted
#: origin (or omits the reference). Every ack stays well-signed so the
#: signature axis passes and only the counterparty axis fails.
NEW_CASES = (
    "asqav-35-counterparty-peer-swapped",
    "asqav-36-counterparty-origin-forged",
    "asqav-37-counterparty-divergent-terms",
    "asqav-38-counterparty-received-only",
    "asqav-39-counterparty-cross-replay",
)

#: Hole-pinning families for the still-open 731 step-3 probes. Each binding
#: rederives, so the offline verdict reads verified: the vector pins the hole.
PROBE_CASES = (
    "asqav-40-counterparty-self-binding",
    "asqav-41-counterparty-omitted-peer",
    "asqav-42-counterparty-same-binding-second-ack",
    "asqav-43-counterparty-unknown-member",
)


def jcs(value: object) -> bytes:
    """Fixture members use ASCII keys and integers; compute independently of verifiers."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def key(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(
        f"asqav public counterparty conformance fixture seed: {label}".encode()
    ).digest())


def public_key(label: str) -> bytes:
    return key(label).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def payload(label: str) -> dict:
    context = {"test": "counterparty binding", "party": label}
    encoded = jcs(context)
    digest = hashlib.sha256(encoded).hexdigest()
    return {
        "v": 1, "mode": "payload", "type": "protectmcp:decision",
        "issuer_id": label, "agent_id": f"agent_{label}", "action_id": f"action_{label}",
        "issued_at": "2026-09-01T12:00:00+00:00", "previousReceiptHash": "0" * 64,
        "action_ref": f"sha256:{digest}", "context": context,
        "payload_digest": {"hash": digest, "size": len(encoded)},
        "policy_digest": "sha256:" + hashlib.sha256(b"fixture policy").hexdigest(),
        "decision": "allow",
    }


def sign_and_anchor(body: dict, *, gen_time: str = "20260901120001Z") -> dict:
    issuer = body["issuer_id"]
    envelope = {"payload": body, "signature": {
        "alg": "Ed25519", "kid": issuer,
        "sig": base64.b64encode(key(issuer).sign(jcs(body))).decode(),
    }}
    tst = make_tst_info(hashlib.sha256(jcs(envelope)).digest(), gen_time)
    token = make_timestamp_resp(tst, key("tsa").sign(tst), sig_alg_oid="1.3.101.112")
    envelope["anchors"] = [{"type": "rfc3161", "value": base64.b64encode(token).decode()}]
    return envelope


def fixtures() -> dict[str, dict[str, object]]:
    origin = sign_and_anchor(payload("origin"))
    origin["anchors"] += sign_and_anchor(payload("origin"), gen_time="20260901120002Z")["anchors"]
    digest = base64.b64encode(hashlib.sha256(jcs({
        "payload": origin["payload"], "signature": origin["signature"],
    })).digest()).decode()
    files = {}
    for name, label, failure in CASES:
        binding = {"scope": SCOPE, "receipt_ref": "sig_counterparty_origin", "envelope_hash": digest,
                   "expect_ack_from": "ack"}
        if label == "mismatch":
            binding["envelope_hash"] = base64.b64encode(hashlib.sha256(jcs(origin)).digest()).decode()
        elif label == "legacy_scope":
            del binding["scope"]
        elif label == "unrecognised_scope":
            binding["scope"] = "unsupported_fixture_scope"
        body = payload("ack")
        body.update(type="protectmcp:acknowledgment", counterparty_binding=binding)
        expected = {"format": "asqav-native", "outcome": "unverified" if failure else "verified",
                    "reason_code": "" if label == "matches" else f"counterparty_{label}",
                    "notes": f"Signed acknowledgment with binding result {label}; signature and local TSA verify."}
        if failure:
            expected["failure_class"] = failure
        files[name] = {
            "receipt.json": sign_and_anchor(body), "originating_envelope.json": copy.deepcopy(origin),
            "jwks.json": {"keys": [{"kid": who, "issuer_id": who, "alg": "Ed25519", "status": "active",
                                    "public_key": base64.b64encode(public_key(who)).decode()}
                                   for who in ("origin", "ack")]},
            "expected.json": expected,
            "tsa_trust.pem": key("tsa").public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode(),
        }
    files.update(_adversarial_fixtures(origin, digest))
    files.update(_probe_fixtures(origin, digest))
    return files


def _adversarial_fixtures(origin: dict, digest: str) -> dict[str, dict[str, object]]:
    """Signed dirs for bindings that deliberately disagree with the emitted origin."""
    out: dict[str, dict[str, object]] = {}

    def emit(name: str, binding: dict, emitted_origin: dict, label: str) -> None:
        body = payload("ack")
        body.update(type="protectmcp:acknowledgment", counterparty_binding=binding)
        out[name] = {
            "receipt.json": sign_and_anchor(body),
            "originating_envelope.json": copy.deepcopy(emitted_origin),
            "jwks.json": {"keys": [{"kid": who, "issuer_id": who, "alg": "Ed25519", "status": "active",
                                        "public_key": base64.b64encode(public_key(who)).decode()}
                                       for who in ("origin", "ack")]},
            "expected.json": {
                "format": "asqav-native", "outcome": "unverified",
                "reason_code": f"counterparty_{label}", "failure_class": "invalid",
                "notes": f"Signed acknowledgment with binding result {label}; signature and local TSA verify.",
            },
            "tsa_trust.pem": key("tsa").public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode(),
        }

    base = {"scope": SCOPE, "receipt_ref": "sig_counterparty_origin",
            "envelope_hash": digest, "expect_ack_from": "ack"}
    emit(NEW_CASES[0], {**base, "expect_ack_from": "peer"}, origin, "kid_mismatch")
    forged = copy.deepcopy(origin)
    forged["payload"]["action_id"] = "action_forged_after_signing"
    emit(NEW_CASES[1], dict(base), forged, "mismatch")
    other = sign_and_anchor(payload("origin-b"))
    emit(NEW_CASES[2], dict(base), other, "mismatch")
    received = {k: v for k, v in base.items() if k not in ("receipt_ref", "envelope_hash")}
    emit(NEW_CASES[3], received, origin, "malformed")
    replay_terms = {"payload": {"terms": "other-interaction"}, "signature": origin["signature"]}
    replay_hash = base64.b64encode(hashlib.sha256(jcs(replay_terms)).digest()).decode()
    emit(NEW_CASES[4], {**base, "receipt_ref": "sig_other_interaction",
                        "envelope_hash": replay_hash}, origin, "mismatch")
    return out


def _probe_fixtures(origin: dict, digest: str) -> dict[str, dict[str, object]]:
    """Verified dirs pinning the still-open 731 step-3 admit holes offline."""
    out: dict[str, dict[str, object]] = {}

    def emit_verified(name: str, body: dict, emitted_origin: dict, notes: str) -> None:
        out[name] = {
            "receipt.json": sign_and_anchor(body),
            "originating_envelope.json": copy.deepcopy(emitted_origin),
            "jwks.json": {"keys": [{"kid": who, "issuer_id": who, "alg": "Ed25519", "status": "active",
                                        "public_key": base64.b64encode(public_key(who)).decode()}
                                       for who in ("origin", "ack")]},
            "expected.json": {"format": "asqav-native", "outcome": "verified",
                              "reason_code": "", "notes": notes},
            "tsa_trust.pem": key("tsa").public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode(),
        }

    base = {"scope": SCOPE, "receipt_ref": "sig_counterparty_origin",
            "envelope_hash": digest, "expect_ack_from": "ack"}
    self_body = payload("origin")
    self_body.update(type="protectmcp:acknowledgment", action_id="action_origin_self_ack",
                     counterparty_binding={**base, "expect_ack_from": "origin"})
    emit_verified(PROBE_CASES[0], self_body, origin,
                  "Self-binding verifies: origin and acknowledger are the same party and no axis compares them.")
    omitted = {k: v for k, v in base.items() if k != "expect_ack_from"}
    omitted_body = payload("ack")
    omitted_body.update(type="protectmcp:acknowledgment", counterparty_binding=omitted)
    emit_verified(PROBE_CASES[1], omitted_body, origin,
                  "Omitted peer verifies: without expect_ack_from the kid cross-check is skipped.")
    replay_body = payload("ack")
    replay_body.update(type="protectmcp:acknowledgment", action_id="action_ack_second_look",
                       counterparty_binding=dict(base))
    emit_verified(PROBE_CASES[2], replay_body, origin,
                  "Same binding bytes verify a second ack: one origin corroborates two acknowledgments.")
    unknown = dict(base, bogus_future_member="must-understand")
    unknown_body = payload("ack")
    unknown_body.update(type="protectmcp:acknowledgment", action_id="action_ack_unknown_member",
                        counterparty_binding=unknown)
    emit_verified(PROBE_CASES[3], unknown_body, origin,
                  "Unknown binding member verifies: the axis reads known keys and ignores the rest.")
    return out


def generated_files() -> dict[Path, str]:
    output = {}
    for name, files in fixtures().items():
        for filename, value in files.items():
            output[CORPUS / name / filename] = value if isinstance(value, str) else json.dumps(value, indent=2) + "\n"
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    drift = []
    for path, text in generated_files().items():
        if not path.exists() or path.read_text() != text:
            drift.append(str(path.relative_to(ROOT)))
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
    print(f"counterparty fixtures: {len(CASES) + len(NEW_CASES) + len(PROBE_CASES)} cases; {len(drift)} file(s) {'drifted' if args.check else 'regenerated'}")
    if args.check and drift:
        print("\n".join(drift))
    return int(args.check and bool(drift))


if __name__ == "__main__":
    raise SystemExit(main())
