# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Seeded, locally signed receipts with split-zone floats and the one verdict every verifier owes.

Offline only (ASQAV_API_URL is a closed local port); --check never writes."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

os.environ["ASQAV_API_URL"] = "http://127.0.0.1:9"

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "verifier" / "verdict-parity-cases"
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "python" / "tests"))
from tsa_testkit import make_timestamp_resp, make_tst_info  # noqa: E402

from asqav.counterparty import verify_counterparty_binding  # noqa: E402
from asqav.verifier.verify_receipt import canonical_json  # noqa: E402

ISSUER = "Asqav Ltd"
KID = "verdict-parity-747"
VERIFIED = {"outcome": "verified", "failure_class": None}
INVALID = {"outcome": "unverified", "failure_class": "invalid"}
UNVERIFIABLE = {"outcome": "unverified", "failure_class": "unverifiable"}

#: name -> (float signed, alternate spellings, outcome, tamper outcome)
RECEIPT_CASES = {
    "small": (1e-7, ["1e-7", "0.0000001"], VERIFIED, INVALID),
    "large": (7.635507284768651e16, ["76355072847686510.0", "7.635507284768651E16"],
              UNVERIFIABLE, UNVERIFIABLE),
    "negzero": (-0.0, ["-0e0", "-0.000"], UNVERIFIABLE, UNVERIFIABLE),
    "float500": (500.0, ["5e2", "500.00"], UNVERIFIABLE, INVALID),
    "half": (0.5, ["5e-1", "0.50"], VERIFIED, INVALID),
}

#: name -> (float in the originating envelope, outcome)
COUNTERPARTY_CASES = {
    "small": (1e-7, VERIFIED),
    "large": (7.635507284768651e16, UNVERIFIABLE),
    "negzero": (-0.0, UNVERIFIABLE),
    "float500": (500.0, UNVERIFIABLE),
    "half": (0.5, VERIFIED),
}


#: name -> (signed number, [(spelling, outcome, JSON.parse outcome)]). JSON.parse
#: verifies an integer respelled as a float and refuses any -0.
RESPELLED_RECEIPTS = {
    "int500": (500, [("500.0", UNVERIFIABLE, VERIFIED), ("5e2", UNVERIFIABLE, VERIFIED)]),
    "int0": (0, [("0.0", UNVERIFIABLE, VERIFIED), ("-0", VERIFIED, UNVERIFIABLE),
                 ("-0.0", UNVERIFIABLE, UNVERIFIABLE)]),
    "negzero": (-0.0, [("-0", INVALID, UNVERIFIABLE)]),
    "poszero": (0.0, [("-0", INVALID, UNVERIFIABLE)]),
}

#: name -> (origin's signed number, spelling, outcome, JSON.parse outcome)
RESPELLED_ORIGINS = {
    "int500": (500, "500.0", UNVERIFIABLE, VERIFIED),
    "int0": (0, "-0", VERIFIED, UNVERIFIABLE),
    "negzero": (-0.0, "-0", INVALID, UNVERIFIABLE),
    "poszero": (0.0, "-0", INVALID, UNVERIFIABLE),
}


#: Integer lexemes past 2**53, with 2**53 itself as the in-range control. Raw-text
#: ingest refuses them anywhere; JSON.parse rounds them, so it sees only signed ones.
def _seeded(label: str, low: int, high: int) -> int:
    # A seeded draw in [low, high) that is stable across Python versions.
    digest = hashlib.sha256(f"asqav unsafe integer seed: {label}".encode()).digest()
    return low + int.from_bytes(digest, "big") % (high - low)


#: Signed floats past 2**53; each is respelled as its exact integer value.
UNSAFE_SIGNED = [7.635507284768651e16, -7.635507284768651e16,
                 float(_seeded("signed", 2**60, 10**20))]
UNSAFE_UNSIGNED = ["9007199254740993", "-9007199254740993", "1" + "0" * 30,
                   str(_seeded("unsigned", 2**53 + 1, 2**64))]
SAFE_BOUND = "9007199254740992"


def key(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(f"asqav verdict parity fixture seed: {label}".encode()).digest()
    )


def public_b64(label: str) -> str:
    raw = key(label).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def tsa_pem() -> str:
    pub = key("tsa").public_key()
    return pub.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()


def jwks() -> dict:
    return {"keys": [{"kid": KID, "issuer_id": ISSUER, "alg": "Ed25519", "status": "active",
                      "public_key": public_b64("issuer")}]}


def payload(name: str, x: float) -> dict:
    ctx = {"subject": f"float-{name}", "tool": "demo.action", "decision": "allow", "score": x}
    enc = canonical_json(ctx)
    digest = hashlib.sha256(enc).hexdigest()
    return {
        "type": "protectmcp:decision", "v": 1, "issued_at": "2026-09-01T12:00:00+00:00",
        "issuer_id": ISSUER, "agent_id": "agt_demo_001", "action_ref": f"sha256:{digest}",
        "context": ctx, "payload_digest": {"hash": digest, "size": len(enc)}, "risk": x,
        "policy_digest": "sha256:" + hashlib.sha256(b"fixture policy").hexdigest(),
        "previousReceiptHash": "0" * 64, "decision": "allow", "mode": "payload",
        "tool_name": "demo.action",
    }


def sign_and_anchor(body: dict) -> dict:
    env = {"payload": body, "signature": {
        "alg": "Ed25519", "kid": KID,
        "sig": base64.b64encode(key("issuer").sign(canonical_json(body))).decode(),
    }}
    tst = make_tst_info(hashlib.sha256(canonical_json(env)).digest(), "20260901120001Z")
    token = make_timestamp_resp(tst, key("tsa").sign(tst), sig_alg_oid="1.3.101.112")
    env["anchors"] = [{"type": "rfc3161", "value": base64.b64encode(token).decode()}]
    return env


def tampered(x: float) -> float:
    # One ulp up keeps a non-zero value's zone; a signed zero flips to 0.0.
    return math.nextafter(x, math.inf) if x != 0 else 0.0


def parsed_outcome(lexeme: str, outcome: dict) -> dict:
    # JSON.parse reads an integer-valued float lexeme back as an integer, so the
    # profile refusal cannot fire and the signature check fails instead; -0 is refused.
    value = float(lexeme)
    if outcome == UNVERIFIABLE and value.is_integer() and abs(value) < 2**53 and not (
        value == 0 and math.copysign(1, value) < 0
    ):
        return INVALID
    return outcome


def respell(text: str, keys: tuple[str, ...], lexeme: str, spelled: str) -> str:
    """Respell the value under each of keys, which must each occur once."""
    for name in keys:
        pattern = rf'("{name}": ){re.escape(lexeme)}(?=[,\n])'
        text, count = re.subn(pattern, lambda m: m.group(1) + spelled, text)
        assert count == 1, (name, lexeme, count)
    return text


class _NegativeZero(Exception):
    pass


def _js_number(lexeme: str) -> float | int:
    # The number JSON.parse yields, as Python canonical_json would write it.
    value = float(lexeme)
    if value == 0 and lexeme.startswith("-"):
        raise _NegativeZero
    return int(value) if value.is_integer() and abs(value) < 2**53 else value


def binding_labels(ack_text: str, origin_text: str) -> dict:
    """verify_counterparty_binding from json.loads, then from JSON.parse's view of the text."""
    ack = json.loads(ack_text)
    label = verify_counterparty_binding(ack, json.loads(origin_text)).label
    try:
        shaped = json.loads(origin_text, parse_int=_js_number, parse_float=_js_number)
    except _NegativeZero:
        return {"binding": label, "parsed_binding": "unresolved"}
    return {"binding": label, "parsed_binding": verify_counterparty_binding(ack, shaped).label}


def respelled_receipt_files() -> tuple[dict, list]:
    files, manifest = {}, []
    for name, (x, spellings) in RESPELLED_RECEIPTS.items():
        text = json.dumps(sign_and_anchor(payload(name, x)), indent=2) + "\n"
        for i, (spelled, outcome, parsed) in enumerate(spellings):
            case = f"receipt-respelled-{name}-{i}"
            files[case] = {"receipt.json": respell(text, ("risk", "score"), repr(x), spelled),
                           "jwks.json": jwks(), "tsa_trust.pem": tsa_pem()}
            manifest.append({"dir": case, "lexeme": spelled, **outcome, "parsed": parsed})
    return files, manifest


def receipt_files() -> tuple[dict, list]:
    files, manifest = {}, []
    for name, (x, alts, outcome, tamper_outcome) in RECEIPT_CASES.items():
        text = json.dumps(sign_and_anchor(payload(name, x)), indent=2) + "\n"
        lexeme = repr(x)
        assert text.count(lexeme) == 2, (name, lexeme)
        spellings = [("repr", lexeme, outcome)]
        spellings += [(f"alt{i}", a, outcome) for i, a in enumerate(alts)]
        spellings.append(("tamper", repr(tampered(x)), tamper_outcome))
        for suffix, spelled, expect in spellings:
            case = f"receipt-{name}-{suffix}"
            files[case] = {"receipt.json": text.replace(lexeme, spelled), "jwks.json": jwks(),
                           "tsa_trust.pem": tsa_pem()}
            manifest.append({"dir": case, "lexeme": spelled, **expect,
                             "parsed": parsed_outcome(spelled, expect)})
    return files, manifest


def with_note(text: str, lexeme: str) -> str:
    # An unsigned top-level member no signature or binding covers.
    head = '{\n  "payload"'
    assert text.startswith(head), text[:40]
    return f'{{\n  "note": {lexeme},\n  "payload"' + text[len(head):]


def unsafe_receipt_files() -> tuple[dict, list]:
    """Integers past 2**53 in signed and unsigned receipt positions."""
    files, manifest = {}, []
    rows = []
    for i, x in enumerate(UNSAFE_SIGNED):
        signed = json.dumps(sign_and_anchor(payload(f"unsafe{i}", x)), indent=2) + "\n"
        lex = str(int(x))
        rows.append((f"signed{i}", respell(signed, ("risk", "score"), repr(x), lex), lex,
                     UNVERIFIABLE))
    half = json.dumps(sign_and_anchor(payload("half", 0.5)), indent=2) + "\n"
    rows += [(f"unsigned{i}", with_note(half, lex), lex, VERIFIED)
             for i, lex in enumerate(UNSAFE_UNSIGNED)]
    rows.append(("unsigned-bound", with_note(half, SAFE_BOUND), SAFE_BOUND, VERIFIED))
    for suffix, text, lexeme, parsed in rows:
        case = f"receipt-unsafe-int-{suffix}"
        files[case] = {"receipt.json": text, "jwks.json": jwks(), "tsa_trust.pem": tsa_pem()}
        outcome = VERIFIED if lexeme == SAFE_BOUND else UNVERIFIABLE
        manifest.append({"dir": case, "lexeme": lexeme, **outcome, "parsed": parsed,
                         "unsafe_integer": lexeme != SAFE_BOUND})
    return files, manifest


def unsafe_counterparty_files(gcv) -> tuple[dict, list]:
    """Integers past 2**53 signed by the origin, unsigned in it, and unsigned in the ack."""
    files, manifest = {}, []
    x = UNSAFE_SIGNED[2]
    made, origin = bound_origin(gcv, x)
    rows = [("origin-signed", made, respell(origin, ("price",), repr(x), str(int(x))),
             str(int(x)), UNVERIFIABLE)]
    made, origin = bound_origin(gcv, 0.5)
    rows += [("origin-unsigned", made, with_note(origin, UNSAFE_UNSIGNED[0]), UNSAFE_UNSIGNED[0],
              UNVERIFIABLE)]
    ack = with_note(json.dumps(made["receipt.json"], indent=2) + "\n", UNSAFE_UNSIGNED[1])
    rows += [("ack-unsigned", {**made, "receipt.json": ack}, origin, UNSAFE_UNSIGNED[1],
              VERIFIED)]
    for suffix, members, origin_text, lexeme, parsed in rows:
        case = f"counterparty-unsafe-int-{suffix}"
        files[case] = {**members, "originating_envelope.json": origin_text}
        ack_text = members["receipt.json"]
        ack_text = ack_text if isinstance(ack_text, str) else _ack_text(members)
        manifest.append({"dir": case, "lexeme": lexeme, **UNVERIFIABLE, "parsed": parsed,
                         "unsafe_integer": True, **binding_labels(ack_text, origin_text)})
    return files, manifest


def _gcv():
    spec = importlib.util.spec_from_file_location(
        "gcv", ROOT / "verifier" / "generate_counterparty_vectors.py")
    gcv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gcv)
    return gcv


def bound_origin(gcv, x: float | int) -> tuple[dict, str]:
    """An acknowledgment bound to an origin that signed price x, and the origin text."""
    base_payload = gcv.payload

    def with_price(label: str) -> dict:
        body = base_payload(label)
        if label == "origin":
            body["terms"] = {"price": x}  # signed by the origin, outside its context
        return body
    gcv.payload = with_price
    try:
        made = gcv.fixtures()["asqav-31-counterparty-scope-match"]
    finally:
        gcv.payload = base_payload
    made = {k: v for k, v in made.items() if k != "expected.json"}
    return made, json.dumps(made["originating_envelope.json"], indent=2) + "\n"


def _ack_text(made: dict) -> str:
    return json.dumps(made["receipt.json"])


def counterparty_files() -> tuple[dict, list]:
    gcv = _gcv()
    files, manifest = {}, []
    for name, (x, outcome) in COUNTERPARTY_CASES.items():
        made, origin = bound_origin(gcv, x)
        lexeme = repr(x)
        assert origin.count(lexeme) == 1, (name, lexeme)
        case = f"counterparty-{name}"
        files[case] = {**made, "originating_envelope.json": origin}
        manifest.append({"dir": case, "lexeme": lexeme, **outcome,
                         "parsed": parsed_outcome(lexeme, outcome),
                         **binding_labels(_ack_text(made), origin)})
        # Tamper control: the origin's float moves after the acknowledger bound its hash.
        swapped = repr(tampered(x)) if x != 0 else "0.0"
        tampered_origin = origin.replace(lexeme, swapped)
        files[f"{case}-tamper"] = {**copy.deepcopy(made),
                                   "originating_envelope.json": tampered_origin}
        tamper_expect = UNVERIFIABLE if float(swapped).is_integer() else INVALID
        manifest.append({"dir": f"{case}-tamper", "lexeme": swapped, **tamper_expect,
                         "parsed": parsed_outcome(swapped, tamper_expect),
                         **binding_labels(_ack_text(made), tampered_origin)})
    for name, (x, spelled, outcome, parsed) in RESPELLED_ORIGINS.items():
        made, origin = bound_origin(gcv, x)
        respelled = respell(origin, ("price",), repr(x), spelled)
        case = f"counterparty-respelled-{name}"
        files[case] = {**made, "originating_envelope.json": respelled}
        manifest.append({"dir": case, "lexeme": spelled, **outcome, "parsed": parsed,
                         **binding_labels(_ack_text(made), respelled)})
    uf, um = unsafe_counterparty_files(gcv)
    return {**files, **uf}, manifest + um


def generated_files() -> dict[Path, str]:
    rf, rm = receipt_files()
    sf, sm = respelled_receipt_files()
    uf, um = unsafe_receipt_files()
    cf, cm = counterparty_files()
    out = {}
    for case, members in {**rf, **sf, **uf, **cf}.items():
        for fn, value in members.items():
            text = value if isinstance(value, str) else json.dumps(value, indent=2) + "\n"
            out[OUT / case / fn] = text
    out[OUT / "manifest.json"] = json.dumps({"cases": rm + sm + um + cm}, indent=2) + "\n"
    return out


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
    verb = "drifted" if args.check else "regenerated"
    print(f"verdict parity cases: {len(drift)} file(s) {verb}")
    if args.check and drift:
        print("\n".join(drift))
    return int(args.check and bool(drift))


if __name__ == "__main__":
    raise SystemExit(main())
