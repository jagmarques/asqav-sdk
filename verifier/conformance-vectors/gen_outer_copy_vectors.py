"""Generate the outer-copy conformance vectors (criterion 742).

An Audit Pack entry restates signed values (agent, action, time, signature
string, key, signed bytes) as plain top-level members no signature covers.
asqav-43 is a real prod receipt with those copies added by the export rule, all
agreeing, so the family has a verified control. asqav-44..48 each edit one copy
and leave every signed byte alone: the signature still verifies, and every
verifier must still refuse the entry because a copy disagrees with the signed
payload.

The source is the payload-mode prod receipt of asqav-06, whose signed payload
carries agent_id, action_id, action_type and issued_at. The export-side copies
are derived from that signed payload exactly as the export derives them from
the stored row. The hash-mode prod export of 2026-09-02 (no signed action_type)
is exercised by python/tests/test_outer_copies.py instead.

Usage: python verifier/conformance-vectors/gen_outer_copy_vectors.py
Re-freeze the corpus lock afterwards: python verifier/freeze_corpus_lock.py
"""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PAYLOAD_VECTOR = _HERE / "asqav-06-mldsa65-payload-prod"

REASON = "outer_copy_mismatch"


def _jcs(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _agreeing_entry() -> tuple[dict, dict]:
    """asqav-06 plus the export-side copies, derived from its own signed payload."""
    base = _load(_PAYLOAD_VECTOR / "receipt.json")
    jwks = _load(_PAYLOAD_VECTOR / "jwks.json")
    payload, sig = base["payload"], base["signature"]
    entry = {
        "agent_id": payload["agent_id"],
        "action_id": payload["action_id"],
        "action_type": payload["action_type"],
        "algorithm": sig["alg"],
        "signed_at": payload["issued_at"].rstrip("Z"),
        "compliance_mode": True,
        "payload": payload,
        "signature": sig,
        "anchors": base.get("anchors", []),
        "signature_b64": sig["sig"],
        "public_key_b64": jwks["keys"][0]["public_key"],
        "message_b64": base64.b64encode(_jcs(payload)).decode(),
    }
    return entry, jwks


def _flip_last_byte(b64: str) -> str:
    raw = bytearray(base64.b64decode(b64))
    raw[-1] ^= 0x01
    return base64.b64encode(bytes(raw)).decode()


def _vectors() -> list[tuple[str, dict, dict, dict]]:
    agree, jwks = _agreeing_entry()
    src = (
        "Real prod payload-mode receipt (asqav-06, ML-DSA-65) with the export-side copies "
        "added from its own signed payload. "
    )
    out = [("asqav-43-outer-copies-agree", agree, jwks, {
        "format": "asqav-native", "outcome": "verified", "reason_code": "",
        "notes": src + "Every plain copy agrees with the signed payload, so the outer_copies "
        "axis passes. signature_id is not carried (no signed counterpart). Control for asqav-44..48.",
    })]

    def edited(member, value, name, what):
        entry = copy.deepcopy(agree)
        entry[member] = value
        out.append((name, entry, jwks, {
            "format": "asqav-native", "outcome": "unverified", "failure_class": "invalid",
            "reason_code": REASON,
            "notes": src + what + " The signed bytes are untouched, so the signature axis "
            "still passes; the outer_copies axis FAILs and the verdict is unverified/invalid.",
        }))

    other = copy.deepcopy(agree["payload"])
    other["agent_id"] = "agt_someoneElse000000"
    edited("signed_at", "2026-03-03T00:00:00", "asqav-44-outer-signed-at-edited",
           "The outer signed_at is moved to 3 March while the signed issued_at stays in June.")
    edited("agent_id", "agt_someoneElse000000", "asqav-45-outer-agent-id-edited",
           "The outer agent_id names a different agent than the signed payload.")
    edited("action_type", "payment:release", "asqav-46-outer-action-type-edited",
           "The outer action_type differs from the signed action_type.")
    edited("message_b64", base64.b64encode(_jcs(other)).decode(),
           "asqav-47-outer-message-copy-edited",
           "The outer message_b64 carries bytes that are not the canonical signed payload.")
    edited("public_key_b64", _flip_last_byte(agree["public_key_b64"]),
           "asqav-48-outer-public-key-edited",
           "The outer public_key_b64 is not the key the signature verified against.")
    return out


def _write(name: str, entry: dict, jwks: dict, expected: dict) -> None:
    out = _HERE / name
    out.mkdir(parents=True, exist_ok=True)
    for fname, doc in (("receipt.json", entry), ("jwks.json", jwks), ("expected.json", expected)):
        (out / fname).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _update_manifest(rows: list[tuple[str, dict, dict, dict]]) -> None:
    path = _HERE / "manifest.json"
    manifest = [m for m in json.loads(path.read_text(encoding="utf-8"))
                if m["dir"] not in {r[0] for r in rows}]
    for name, _entry, _jwks, expected in rows:
        row = {"dir": name, **expected}
        row.setdefault("failure_class", "")
        if not row["failure_class"]:
            del row["failure_class"]
        manifest.append(row)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    rows = sorted(_vectors(), key=lambda r: r[0])
    for name, entry, jwks, expected in rows:
        _write(name, entry, jwks, expected)
        print("wrote", name)
    _update_manifest(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
