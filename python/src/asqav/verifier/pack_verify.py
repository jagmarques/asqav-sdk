"""Offline verification of a whole exported Audit Pack, reusing verify_receipt for every check.
Nothing here reaches the network."""

from __future__ import annotations

import hashlib

from asqav.verifier import verify_receipt as vr

VERIFIED = "verified"
NOT_OFFLINE = "not_verifiable_offline"
INVALID = "invalid"

#: Process exit code per pack status: 0 all verified, 2 some not verifiable offline, 1 any invalid.
EXIT_CODES = {VERIFIED: 0, NOT_OFFLINE: 2, INVALID: 1}
_RANK = {VERIFIED: 0, NOT_OFFLINE: 1, INVALID: 2}

#: Optional pack members the bundle signature covers whenever they are carried.
_PRESENCE_MEMBERS = (
    "content_disclosure",
    "regime_mapping_disclaimer",
    "regime_provenance",
    "jwks",
    "exit_manifest",
    "tsa_trust",
)

_FLAT_REASON = (
    "non-compliance-mode receipt: it has no signed receipt envelope "
    "(issuer, chain, expiry), so it cannot be verified offline"
)
_WITHHELD_REASON = (
    "signed bytes withheld at export; re-export with --include-signed-bytes to verify it"
)


def signed_bundle_payload(pack: dict) -> dict:
    """The projection of the pack that the bundle signature covers (export route contract)."""
    receipts = pack.get("receipts") or []
    payload = {
        "organization_id": pack.get("organization_id"),
        "start": pack.get("start"),
        "end": pack.get("end"),
        "agent_id": pack.get("agent_id"),
        "only_compliance": pack.get("only_compliance"),
        "algorithm_registry_version": pack.get("algorithm_registry_version"),
        "receipt_count": pack.get("receipt_count", len(receipts)),
        "receipts": receipts,
        "revocation_manifest": pack.get("revocation_manifest") or [],
        "regime_mapping": pack.get("regime_mapping") or {},
    }
    for key in _PRESENCE_MEMBERS:
        if pack.get(key) is not None:
            payload[key] = pack[key]
    if pack.get("attested_framework_coverage"):
        payload["attested_framework_coverage"] = pack["attested_framework_coverage"]
    return payload


def pack_tsa_keys(pack: dict) -> list:
    """TSA certificates or keys the pack carries under ``tsa_trust`` (PEM or base64)."""
    raw = pack.get("tsa_trust")
    items = raw if isinstance(raw, list) else [raw]
    return [vr.tsa_key_material(i.encode()) for i in items if isinstance(i, str) and i.strip()]


def _key_bytes(entry) -> bytes | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("public_key") or entry.get("pub")
    try:
        return vr._b64decode(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _jwks_keys(jwks) -> list:
    keys = jwks.get("keys") if isinstance(jwks, dict) else None
    return [k for k in keys if isinstance(k, dict)] if isinstance(keys, list) else []


def _outcome(status: str, reason: str) -> dict:
    return {"status": status, "reason": reason}


def check_bundle(pack: dict, jwks: dict) -> dict:
    """Recompute the bundle digest, verify its signature, and tie the key to the key set."""
    digest = pack.get("bundle_digest")
    sig = pack.get("bundle_signature")
    key_b64 = pack.get("bundle_public_key")
    if not (isinstance(digest, str) and isinstance(sig, str) and isinstance(key_b64, str)):
        return _outcome(NOT_OFFLINE, "pack carries no bundle signature")
    try:
        msg = vr.canonical_json(signed_bundle_payload(pack))
        key, sig_bytes = vr._b64decode(key_b64), vr._b64decode(sig)
    except (ValueError, TypeError, RecursionError) as exc:
        return _outcome(INVALID, f"bundle cannot be read: {exc}")
    if "sha256:" + hashlib.sha256(msg).hexdigest() != digest:
        return _outcome(INVALID, "bundle_digest does not match the pack contents")
    alg = pack.get("bundle_signature_algorithm") or "ML-DSA-65"
    result, note = vr.verify_signature(key, msg, sig_bytes, alg)
    if result != "PASS":
        return _outcome(INVALID if result == "FAIL" else NOT_OFFLINE, f"bundle signature: {note}")
    entry = next((k for k in _jwks_keys(jwks) if _key_bytes(k) == key), None)
    if entry is None:
        return _outcome(NOT_OFFLINE, "bundle signature valid, but its key is not in the key set")
    org = pack.get("organization_id")
    if org and entry.get("org_id") and entry.get("org_id") != org:
        return _outcome(INVALID, "bundle signing key belongs to another organization")
    return _outcome(VERIFIED, f"bundle signature valid under key {entry.get('kid')}")


def _chain_key(payload: dict) -> tuple:
    seq = payload.get("seq")
    ordinal = seq if isinstance(seq, int) and not isinstance(seq, bool) else float("inf")
    return ordinal, str(payload.get("issued_at", ""))


def _consecutive(prev: dict, cur: dict) -> bool:
    a, b = prev.get("seq"), cur.get("seq")
    if isinstance(a, int) and isinstance(b, int):
        return b == a + 1
    return True


def chain_predecessors(receipts: list) -> dict:
    """Index of each envelope receipt -> its predecessor payload in the pack, per agent chain."""
    chains: dict[str, list[int]] = {}
    for i, entry in enumerate(receipts):
        payload = entry.get("payload") if isinstance(entry, dict) else None
        if isinstance(payload, dict) and not entry.get("signed_bytes_withheld"):
            agent = str(payload.get("agent_id") or entry.get("agent_id") or "")
            chains.setdefault(agent, []).append(i)
    preds: dict[int, dict | None] = {}
    for idxs in chains.values():
        idxs.sort(key=lambda i: _chain_key(receipts[i]["payload"]))
        preds[idxs[0]] = None
        for prev, cur in zip(idxs, idxs[1:]):
            p, c = receipts[prev]["payload"], receipts[cur]["payload"]
            preds[cur] = p if _consecutive(p, c) else None
    return preds


def _expiry_note(payload: dict, axes: list) -> str | None:
    axis = next((a for a in axes if a["name"] == "expiry"), None)
    if axis is None or axis["result"] != "FAIL" or "lapsed" not in axis["note"]:
        return None
    issued, expires = vr._parse_stamp(payload.get("issued_at")), vr._parse_stamp(payload.get("expires_at"))
    if issued is not None and expires is not None and issued <= expires:
        return f"expires_at {payload.get('expires_at')} has passed; signed before it, so not a failure"
    return f"expires_at {payload.get('expires_at')} is not after issued_at"


def _blocking_reasons(status: str, axes: list) -> list:
    if status == INVALID:
        rows = [a for a in axes if a["failure_class"] == INVALID and a["name"] != "expiry"]
    else:
        rows = [
            a for a in axes
            if a["result"] != "PASS" and a["name"] != "expiry"
            and not (a["name"] == "chain" and a["result"] == "SKIPPED")
        ]
    return [f"{a['name']}: {' '.join(a['note'].split())}" for a in rows]


def _envelope_result(entry: dict, jwks: dict, predecessor, tsa_keys, headers) -> dict:
    res = vr.run_structured(
        entry, jwks, predecessor, trusted_tsa_keys=tsa_keys, bitcoin_headers=headers
    )
    if res["verdict"] in (vr.VERDICT_VERIFIED, vr.VERDICT_VERIFIED_KEYED):
        status = VERIFIED
    else:
        status = INVALID if res["failure_class"] == vr.FAILURE_INVALID else NOT_OFFLINE
    axis = {a["name"]: a for a in res["axes"]}
    payload = entry.get("payload")
    expiry = _expiry_note(payload if isinstance(payload, dict) else {}, res["axes"])
    return {
        "status": status,
        "verdict": res["verdict"],
        "reasons": _blocking_reasons(status, res["axes"]),
        "signature": axis["signature"]["note"] if "signature" in axis else None,
        "chain": axis["chain"]["note"] if "chain" in axis else None,
        "notes": [expiry] if expiry else [],
        "axes": res["axes"],
    }


def _bare(status: str, reason: str) -> dict:
    return {"status": status, "reasons": [reason], "signature": None, "chain": None, "notes": []}


def _flat_signature_results(entry: dict, jwks: dict) -> list:
    agent = entry.get("agent_id")
    keys = [k for k in _jwks_keys(jwks) if agent and k.get("agent_id") == agent]
    msg, sig = entry.get("message_b64"), entry.get("signature_b64")
    if not (keys and isinstance(msg, str) and msg and isinstance(sig, str) and sig):
        return []
    try:
        msg_bytes, sig_bytes = vr._b64decode(msg), vr._b64decode(sig)
    except ValueError:
        return []
    alg = entry.get("algorithm")
    return [vr.verify_signature(_key_bytes(k) or b"", msg_bytes, sig_bytes, alg)[0] for k in keys]


def _flat_result(entry: dict, jwks: dict) -> dict:
    """A flat receipt is never verified offline; a signature its agent's keys all reject is invalid."""
    out = _bare(NOT_OFFLINE, _FLAT_REASON)
    results = _flat_signature_results(entry, jwks)
    if "PASS" in results:
        out["signature"] = "signature valid under the agent's published key"
    elif results and all(r == "FAIL" for r in results):
        out["status"] = INVALID
        out["reasons"] = [f"signature does not verify under any key of agent {entry.get('agent_id')}"]
    return out


def verify_entry(entry, jwks: dict, predecessor=None, *, tsa_keys=None, headers=None) -> dict:
    """Status, reasons and notes for one pack entry."""
    if not isinstance(entry, dict):
        return _bare(INVALID, "entry is not a JSON object")
    head = {"signature_id": entry.get("signature_id"), "agent_id": entry.get("agent_id")}
    if entry.get("signed_bytes_withheld"):
        return {**head, **_bare(NOT_OFFLINE, _WITHHELD_REASON)}
    if isinstance(entry.get("payload"), dict):
        return {**head, **_envelope_result(entry, jwks, predecessor, tsa_keys, headers)}
    if entry.get("compliance_mode") is False:
        return {**head, **_flat_result(entry, jwks)}
    return {**head, **_envelope_result(entry, jwks, None, tsa_keys, headers)}


def _worst(statuses) -> str:
    return max(statuses, key=_RANK.__getitem__, default=VERIFIED)


def verify_pack(pack: dict, *, jwks: dict | None = None, tsa_keys=None, bitcoin_headers=None) -> dict:
    """Per-receipt and pack report; ``jwks`` replaces the pack's key snapshot, ``tsa_keys`` adds
    to the pack's TSA material. ``exit_code`` is 0 all verified, 2 some not offline, 1 invalid."""
    key_set = jwks if jwks is not None else pack.get("jwks")
    key_set = key_set if isinstance(key_set, dict) else {"keys": []}
    pinned = pack_tsa_keys(pack) + list(tsa_keys or [])
    receipts = pack.get("receipts") if isinstance(pack.get("receipts"), list) else []
    preds = chain_predecessors(receipts)
    results = [
        verify_entry(r, key_set, preds.get(i), tsa_keys=pinned, headers=bitcoin_headers)
        for i, r in enumerate(receipts)
    ]
    bundle = check_bundle(pack, key_set)
    counts = {s: sum(1 for r in results if r["status"] == s) for s in (VERIFIED, NOT_OFFLINE, INVALID)}
    status = _worst([bundle["status"]] + [r["status"] for r in results])
    return {
        "status": status,
        "exit_code": EXIT_CODES[status],
        "bundle": bundle,
        "receipts": results,
        "summary": {"total": len(results), **counts},
        "trust": {
            "keys": "supplied key set" if jwks is not None else "pack key snapshot",
            "key_count": len(_jwks_keys(key_set)),
            "tsa_keys": len(pinned),
        },
        "pack": {
            "organization_id": pack.get("organization_id"),
            "start": pack.get("start"),
            "end": pack.get("end"),
            "bundle_digest": pack.get("bundle_digest"),
        },
    }
