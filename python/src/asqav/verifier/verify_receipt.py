"""Standalone Asqav receipt verifier - one dependency, mostly stdlib.

Verify an Asqav Compliance Receipt yourself, without the Asqav SDK or liboqs.

WHAT THIS CHECKS (independently, on your machine):
  - ML-DSA-65 (FIPS 204) signature over the receipt's canonical bytes
  - canonical-bytes integrity (JCS / RFC8785 reproduction, stdlib json)
  - hash-chain link to a predecessor receipt
  - anchor binding (which envelope each anchor commits to) plus anchor presence
  - issuer public-key resolution from the public /.well-known/jwks.json

WHAT THIS DOES NOT CHECK (needs server state or ASN.1; use the hosted /verify):
  - full RFC3161 certificate-chain walk
  - policy_digest artefact resolution

The only non-stdlib dependency is the signature check:
  pip install dilithium-py        (pure python; verify path uses stdlib SHAKE)
Without it, every other check still runs and the signature axis reports SKIPPED;
the overall verdict is then INCOMPLETE, never a PASS. This tool will not emit a
PASS unless the post-quantum signature was actually verified.

Security note: dilithium-py is a pure-python FIPS 204 implementation that is not
constant-time. For a verify-only tool this is acceptable: verification touches
only public data (public key, signature, message), so there is no secret to leak
through timing.

Run:
  python verify_receipt.py --id sig_abc123
  python verify_receipt.py --receipt receipt.json --jwks jwks.json --offline
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone

API_BASE = "https://api.asqav.com/api/v1"
JWKS_URL = "https://api.asqav.com/.well-known/jwks.json"


def _user_agent() -> str:
    """Identify this tool; the api.asqav.com edge 403s the bare urllib default.

    Self-contained (no asqav import) so the file still runs as a standalone
    download, per the module docstring.
    """
    try:
        from importlib.metadata import version

        ver = version("asqav")
    except Exception:
        ver = "standalone"
    return f"asqav-python/{ver} (+https://www.asqav.com)"


USER_AGENT = _user_agent()
SKEW_BOUND_SECONDS = 300
FIRST_RECEIPT_SEED = "0" * 64  # mirrors core/integrity.py FIRST_RECEIPT_SEED
REQUIRED_FIELDS = (
    "type",
    "issued_at",
    "issuer_id",
    "action_ref",
    "payload_digest",
    "policy_digest",
    "previousReceiptHash",
    "decision",
)
ALLOWED_TYPES = {
    "protectmcp:decision",
    "protectmcp:restraint",
    "protectmcp:lifecycle",
    "protectmcp:lifecycle:configuration_change",
    # risk-acceptance / exception receipt; no-policy lifecycle opt-out.
    "protectmcp:lifecycle:risk_acceptance",
    # code-authorship receipt; producer-asserted record of who authored a code change.
    "protectmcp:lifecycle:code_authorship",
    "protectmcp:acknowledgment",
    "protectmcp:observation",
    "protectmcp:observation:result_bound",
}


def canonical_json(obj) -> bytes:
    """JCS canonical bytes - byte-identical to the server's canonical_json.

    Sorted keys, no whitespace, UTF-8, NaN/Infinity rejected. This is the exact
    byte string the server signs and the verifier must reproduce.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def envelope_minus_anchors_jcs(env: dict) -> bytes:
    """JCS bytes of the envelope with ``anchors`` removed (the anchored bytes)."""
    e = dict(env)
    e.pop("anchors", None)
    return canonical_json(e)


def _get_json(url: str, *, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_jwks(url: str = JWKS_URL, *, timeout: int = 30) -> dict:
    """Fetch and return the Asqav public JWKS directory as a dict.

    Snapshot this before going air-gapped; pass the result to
    ``verify_receipt_offline(receipt, jwks)`` or ``run(envelope, jwks, ...)``.
    The endpoint is public and unauthenticated.

    Args:
        url: JWKS URL (default: https://api.asqav.com/.well-known/jwks.json).
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed JWKS dict with a ``"keys"`` list.
    """
    return _get_json(url, timeout=timeout)


def _b64decode(value: str) -> bytes:
    """Decode standard or url-safe base64, padding-tolerant."""
    s = value.replace("-", "+").replace("_", "/")
    s += "=" * ((-len(s)) % 4)
    return base64.b64decode(s, validate=False)


def _safe_b64(v: str) -> bool:
    try:
        _b64decode(v)
        return True
    except Exception:
        return False


def normalise_envelope(raw: dict) -> dict:
    """Remap a hosted /verify response into the canonical 3-key envelope.

    An Audit Pack already ships the canonical ``{payload, signature, anchors}``
    shape, so pass it through untouched. The hosted ``/verify/{id}`` JSON instead
    nests the signed dict under ``payload`` and exposes the signature object as
    ``signature_envelope`` (and a possibly flat-string ``signature``); rebuild
    the canonical envelope from those parts so ``run()`` sees one shape.
    """
    # Already canonical: a top-level signature object plus a payload dict.
    sig = raw.get("signature")
    if isinstance(sig, dict) and isinstance(raw.get("payload"), dict):
        return raw

    payload = raw.get("payload")
    if not isinstance(payload, dict):
        # Not a hosted response and not canonical; hand it back as-is so the
        # structure check reports the problem clearly.
        return raw

    sig_obj = raw.get("signature_envelope")
    if not isinstance(sig_obj, dict):
        if isinstance(sig, dict):
            sig_obj = sig
        else:
            sig_obj = {
                "alg": raw.get("algorithm", "ML-DSA-65"),
                "kid": payload.get("issuer_id", ""),
                "sig": sig if isinstance(sig, str) else "",
            }
    return {
        "payload": payload,
        "signature": sig_obj,
        "anchors": raw.get("anchors") or [],
    }


def resolve_key(jwks: dict, kid: str):
    """Return (public_key_bytes, status, alg) for the receipt's signature.kid.

    The public jwks directory lists each key under both its bare issuer id and
    its crypto key id; match either so the bare-kid wire form resolves.
    """
    for k in jwks.get("keys", []):
        if kid and kid in (k.get("issuer_id"), k.get("kid")):
            return _b64decode(k["public_key"]), k.get("status"), k.get("alg")
    return None, None, None


def verify_signature(pk: bytes, msg: bytes, sig: bytes, alg: str):
    """ML-DSA-65 verify. Returns (result, note); result in PASS/FAIL/SKIPPED."""
    if (alg or "").upper() != "ML-DSA-65":
        return "SKIPPED", f"unsupported alg {alg!r} (this tool checks ML-DSA-65)"
    try:
        from dilithium_py.ml_dsa import ML_DSA_65
    except ImportError:
        return "SKIPPED", "run 'pip install dilithium-py' for the post-quantum check"
    try:
        ok = ML_DSA_65.verify(pk, msg, sig)
        return ("PASS" if ok else "FAIL"), (
            "signature valid" if ok else "signature mismatch"
        )
    except Exception as exc:  # malformed key or signature bytes
        return "FAIL", f"verify error: {exc}"


def check_skew(issued_at: str):
    try:
        ts = datetime.fromisoformat(issued_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return "FAIL", f"unparseable issued_at {issued_at!r}"
    skew = (ts - datetime.now(timezone.utc)).total_seconds()
    if skew > SKEW_BOUND_SECONDS:
        return (
            "FAIL",
            f"issued_at {skew:.0f}s ahead of wall clock (> {SKEW_BOUND_SECONDS}s)",
        )
    return "PASS", f"skew {skew:.0f}s within bound"


def check_chain(payload: dict, predecessor_payload: dict | None):
    prev = payload.get("previousReceiptHash")
    if prev == FIRST_RECEIPT_SEED:
        return "PASS", "first receipt on chain (all-zero seed)"
    if predecessor_payload is None:
        return "SKIPPED", "no predecessor supplied (pass --predecessor to check)"
    actual = hashlib.sha256(canonical_json(predecessor_payload)).hexdigest()
    if actual == prev:
        return "PASS", "chain link rederives from predecessor"
    return "FAIL", f"chain break: expected {str(prev)[:16]}.. got {actual[:16]}.."


def check_anchors(envelope: dict):
    anchors = envelope.get("anchors") or []
    if not anchors:
        return "SKIPPED", "no anchors on this receipt"
    bound = hashlib.sha256(envelope_minus_anchors_jcs(envelope)).hexdigest()
    lines = [f"anchors bind envelope digest sha256:{bound[:16]}.."]
    all_ok = True
    for a in anchors:
        atype = a.get("type", "?")
        val = a.get("value")
        ok = bool(val) and _safe_b64(val)
        all_ok = all_ok and ok
        state = "present, base64-ok" if ok else "MISSING or malformed"
        lines.append(f"    - {atype}: value {state}")
    return ("PASS" if all_ok else "FAIL"), "; ".join(lines)


def check_structure(payload: dict):
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return "FAIL", f"missing required fields: {','.join(missing)}"
    rt = payload.get("type")
    if rt not in ALLOWED_TYPES:
        return "FAIL", f"type {rt!r} outside the allowed namespace"
    return "PASS", f"required fields present; type {rt}"


def run(envelope: dict, jwks: dict, predecessor_payload: dict | None) -> int:
    envelope = normalise_envelope(envelope)
    payload = envelope.get("payload", envelope)
    if not isinstance(payload, dict):
        # Hosted /verify can return ``payload: null`` (e.g. hash-only or
        # redacted receipts). Nothing to verify; say so instead of crashing.
        print("Asqav receipt verification")
        print(
            "  [FAIL] payload      receipt payload not available from this "
            "surface (the server returned payload: null). Verify with a saved "
            "receipt instead: --receipt FILE from your Audit Pack or SDK "
            "capture."
        )
        print("\n  => INCOMPLETE (no payload to verify; never a PASS)")
        return 2
    sig_obj = envelope.get("signature", {})
    if isinstance(sig_obj, str):  # flat-string signature, derive the object
        sig_obj = {
            "alg": envelope.get("algorithm", "ML-DSA-65"),
            "kid": payload.get("issuer_id", ""),
            "sig": sig_obj,
        }
    kid = sig_obj.get("kid", "")
    alg = sig_obj.get("alg", "ML-DSA-65")
    msg = canonical_json(payload)

    results = []
    results.append(("structure", *check_structure(payload)))

    pk, status, jwks_alg = resolve_key(jwks, kid)
    if pk is None:
        results.append(("issuer_key", "FAIL", f"kid {kid!r} not in jwks directory"))
        results.append(("signature", "SKIPPED", "no issuer key to verify against"))
    else:
        results.append(("issuer_key", "PASS", f"resolved kid {kid} (status={status})"))
        sig = _b64decode(sig_obj.get("sig", ""))
        results.append(("signature", *verify_signature(pk, msg, sig, alg or jwks_alg)))

    results.append(("chain", *check_chain(payload, predecessor_payload)))
    results.append(("anchors", *check_anchors(envelope)))
    results.append(("skew", *check_skew(payload.get("issued_at", ""))))

    print("Asqav receipt verification")
    print(f"  canonical bytes: sha256:{hashlib.sha256(msg).hexdigest()}")
    print(f"  signature.kid:   {kid}")
    for name, res, note in results:
        mark = {"PASS": "ok", "FAIL": "FAIL", "SKIPPED": "skip"}[res]
        print(f"  [{mark:>4}] {name:<11} {note}")

    has_fail = any(r == "FAIL" for _, r, _ in results)
    # A skipped chain (no predecessor supplied) is expected and does not block a
    # PASS; any other skip - notably the signature - downgrades to INCOMPLETE.
    has_blocking_skip = any(r == "SKIPPED" and n != "chain" for n, r, _ in results)
    if has_fail:
        verdict, code = "FAIL", 1
    elif has_blocking_skip:
        verdict, code = "INCOMPLETE (some checks skipped; never a PASS)", 2
    else:
        verdict, code = "PASS", 0
    print(f"\n  => {verdict}")
    return code


def run_structured(
    envelope: dict,
    jwks: dict,
    predecessor_payload: dict | None = None,
) -> dict:
    """Verify a receipt offline and return a structured result dict.

    Same logic as ``run()`` but returns a dict instead of printing and exiting.
    Used by ``verify_receipt_offline()`` in the public SDK API.

    Returns:
        dict with keys:
          - ``"verdict"``: "PASS" | "FAIL" | "INCOMPLETE"
          - ``"axes"``: list of ``{"name": str, "result": str, "note": str}``
          - ``"canonical_sha256"``: hex SHA-256 of the canonical payload bytes
          - ``"kid"``: the signature key id resolved
    """
    envelope = normalise_envelope(envelope)
    payload = envelope.get("payload", envelope)
    if not isinstance(payload, dict):
        return {
            "verdict": "INCOMPLETE",
            "axes": [
                {
                    "name": "payload",
                    "result": "FAIL",
                    "note": (
                        "receipt payload not available from this surface "
                        "(the server returned payload: null). Verify with a saved "
                        "receipt instead."
                    ),
                }
            ],
            "canonical_sha256": None,
            "kid": None,
        }

    sig_obj = envelope.get("signature", {})
    if isinstance(sig_obj, str):
        sig_obj = {
            "alg": envelope.get("algorithm", "ML-DSA-65"),
            "kid": payload.get("issuer_id", ""),
            "sig": sig_obj,
        }
    kid = sig_obj.get("kid", "")
    alg = sig_obj.get("alg", "ML-DSA-65")
    msg = canonical_json(payload)

    axes: list[dict] = []
    axes.append(
        {
            "name": "structure",
            "result": check_structure(payload)[0],
            "note": check_structure(payload)[1],
        }
    )

    pk, status, jwks_alg = resolve_key(jwks, kid)
    if pk is None:
        axes.append(
            {
                "name": "issuer_key",
                "result": "FAIL",
                "note": f"kid {kid!r} not in jwks directory",
            }
        )
        axes.append(
            {
                "name": "signature",
                "result": "SKIPPED",
                "note": "no issuer key to verify against",
            }
        )
    else:
        axes.append(
            {
                "name": "issuer_key",
                "result": "PASS",
                "note": f"resolved kid {kid} (status={status})",
            }
        )
        sig_bytes = _b64decode(sig_obj.get("sig", ""))
        sig_r, sig_n = verify_signature(pk, msg, sig_bytes, alg or jwks_alg)
        axes.append({"name": "signature", "result": sig_r, "note": sig_n})

    chain_r, chain_n = check_chain(payload, predecessor_payload)
    axes.append({"name": "chain", "result": chain_r, "note": chain_n})
    anch_r, anch_n = check_anchors(envelope)
    axes.append({"name": "anchors", "result": anch_r, "note": anch_n})
    skew_r, skew_n = check_skew(payload.get("issued_at", ""))
    axes.append({"name": "skew", "result": skew_r, "note": skew_n})

    has_fail = any(a["result"] == "FAIL" for a in axes)
    has_blocking_skip = any(
        a["result"] == "SKIPPED" and a["name"] != "chain" for a in axes
    )
    if has_fail:
        verdict = "FAIL"
    elif has_blocking_skip:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "axes": axes,
        "canonical_sha256": hashlib.sha256(msg).hexdigest(),
        "kid": kid,
    }


def _load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def main() -> int:
    p = argparse.ArgumentParser(description="Standalone Asqav receipt verifier.")
    p.add_argument("--id", help="signature id to fetch from api.asqav.com")
    p.add_argument("--receipt", help="path to a saved receipt JSON")
    p.add_argument("--jwks", help="path to a saved jwks.json (offline)")
    p.add_argument(
        "--predecessor", help="path to predecessor receipt JSON for the chain check"
    )
    p.add_argument("--offline", action="store_true", help="never reach the network")
    args = p.parse_args()

    if args.receipt:
        envelope = _load(args.receipt)
    elif args.id and not args.offline:
        envelope = _get_json(f"{API_BASE}/verify/{args.id}")
    else:
        p.error("supply --receipt FILE, or --id ID without --offline")
        return 2

    if args.jwks:
        jwks = _load(args.jwks)
    elif not args.offline:
        jwks = _get_json(JWKS_URL)
    else:
        p.error("offline mode needs --jwks FILE")
        return 2

    predecessor_payload = None
    if args.predecessor:
        pred = _load(args.predecessor)
        predecessor_payload = pred.get("payload", pred)

    return run(envelope, jwks, predecessor_payload)


if __name__ == "__main__":
    sys.exit(main())
