"""Canonicalization shared across format adapters.

Two callable shapes, one code path:
  - ``jcs(obj)``        : RFC 8785 JCS with NFC normalisation of every string,
                          for formats that mandate full RFC 8785 (AERF, ACTA, VC).
  - ``asqav_jcs(obj)``  : the Asqav cloud's historical JCS WITHOUT NFC, kept
                          byte-identical to ``verify_receipt.canonical_json`` so
                          the native adapter reproduces exactly what the cloud
                          signed.

GAP NOTE: the cloud's canonicaliser predates the NFC requirement and does not
normalise strings; ``asqav_jcs`` mirrors that on purpose. ``jcs`` is the spec
form. ASCII-only payloads (the common case) produce identical bytes either way.
"""
from __future__ import annotations

import json
import unicodedata
from typing import Any


def _nfc(obj: Any) -> Any:
    """Recursively NFC-normalise every string key and value."""
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    if isinstance(obj, dict):
        return {_nfc(k): _nfc(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nfc(v) for v in obj]
    return obj


def _dump(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def jcs(obj: Any) -> bytes:
    """RFC 8785 JCS bytes with NFC string normalisation."""
    return _dump(_nfc(obj))


def asqav_jcs(obj: Any) -> bytes:
    """Asqav cloud JCS bytes (no NFC); byte-identical to the cloud signer."""
    return _dump(obj)
