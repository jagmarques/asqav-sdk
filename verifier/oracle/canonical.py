"""Canonicalization shared across format adapters.

Three callable shapes, because three signers disagree on what "JCS" means:

  - ``jcs(obj)``         : NFC-normalised, Python ``sort_keys`` (Unicode code
                           point) ordering, numbers preserved as Python emits
                           them. This matches the AERF and ACTA reference
                           producers, whose Go canonicaliser writes numbers
                           verbatim and sorts keys byte-wise. For their ASCII
                           field set the code-point order equals the byte order,
                           so this reproduces exactly the bytes those producers
                           signed.
  - ``jcs_rfc8785(obj)`` : strict RFC 8785 - UTF-16 code-unit key ordering and
                           ECMAScript ``Number.prototype.toString`` number output,
                           plus NFC. This is what the agent-receipts producers
                           sign, and it byte-matches the agent-receipts upstream
                           canonicalization vectors. The two diverge from ``jcs``
                           on supplementary-plane keys and on non-integral or
                           large-magnitude numbers.
  - ``asqav_jcs(obj)``   : the Asqav cloud's historical JCS WITHOUT NFC, kept
                           byte-identical to ``verify_receipt.canonical_json`` so
                           the native adapter reproduces exactly what the cloud
                           signed.

INTEROP NOTE: AERF SPEC.md §5.1 and the agent-receipts spec both cite "full RFC
8785 (JCS)", but the AERF Go reference (``verifiers/go/internal/aerf/canonical.go``)
preserves numbers via ``json.Number`` verbatim and sorts with ``sort.Strings``
(UTF-8 byte order), so an AERF receipt carrying ``1250.0`` is signed over the
bytes ``1250.0``, not the RFC 8785 ``1250``. ``jcs`` therefore deliberately keeps
the AERF dialect; only the agent-receipts adapter uses ``jcs_rfc8785``.
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


def jcs(obj: Any) -> bytes:
    """AERF/ACTA-dialect JCS: NFC, code-point key sort, numbers as Python emits them."""
    return json.dumps(
        _nfc(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def asqav_jcs(obj: Any) -> bytes:
    """Asqav cloud JCS bytes (no NFC); byte-identical to the cloud signer."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def jcs_rfc8785(obj: Any) -> bytes:
    """Strict RFC 8785 JCS bytes (UTF-16 key sort, ECMAScript numbers) with NFC."""
    return _serialize_8785(_nfc(obj)).encode("utf-8")


def _utf16_key(key: str) -> tuple[int, ...]:
    """UTF-16 BE code-unit sequence for ``key`` - the RFC 8785 key-ordering key."""
    raw = key.encode("utf-16-be")
    return tuple(int.from_bytes(raw[i : i + 2], "big") for i in range(0, len(raw), 2))


def _number_8785(value: float) -> str:
    """ECMAScript ``Number.prototype.toString`` form of a JSON number (RFC 8785)."""
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite number not allowed in JCS: {value!r}")
    if isinstance(value, int):
        return str(value)
    if value == int(value) and abs(value) < 1e21:
        return str(int(value))  # ECMAScript renders -0.0 and 5.0 as 0 and 5
    return _ecma_float(float(value))


def _ecma_float(value: float) -> str:
    """Non-integral float as ECMA-262 6.1.6.1.20 Number-to-String (shortest round-trip)."""
    sign = "-" if value < 0 else ""
    s, exp = _shortest_digits(abs(value))
    k = len(s)
    n = exp + k  # decimal point sits after digit n; ECMAScript chooses fixed vs exponential by n
    if k <= n <= 21:
        return f"{sign}{s}{'0' * (n - k)}"
    if 0 < n <= 21:
        return f"{sign}{s[:n]}.{s[n:]}"
    if -6 < n <= 0:
        return f"{sign}0.{'0' * -n}{s}"
    tail = s[0] if k == 1 else f"{s[0]}.{s[1:]}"
    return f"{sign}{tail}e{'+' if n - 1 >= 0 else '-'}{abs(n - 1)}"


def _shortest_digits(value: float) -> tuple[str, int]:
    """Return (digit string with no trailing zeros, base-10 exponent) for a positive float."""
    mantissa, _, exp = repr(value).partition("e")
    exponent = int(exp) if exp else 0
    if "." in mantissa:
        whole, frac = mantissa.split(".")
        exponent -= len(frac)
        mantissa = whole + frac
    digits = mantissa.lstrip("0") or "0"
    stripped = digits.rstrip("0")
    if stripped:
        exponent += len(digits) - len(stripped)
        digits = stripped
    return digits, exponent


def _serialize_8785(obj: Any) -> str:
    """RFC 8785 serialisation: UTF-16 sorted keys, ECMAScript numbers, minimal strings."""
    if obj is None or isinstance(obj, bool):
        return json.dumps(obj)
    if isinstance(obj, (int, float)):
        return _number_8785(obj)
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, list):
        return "[" + ",".join(_serialize_8785(v) for v in obj) + "]"
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: _utf16_key(kv[0]))
        body = ",".join(json.dumps(k, ensure_ascii=False) + ":" + _serialize_8785(v) for k, v in items)
        return "{" + body + "}"
    raise TypeError(f"unserialisable type in JCS input: {type(obj).__name__}")
