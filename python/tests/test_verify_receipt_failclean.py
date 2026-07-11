"""Report tool fails clean on missing / empty / non-object input.

Regression guard for the goal9 SDK example fix: verify_receipt must never leak
a urllib/json traceback on bad input. It exits nonzero with one readable line,
and reads a receipt from stdin via ``--receipt -``.
"""

from __future__ import annotations

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.verifier import verify_receipt as vr


def test_run_rejects_non_object_envelope() -> None:
    # A JSON array (non-object) must not crash run(); it returns 2 (INCOMPLETE).
    assert vr.run([], {"keys": []}, None) == 2


def test_run_structured_rejects_non_object() -> None:
    out = vr.run_structured("not-an-object", {"keys": []})
    assert out["verdict"] == "INCOMPLETE"
    assert out["canonical_sha256"] is None


def test_parse_object_rejects_array() -> None:
    with pytest.raises(vr.VerifierInputError):
        vr._parse_object("[]", "stdin")


def test_parse_object_rejects_empty() -> None:
    with pytest.raises(vr.VerifierInputError):
        vr._parse_object("   ", "stdin")


def test_parse_object_rejects_bad_json() -> None:
    with pytest.raises(vr.VerifierInputError):
        vr._parse_object("{not json", "stdin")


def test_parse_object_accepts_object() -> None:
    assert vr._parse_object('{"a": 1}', "stdin") == {"a": 1}


def test_load_dash_reads_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"a": 1}'))
    assert vr._load("-") == {"a": 1}


def test_load_dash_non_object_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("[]"))
    with pytest.raises(vr.VerifierInputError):
        vr._load("-")


def test_load_missing_file_raises_input_error() -> None:
    # A missing file surfaces as VerifierInputError, not a raw OSError traceback.
    with pytest.raises(vr.VerifierInputError):
        vr._load("/no/such/receipt/file.json")


def test_check_anchors_rejects_non_list_string() -> None:
    # A string "anchors" value must not crash check_anchors; it FAILs the axis.
    result, note = vr.check_anchors({"anchors": "str"})
    assert result == "FAIL"


def test_check_anchors_rejects_non_dict_entries() -> None:
    # A list of bare strings must not crash on a.get(); each entry FAILs the axis.
    result, note = vr.check_anchors({"anchors": ["x"]})
    assert result == "FAIL"


def test_check_anchors_rejects_non_dict_entries_int() -> None:
    result, note = vr.check_anchors({"anchors": [1]})
    assert result == "FAIL"


def test_run_structured_handles_malformed_anchors_string() -> None:
    # printf '{"anchors":"str"}' | verify_receipt --receipt - must not raise.
    envelope = {"payload": {"type": "x"}, "anchors": "str"}
    out = vr.run_structured(envelope, {"keys": []})
    assert out["verdict"] in ("FAIL", "INCOMPLETE")


def test_run_structured_handles_malformed_anchor_entries() -> None:
    envelope = {"payload": {"type": "x"}, "anchors": ["x"]}
    out = vr.run_structured(envelope, {"keys": []})
    assert out["verdict"] in ("FAIL", "INCOMPLETE")


# === criterion 446: complete the fail-clean surface ===

# S1: NaN / Infinity inside the receipt (canonical_json rejects them, allow_nan=False)


def _payload() -> dict:
    return {
        "type": "protectmcp:decision",
        "issued_at": "2026-05-04T09:14:22.000000Z",
        "issuer_id": "kid-x",
        "action_ref": "sha256:" + "8" * 64,
        "payload_digest": {"hash": "8" * 64, "size": 512},
        "policy_digest": "sha256:" + "3" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": "observation",
    }


def _sig() -> dict:
    return {"alg": "ML-DSA-65", "kid": "kid-x", "sig": "AAAA"}


def test_run_nan_in_payload_fails_clean(capsys) -> None:
    # A non-finite float in the payload must not raise ValueError out of run().
    env = {"payload": {**_payload(), "score": float("nan")}, "signature": _sig(), "anchors": []}
    code = vr.run(env, {"keys": []}, None)
    out = capsys.readouterr().out
    assert code == 2
    assert "non-finite" in out.lower()
    assert "PASS" not in out.replace("never a PASS", "")


def test_run_infinity_in_payload_fails_clean(capsys) -> None:
    env = {"payload": {**_payload(), "score": float("inf")}, "signature": _sig(), "anchors": []}
    code = vr.run(env, {"keys": []}, None)
    assert code == 2


def test_run_structured_nan_fails_clean() -> None:
    env = {"payload": {**_payload(), "score": float("nan")}, "signature": _sig(), "anchors": []}
    out = vr.run_structured(env, {"keys": []})
    assert out["verdict"] == "INCOMPLETE"


# S2: non-dict "signature" on a flat receipt (no payload wrapper) -> kid = sig_obj.get(...)


def test_run_non_dict_signature_fails_clean(capsys) -> None:
    env = {**_payload(), "signature": [1, 2, 3]}
    code = vr.run(env, {"keys": []}, None)
    assert code != 0  # never a PASS, never an AttributeError


def test_run_structured_non_dict_signature_fails_clean() -> None:
    env = {**_payload(), "signature": [1, 2, 3]}
    out = vr.run_structured(env, {"keys": []})
    assert out["verdict"] in ("FAIL", "INCOMPLETE")


# S3: malformed JWKS "keys" (string / list of non-dicts) -> resolve_key must not crash


def test_resolve_key_keys_is_string() -> None:
    assert vr.resolve_key({"keys": "abc"}, "kid-x") == (None, None, None)


def test_resolve_key_keys_is_list_of_ints() -> None:
    assert vr.resolve_key({"keys": [123]}, "kid-x") == (None, None, None)


def test_resolve_revoked_at_malformed_keys_no_crash() -> None:
    # The revoked_at helper shares the loop shape; guard it too so a direct call is safe.
    assert vr.resolve_revoked_at({"keys": "abc"}, "kid-x") is None
    assert vr.resolve_revoked_at({"keys": [123]}, "kid-x") is None


# S4: a JWK whose matching key lacks "public_key" -> resolution failure, not KeyError


def test_resolve_key_missing_public_key_no_crash() -> None:
    jwks = {"keys": [{"kid": "kid-x", "kty": "OKP", "crv": "Ed25519", "x": "abc"}]}
    assert vr.resolve_key(jwks, "kid-x") == (None, None, None)


# S5: non-string "sig" with a resolvable kid -> _b64decode must not hit .replace on a non-str


def _resolvable_jwks() -> dict:
    return {"keys": [{"kid": "kid-x", "public_key": "AAAA", "alg": "ML-DSA-65", "status": "active"}]}


def test_run_non_string_sig_fails_clean(capsys) -> None:
    env = {"payload": _payload(), "signature": {"alg": "ML-DSA-65", "kid": "kid-x", "sig": 123}, "anchors": []}
    code = vr.run(env, _resolvable_jwks(), None)
    out = capsys.readouterr().out
    assert code != 0
    assert "[  ok] issuer_key" in out  # the kid DID resolve; only the sig bytes are bad


def test_run_structured_non_string_sig_fails_clean() -> None:
    env = {"payload": _payload(), "signature": {"alg": "ML-DSA-65", "kid": "kid-x", "sig": 123}, "anchors": []}
    out = vr.run_structured(env, _resolvable_jwks())
    assert out["verdict"] in ("FAIL", "INCOMPLETE")


# S6: binary / non-UTF8 --receipt file -> VerifierInputError, not a raw UnicodeDecodeError


def test_load_binary_file_raises_input_error(tmp_path) -> None:
    p = tmp_path / "binary.receipt"
    p.write_bytes(b"\xff\xfe\x00\x01\x80\x81receipt")
    with pytest.raises(vr.VerifierInputError):
        vr._load(str(p))


# Falsy-anchors laundering: {} / "" / 0 are malformed, not "no anchors"


def test_check_anchors_empty_dict_is_malformed() -> None:
    result, _ = vr.check_anchors({"anchors": {}})
    assert result == "FAIL"


def test_check_anchors_empty_string_is_malformed() -> None:
    result, _ = vr.check_anchors({"anchors": ""})
    assert result == "FAIL"


def test_check_anchors_zero_is_malformed() -> None:
    result, _ = vr.check_anchors({"anchors": 0})
    assert result == "FAIL"


def test_check_anchors_absent_is_skipped() -> None:
    assert vr.check_anchors({})[0] == "SKIPPED"


def test_check_anchors_null_is_skipped() -> None:
    assert vr.check_anchors({"anchors": None})[0] == "SKIPPED"


def test_check_anchors_empty_list_is_skipped() -> None:
    assert vr.check_anchors({"anchors": []})[0] == "SKIPPED"


def test_normalise_envelope_preserves_falsy_anchors() -> None:
    # The reconstruct path must hand the malformed anchors value to check_anchors,
    # not launder it to [] before the guard runs.
    env = vr.normalise_envelope({"payload": _payload(), "anchors": {}})
    assert env["anchors"] == {}


def _write_receipt(tmp_path, anchors) -> tuple[str, str]:
    import json

    rp = tmp_path / "receipt.json"
    jp = tmp_path / "jwks.json"
    rp.write_text(json.dumps({"payload": _payload(), "signature": _sig(), "anchors": anchors}))
    jp.write_text('{"keys": []}')
    return str(rp), str(jp)


@pytest.mark.parametrize("anchors", [{}, "", 0])
def test_end_to_end_falsy_anchors_fail_malformed(tmp_path, monkeypatch, capsys, anchors) -> None:
    # The contract's end-to-end proof: a --receipt file with a falsy non-list
    # anchors value must reach check_anchors as malformed, through the whole tool.
    rp, jp = _write_receipt(tmp_path, anchors)
    monkeypatch.setattr(sys, "argv", ["verify_receipt.py", "--receipt", rp, "--jwks", jp, "--offline"])
    vr.main()
    out = capsys.readouterr().out
    assert "[FAIL] anchors" in out
    assert "not a list" in out


def test_end_to_end_reconstruct_path_falsy_anchors(tmp_path, monkeypatch, capsys) -> None:
    # No signature dict -> normalise_envelope reconstruct path -> proves the :222 fix.
    import json

    rp = tmp_path / "receipt.json"
    jp = tmp_path / "jwks.json"
    rp.write_text(json.dumps({"payload": _payload(), "anchors": {}}))
    jp.write_text('{"keys": []}')
    monkeypatch.setattr(
        sys, "argv", ["verify_receipt.py", "--receipt", str(rp), "--jwks", str(jp), "--offline"]
    )
    vr.main()
    out = capsys.readouterr().out
    assert "[FAIL] anchors" in out


# === regression: _contains_non_finite must not blow the recursion limit ===


def _deep_dict(depth: int) -> dict:
    d: dict = {"a": 1.0}
    for _ in range(depth):
        d = {"a": d}
    return d


def test_contains_non_finite_survives_deep_nesting() -> None:
    # A ~1500-deep payload used to raise RecursionError out of the (recursive)
    # guard itself; the iterative walk must return a plain bool instead.
    deep = _deep_dict(1500)
    assert vr._contains_non_finite(deep) is False


def test_contains_non_finite_survives_very_deep_nesting() -> None:
    assert vr._contains_non_finite(_deep_dict(4000)) is False


def test_scan_shape_flags_too_deep() -> None:
    # The depth-capped walk must reject nesting past MAX_NESTING_DEPTH before
    # any caller can hand the structure to the recursive json encoder.
    deep = _deep_dict(vr.MAX_NESTING_DEPTH + 50)
    assert vr._scan_shape(deep, max_depth=vr.MAX_NESTING_DEPTH) == "too_deep"


def test_scan_shape_ok_under_the_cap() -> None:
    shallow = _deep_dict(vr.MAX_NESTING_DEPTH - 50)
    assert vr._scan_shape(shallow, max_depth=vr.MAX_NESTING_DEPTH) is None


def test_run_deep_nesting_no_recursion_error(capsys) -> None:
    # Version-independent: rejected as malformed input before canonical_json
    # (json.dumps) ever sees it, so no dependence on the encoder's own limits.
    env = {"payload": {"a": _deep_dict(1500)}, "signature": _sig(), "anchors": []}
    code = vr.run(env, {"keys": []}, None)
    out = capsys.readouterr().out
    assert code == 2
    assert "nesting exceeds" in out


def test_run_structured_deep_nesting_no_recursion_error() -> None:
    env = {"payload": {"a": _deep_dict(1500)}, "signature": _sig(), "anchors": []}
    out = vr.run_structured(env, {"keys": []})
    assert out["verdict"] == "INCOMPLETE"
    assert "nesting exceeds" in out["axes"][0]["note"]


def test_run_deep_predecessor_no_recursion_error() -> None:
    # The predecessor argument is a separate object from envelope; it must be
    # depth-checked too, since check_chain canonicalises it independently.
    env = {"payload": _payload(), "signature": _sig(), "anchors": []}
    code = vr.run(env, {"keys": []}, {"a": _deep_dict(1500)})
    assert code == 2


def test_parse_object_recursion_error_becomes_input_error(monkeypatch) -> None:
    # Defense-in-depth: json.loads itself raising RecursionError (a pure-Python
    # decoder fallback) must surface as VerifierInputError, not crash the CLI.
    def _boom(text):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(vr.json, "loads", _boom)
    with pytest.raises(vr.VerifierInputError):
        vr._parse_object('{"a": 1}', "stdin")


def test_canonical_json_recursion_error_becomes_incomplete(monkeypatch, capsys) -> None:
    # Defense-in-depth: even if canonical_json itself raises RecursionError
    # (a bypassed or future call path), run() must still report cleanly.
    def _boom(*a, **kw):
        raise RecursionError("maximum recursion depth exceeded while encoding a JSON object")

    monkeypatch.setattr(vr.json, "dumps", _boom)
    env = {"payload": _payload(), "signature": _sig(), "anchors": []}
    code = vr.run(env, {"keys": []}, None)
    out = capsys.readouterr().out
    assert code == 2
    assert "nesting exceeds" in out


def test_describe_value_truncates_huge_string() -> None:
    # A multi-megabyte string payload must not blow up the error line itself.
    huge = "x" * (2 * 1024 * 1024)
    text = vr._describe_value(huge)
    assert len(text) <= 85
    assert "..." in text


# === advisory: the payload-not-an-object message names the actual type ===


def test_run_payload_int_names_actual_type(capsys) -> None:
    code = vr.run({"payload": 42, "signature": "x"}, {"keys": []}, None)
    out = capsys.readouterr().out
    assert code == 2
    assert "int" in out
    assert "42" in out


def test_run_payload_string_names_actual_type(capsys) -> None:
    code = vr.run({"payload": "hello", "signature": "x"}, {"keys": []}, None)
    out = capsys.readouterr().out
    assert code == 2
    assert "str" in out
    assert "hello" in out


def test_run_structured_payload_list_names_actual_type() -> None:
    out = vr.run_structured({"payload": [1, 2, 3], "signature": "x"}, {"keys": []})
    assert out["verdict"] == "INCOMPLETE"
    note = out["axes"][0]["note"]
    assert "list" in note
