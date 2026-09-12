"""Report tool fails clean on missing / empty / non-object input.

Regression guard for the goal9 SDK example fix: verify_receipt must never leak
a urllib/json traceback on bad input. It exits nonzero with one readable line,
and reads a receipt from stdin via ``--receipt -``.
"""

from __future__ import annotations

import base64
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.verifier import verify_receipt as vr


def test_run_rejects_non_object_envelope() -> None:
    # A JSON array (non-object) must not crash run(); unverified/unverifiable -> 2.
    assert vr.run([], {"keys": []}, None) == 2


def test_run_structured_rejects_non_object() -> None:
    out = vr.run_structured("not-an-object", {"keys": []})
    assert out["verdict"] == "unverified"
    assert out["failure_class"] == "unverifiable"
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
    assert out["verdict"] == "unverified"
    assert out["failure_class"] == "invalid"


def test_run_structured_handles_malformed_anchor_entries() -> None:
    envelope = {"payload": {"type": "x"}, "anchors": ["x"]}
    out = vr.run_structured(envelope, {"keys": []})
    assert out["verdict"] == "unverified"
    assert out["failure_class"] == "invalid"


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
    assert out["verdict"] == "unverified"
    assert out["failure_class"] == "unverifiable"


# S2: non-dict "signature" on a flat receipt (no payload wrapper) -> kid = sig_obj.get(...)


def test_run_non_dict_signature_fails_clean(capsys) -> None:
    env = {**_payload(), "signature": [1, 2, 3]}
    code = vr.run(env, {"keys": []}, None)
    assert code != 0  # never a PASS, never an AttributeError


def test_run_structured_non_dict_signature_fails_clean() -> None:
    env = {**_payload(), "signature": [1, 2, 3]}
    out = vr.run_structured(env, {"keys": []})
    assert out["verdict"] == "unverified"


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
    assert out["verdict"] == "unverified"


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
    assert out["verdict"] == "unverified"
    assert out["failure_class"] == "unverifiable"
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
    def _boom(text, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(vr.json, "loads", _boom)
    with pytest.raises(vr.VerifierInputError):
        vr._parse_object('{"a": 1}', "stdin")


def test_canonical_json_recursion_error_becomes_unverifiable(monkeypatch, capsys) -> None:
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
    assert out["verdict"] == "unverified"
    assert out["failure_class"] == "unverifiable"
    note = out["axes"][0]["note"]
    assert "list" in note


# === profile safe-integer precheck: current v=1 digest inputs refuse pre-crypto ===


def _profile_payload(**over) -> dict:
    base = {**_payload(), "v": 1, "previousReceiptHash": "1" * 64}
    base.update(over)
    return base


def _profile_envelope(payload: dict) -> dict:
    return {"payload": payload, "signature": _sig(), "anchors": []}


def _crypto_spies(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict[str, list] = {"verify_signature": [], "canonical_json": []}

    def spy_verify(pk, msg, sig, alg):
        calls["verify_signature"].append((pk, msg, sig, alg))
        raise AssertionError("crypto must not run after a profile refusal")

    def spy_canonical(obj, default=None):
        calls["canonical_json"].append(obj)
        raise AssertionError("canonical_json must not run after a profile refusal")

    monkeypatch.setattr(vr, "verify_signature", spy_verify)
    monkeypatch.setattr(vr, "canonical_json", spy_canonical)
    monkeypatch.setattr(
        vr, "evaluate_anchors", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("anchors must not evaluate after a profile refusal"))
    )
    return calls


def test_run_profile_excluded_number_refuses_before_crypto(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _crypto_spies(monkeypatch)
    env = _profile_envelope(_profile_payload(score=2**53))
    assert vr.run(env, {"keys": []}, None) == 2
    out = capsys.readouterr().out
    assert "profile range +/-(2**53 - 1)" in out
    assert "canonical bytes" not in out


def test_run_structured_profile_excluded_number_refuses_before_crypto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _crypto_spies(monkeypatch)
    env = _profile_envelope(_profile_payload(score=2**53))
    out = vr.run_structured(env, {"keys": []})
    assert out["verdict"] == "unverified"
    assert out["failure_class"] == "unverifiable"
    assert out["canonical_sha256"] is None
    assert len(out["axes"]) == 1
    assert out["axes"][0]["name"] == "input"
    assert "profile range +/-(2**53 - 1)" in out["axes"][0]["note"]


def test_run_structured_clean_v1_reaches_real_crypto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list = []
    orig = vr.verify_signature

    def spy(pk, msg, sig, alg):
        calls.append((pk, msg, sig, alg))
        return orig(pk, msg, sig, alg)

    monkeypatch.setattr(vr, "verify_signature", spy)
    env = _profile_envelope(_profile_payload(score=2**53 - 1))
    vr.run_structured(env, _resolvable_jwks())
    assert calls, "a clean v=1 receipt must reach the signature check"


@pytest.mark.parametrize("version", [None, 2, True, "1", [1]])
def test_run_structured_non_current_versions_skip_the_precheck(version) -> None:
    payload = _profile_payload(score=2**53)
    if version is None:
        del payload["v"]
    else:
        payload["v"] = version
    out = vr.run_structured(_profile_envelope(payload), {"keys": []})
    assert all("profile range" not in a["note"] for a in out["axes"])


def test_run_structured_float_1_0_selects_the_current_profile() -> None:
    payload = _profile_payload(score=2**53)
    payload["v"] = 1.0
    out = vr.run_structured(_profile_envelope(payload), {"keys": []})
    assert "profile range +/-(2**53 - 1)" in out["axes"][0]["note"]


def test_run_structured_outer_v_cannot_reclassify_nested_payload() -> None:
    nested = _profile_payload(score=2**53)
    del nested["v"]
    out = vr.run_structured(
        {"payload": nested, "signature": _sig(), "anchors": [], "v": 1},
        {"keys": []},
    )
    assert all("profile range" not in a["note"] for a in out["axes"])
    nested2 = _profile_payload(score=2**53)
    out2 = vr.run_structured(
        {"payload": nested2, "signature": _sig(), "anchors": [], "v": 2},
        {"keys": []},
    )
    assert "profile range +/-(2**53 - 1)" in out2["axes"][0]["note"]


def test_run_structured_missing_member_still_range_refuses() -> None:
    payload = _profile_payload(score=2**53)
    del payload["type"]
    out = vr.run_structured(_profile_envelope(payload), {"keys": []})
    assert "profile range +/-(2**53 - 1)" in out["axes"][0]["note"]


def test_run_structured_bad_predecessor_refuses_even_without_its_v() -> None:
    payload = _profile_payload(score=1)
    pred = {**_payload(), "previousReceiptHash": "2" * 64, "score": 2**53}
    out = vr.run_structured(_profile_envelope(payload), {"keys": []}, pred)
    assert "profile range +/-(2**53 - 1)" in out["axes"][0]["note"]


def test_run_structured_genesis_skips_predecessor_numbers() -> None:
    payload = _profile_payload(score=1)
    payload["previousReceiptHash"] = vr.FIRST_RECEIPT_SEED
    pred = {**_payload(), "previousReceiptHash": "2" * 64, "score": 2**53}
    out = vr.run_structured(_profile_envelope(payload), {"keys": []}, pred)
    assert all("profile range" not in a["note"] for a in out["axes"])


def test_run_structured_excluded_predecessor_without_member_refuses() -> None:
    # No registry on this path: any supplied predecessor is hashed for a
    # non-genesis link, so an excluded value refuses with no member needed.
    payload = _profile_payload(score=1)
    out = vr.run_structured(
        _profile_envelope(payload), {"keys": []}, {"n": 2**53}
    )
    assert "profile range +/-(2**53 - 1)" in out["axes"][0]["note"]
    assert out["canonical_sha256"] is None


def test_run_excluded_predecessor_without_member_refuses(capsys) -> None:
    payload = _profile_payload(score=1)
    code = vr.run(_profile_envelope(payload), {"keys": []}, {"n": 2**53})
    out = capsys.readouterr().out
    assert code == 2
    assert "profile range +/-(2**53 - 1)" in out


def test_run_structured_safe_predecessor_without_member_reaches_chain() -> None:
    payload = _profile_payload(score=1)
    out = vr.run_structured(
        _profile_envelope(payload), {"keys": []}, {"n": 2**53 - 1}
    )
    assert all("profile range" not in a["note"] for a in out["axes"])
    chain = next(a for a in out["axes"] if a["name"] == "chain")
    assert chain["result"] == "FAIL"
    assert out["canonical_sha256"] is not None


def test_run_safe_predecessor_without_member_reaches_chain(capsys) -> None:
    payload = _profile_payload(score=1)
    code = vr.run(_profile_envelope(payload), {"keys": []}, {"n": 2**53 - 1})
    out = capsys.readouterr().out
    assert code == 1
    assert "profile range" not in out


def test_run_genesis_skips_predecessor_numbers(capsys) -> None:
    payload = _profile_payload(score=1)
    payload["previousReceiptHash"] = vr.FIRST_RECEIPT_SEED
    code = vr.run(_profile_envelope(payload), {"keys": []}, {"n": 2**53})
    out = capsys.readouterr().out
    assert "profile range" not in out


def _binding(**over) -> dict:
    base = {
        "receipt_ref": "r",
        "envelope_hash": base64.b64encode(b"e" * 32).decode(),
    }
    base.update(over)
    return base


def test_run_structured_counterparty_commitment_is_range_checked() -> None:
    payload = _profile_payload(score=1, counterparty_binding=_binding())
    originator = {"payload": {"x": 2**53}, "signature": {"sig": "s"}}
    out = vr.run_structured(_profile_envelope(payload), {"keys": []}, None, counterparty=originator)
    assert "profile range +/-(2**53 - 1)" in out["axes"][0]["note"]


def test_run_counterparty_commitment_is_range_checked(capsys) -> None:
    payload = _profile_payload(score=1, counterparty_binding=_binding())
    originator = {"payload": {"x": 2**53}, "signature": {"sig": "s"}}
    code = vr.run(_profile_envelope(payload), {"keys": []}, None, counterparty=originator)
    out = capsys.readouterr().out
    assert code == 2
    assert "profile range +/-(2**53 - 1)" in out


def test_run_structured_cyclic_counterparty_refuses_finite() -> None:
    # Termination is the assertion: the pre-fix walk looped forever here.
    payload = _profile_payload(score=1, counterparty_binding=_binding())
    counterparty: dict = {"ok": 1}
    counterparty["again"] = counterparty
    out = vr.run_structured(
        _profile_envelope(payload), {"keys": []}, None, counterparty=counterparty
    )
    assert "profile range" in out["axes"][0]["note"]
    assert "cyclic" in out["axes"][0]["note"]
    assert out["canonical_sha256"] is None


def test_run_cyclic_counterparty_refuses_finite(capsys) -> None:
    payload = _profile_payload(score=1, counterparty_binding=_binding())
    counterparty: dict = {"ok": 1}
    counterparty["again"] = counterparty
    code = vr.run(
        _profile_envelope(payload), {"keys": []}, None, counterparty=counterparty
    )
    out = capsys.readouterr().out
    assert code == 2
    assert "cyclic" in out


@pytest.mark.parametrize(
    "binding",
    [
        {"envelope_hash": base64.b64encode(b"e" * 32).decode()},
        {"receipt_ref": "r"},
        {"receipt_ref": "r", "envelope_hash": "!!!"},
        {"receipt_ref": "r", "envelope_hash": base64.b64encode(b"e" * 16).decode()},
        {
            "receipt_ref": "r",
            "envelope_hash": base64.b64encode(b"e" * 32).decode(),
            "expect_ack_from": 7,
        },
    ],
)
def test_run_structured_malformed_binding_skips_counterparty(binding) -> None:
    payload = _profile_payload(score=1, counterparty_binding=binding)
    originator = {"payload": {"x": 2**53}, "signature": {"sig": "s"}}
    out = vr.run_structured(_profile_envelope(payload), {"keys": []}, None, counterparty=originator)
    assert all("profile range" not in a["note"] for a in out["axes"])


def test_run_structured_non_dict_counterparty_skips_commitment() -> None:
    payload = _profile_payload(score=1, counterparty_binding=_binding())
    out = vr.run_structured(_profile_envelope(payload), {"keys": []}, None, counterparty=["x"])
    assert all("profile range" not in a["note"] for a in out["axes"])


def test_run_structured_counterparty_without_binding_is_not_checked() -> None:
    payload = _profile_payload(score=1)
    originator = {"payload": {"x": 2**53}, "signature": {"sig": "s"}}
    out = vr.run_structured(_profile_envelope(payload), {"keys": []}, None, counterparty=originator)
    assert all("profile range" not in a["note"] for a in out["axes"])


def test_run_structured_signature_object_number_refuses_with_anchors() -> None:
    sig = {**_sig(), "n": 2**53}
    env = {
        "payload": _profile_payload(score=1),
        "signature": sig,
        "anchors": [{"type": "x"}],
    }
    out = vr.run_structured(env, {"keys": []})
    assert "profile range +/-(2**53 - 1)" in out["axes"][0]["note"]


@pytest.mark.parametrize("anchors", [[], None, {}, "x"])
def test_run_structured_signature_object_skipped_without_commitment(anchors) -> None:
    sig = {**_sig(), "n": 2**53}
    env = {"payload": _profile_payload(score=1), "signature": sig, "anchors": anchors}
    out = vr.run_structured(env, {"keys": []})
    assert all("profile range" not in a["note"] for a in out["axes"])


def test_run_structured_float_and_list_spellings_refuse() -> None:
    payload = _profile_payload(score=9007199254740992.0)
    out = vr.run_structured(_profile_envelope(payload), {"keys": []})
    assert "profile range +/-(2**53 - 1)" in out["axes"][0]["note"]
    payload2 = _profile_payload(scores=[1, 2**53])
    out2 = vr.run_structured(_profile_envelope(payload2), {"keys": []})
    assert "profile range +/-(2**53 - 1)" in out2["axes"][0]["note"]
    payload3 = _profile_payload(score=5.0, tags=["a"])
    out3 = vr.run_structured(_profile_envelope(payload3), {"keys": []})
    assert all("profile range" not in a["note"] for a in out3["axes"])


@pytest.mark.parametrize("drop", [["issuer_id"], ["previousReceiptHash"], ["issuer_id", "previousReceiptHash"]])
def test_run_structured_missing_members_still_range_refuse(drop) -> None:
    payload = _profile_payload(score=2**53)
    for key in drop:
        del payload[key]
    out = vr.run_structured(_profile_envelope(payload), {"keys": []})
    assert "profile range +/-(2**53 - 1)" in out["axes"][0]["note"]


def test_run_missing_members_still_range_refuse(capsys: pytest.CaptureFixture) -> None:
    payload = _profile_payload(score=2**53)
    del payload["issuer_id"]
    del payload["previousReceiptHash"]
    assert vr.run(_profile_envelope(payload), {"keys": []}, None) == 2
    assert "profile range +/-(2**53 - 1)" in capsys.readouterr().out


def test_run_structured_missing_member_clean_numbers_keep_structure_outcome() -> None:
    payload = _profile_payload(score=1)
    del payload["type"]
    out = vr.run_structured(_profile_envelope(payload), {"keys": []})
    assert all("profile range" not in a["note"] for a in out["axes"])
    assert out["axes"][0]["name"] == "structure"
    assert out["axes"][0]["result"] == "FAIL"
    assert "missing required fields" in out["axes"][0]["note"]


def test_run_string_signature_derives_object(capsys: pytest.CaptureFixture) -> None:
    env = {"payload": _payload(), "signature": "AAAA", "anchors": []}
    assert vr.run(env, {"keys": []}, None) == 2
    assert "canonical bytes" in capsys.readouterr().out


def test_run_structured_string_signature_derives_object() -> None:
    env = {"payload": _payload(), "signature": "AAAA", "anchors": []}
    out = vr.run_structured(env, {"keys": []})
    assert out["canonical_sha256"] is not None


def test_run_structured_null_anchors_return_early() -> None:
    env = {"payload": _payload(), "signature": _sig(), "anchors": None}
    out = vr.run_structured(env, {"keys": []})
    assert out["verdict"] == "unverified"
    assert out["axes"][0]["name"] == "structure"
    assert "null" in out["axes"][0]["note"]


def test_run_structured_canonical_recursion_refuses_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(obj):
        raise RecursionError("probe")

    monkeypatch.setattr(vr, "canonical_json", explode)
    out = vr.run_structured(_profile_envelope(_profile_payload(score=1)), {"keys": []})
    assert out["verdict"] == "unverified"
    assert out["axes"][0]["name"] == "input"
    assert "nesting" in out["axes"][0]["note"]


def _agent_jwks(sign: bool):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    issuer_sk = Ed25519PrivateKey.generate()
    agent_sk = Ed25519PrivateKey.generate()
    other_sk = Ed25519PrivateKey.generate()
    issuer_pk = issuer_sk.public_key().public_bytes_raw()
    agent_pk = agent_sk.public_key().public_bytes_raw()
    payload = _profile_payload(score=1, agent_id="agent-1", org_id="org-1")
    msg = vr.canonical_json(payload)
    signer = agent_sk if sign else other_sk
    sig = base64.b64encode(signer.sign(msg)).decode()
    jwks = {
        "keys": [
            {
                "kid": "kid-x",
                "issuer_id": "did:asqav:issuer",
                "public_key": base64.b64encode(issuer_pk).decode(),
                "alg": "Ed25519",
                "status": "active",
                "agent_id": "someone-else",
            },
            {
                "kid": "kid-a",
                "issuer_id": "did:asqav:issuer",
                "public_key": base64.b64encode(agent_pk).decode(),
                "alg": "Ed25519",
                "status": "active",
                "agent_id": "agent-1",
                "org_id": "org-1",
            },
        ]
    }
    env = {"payload": payload, "signature": {"alg": "Ed25519", "kid": "kid-x", "sig": sig}, "anchors": []}
    return env, jwks


def _ed25519_available() -> bool:
    try:
        import cryptography  # noqa: F401
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: F401

        return True
    except ImportError:
        return False


requires_ed25519 = pytest.mark.skipif(not _ed25519_available(), reason="cryptography not installed")


@requires_ed25519
def test_run_agent_key_adopts_on_verify() -> None:
    env, jwks = _agent_jwks(sign=True)
    out = vr.run_structured(env, jwks)
    key_axis = next(a for a in out["axes"] if a["name"] == "issuer_key")
    assert "kid-a" in key_axis["note"]
    sig_axis = next(a for a in out["axes"] if a["name"] == "signature")
    assert sig_axis["result"] == "PASS"


@requires_ed25519
def test_run_agent_key_exhausted_notes_without_verify() -> None:
    env, jwks = _agent_jwks(sign=False)
    out = vr.run_structured(env, jwks)
    key_axis = next(a for a in out["axes"] if a["name"] == "issuer_key")
    assert "no key published for this agent verified" in key_axis["note"]


@requires_ed25519
def test_run_agent_fallback_paths_through_text_api(capsys: pytest.CaptureFixture) -> None:
    env, jwks = _agent_jwks(sign=True)
    assert vr.run(env, jwks, None) in (1, 2)
    assert "kid-a" in capsys.readouterr().out
    env2, jwks2 = _agent_jwks(sign=False)
    assert vr.run(env2, jwks2, None) in (1, 2)
    assert "no key published for this agent verified" in capsys.readouterr().out


# === T-037 revision 6: a shared acyclic object is not a cycle ===


def _shared_leaf(kind: str):
    if kind == "dict":
        return {"n": 1}
    if kind == "list":
        return [1]
    return (1,)


def _shared_pair(kind: str) -> dict:
    leaf = _shared_leaf(kind)
    return {"a": leaf, "b": leaf}


def _twin_pair(kind: str) -> dict:
    return {"a": _shared_leaf(kind), "b": _shared_leaf(kind)}


def _shared_inputs(position: str, pair: dict):
    payload = _profile_payload(score=1)
    pred = cp = None
    if position == "payload":
        payload["ext"] = pair
        env = _profile_envelope(payload)
    elif position == "signature":
        env = {
            "payload": payload,
            "signature": {**_sig(), "ext": pair},
            "anchors": [{"type": "x"}],
        }
    elif position == "predecessor":
        env = _profile_envelope(payload)
        pred = pair
    else:
        payload["counterparty_binding"] = _binding()
        env = _profile_envelope(payload)
        cp = pair
    return env, pred, cp


_SHARED_POSITIONS = ["payload", "signature", "predecessor", "counterparty"]
_SHARED_KINDS = ["dict", "list", "tuple"]


@pytest.mark.parametrize("position", _SHARED_POSITIONS)
@pytest.mark.parametrize("kind", _SHARED_KINDS)
def test_run_shared_node_matches_unshared_twin(position, kind, capsys) -> None:
    env_s, pred_s, cp_s = _shared_inputs(position, _shared_pair(kind))
    code_s = vr.run(env_s, {"keys": []}, pred_s, counterparty=cp_s)
    text_s = capsys.readouterr().out
    env_t, pred_t, cp_t = _shared_inputs(position, _twin_pair(kind))
    code_t = vr.run(env_t, {"keys": []}, pred_t, counterparty=cp_t)
    text_t = capsys.readouterr().out
    assert (code_s, text_s) == (code_t, text_t)
    assert "cyclic reference" not in text_s


@pytest.mark.parametrize("position", _SHARED_POSITIONS)
@pytest.mark.parametrize("kind", _SHARED_KINDS)
def test_run_structured_shared_node_matches_unshared_twin(position, kind) -> None:
    env_s, pred_s, cp_s = _shared_inputs(position, _shared_pair(kind))
    out_s = vr.run_structured(env_s, {"keys": []}, pred_s, counterparty=cp_s)
    env_t, pred_t, cp_t = _shared_inputs(position, _twin_pair(kind))
    out_t = vr.run_structured(env_t, {"keys": []}, pred_t, counterparty=cp_t)
    assert out_s == out_t
    assert "cyclic reference" not in repr(out_s)


@pytest.mark.parametrize("position", _SHARED_POSITIONS)
def test_shared_out_of_range_matches_twin(position, capsys) -> None:
    bad = {"n": 2**53}
    env_s, pred_s, cp_s = _shared_inputs(position, {"a": bad, "b": bad})
    code_s = vr.run(env_s, {"keys": []}, pred_s, counterparty=cp_s)
    text_s = capsys.readouterr().out
    env_t, pred_t, cp_t = _shared_inputs(
        position, {"a": {"n": 2**53}, "b": {"n": 2**53}}
    )
    code_t = vr.run(env_t, {"keys": []}, pred_t, counterparty=cp_t)
    text_t = capsys.readouterr().out
    assert (code_s, text_s) == (code_t, text_t)
    assert code_s == 2
    assert "excludes integer 9007199254740992" in text_s
    out_s = vr.run_structured(env_s, {"keys": []}, pred_s, counterparty=cp_s)
    out_t = vr.run_structured(env_t, {"keys": []}, pred_t, counterparty=cp_t)
    assert out_s == out_t
    assert "excludes integer 9007199254740992" in out_s["axes"][0]["note"]


def test_profile_range_note_self_dict_cycle_refuses() -> None:
    cyc: dict = {}
    cyc["self"] = cyc
    note = vr.profile_range_note(cyc)
    assert note == (
        "Asqav profile range +/-(2**53 - 1) unverifiable: "
        "cyclic reference at $.self"
    )


def test_profile_range_note_self_list_cycle_refuses() -> None:
    cyc: list = []
    cyc.append(cyc)
    note = vr.profile_range_note(cyc)
    assert note == (
        "Asqav profile range +/-(2**53 - 1) unverifiable: "
        "cyclic reference at $[0]"
    )


def test_profile_range_note_mutual_cycle_refuses() -> None:
    first: dict = {}
    second = {"first": first}
    first["second"] = second
    note = vr.profile_range_note(first)
    assert note is not None
    assert "unverifiable: cyclic reference at " in note


def test_profile_range_note_doubling_dag_returns_none() -> None:
    node: dict = {"x": 1}
    for _ in range(30):
        node = {"l": node, "r": node}
    # Bind the note first: repr of the DAG in a failed assert is exponential.
    note = vr.profile_range_note(node)
    assert note is None


class _CountingDict(dict):
    expansions = 0

    def items(self):
        type(self).expansions += 1
        return super().items()


def test_profile_range_note_expands_each_container_once() -> None:
    _CountingDict.expansions = 0
    node: dict = {"x": 1}
    for _ in range(30):
        node = _CountingDict((("l", node), ("r", node)))
    # Bind the note first: repr of the DAG in a failed assert is exponential,
    # and a dict subclass defeats reprlib truncation, hanging pytest itself.
    note = vr.profile_range_note(node)
    assert note is None
    assert _CountingDict.expansions == 30


# === T-096 revision 7: refusal-path hardening (D1 digit cap, D2 keep-alive) ===


def _decimal_int(digits: int) -> int:
    return 10 ** (digits - 1)


def test_profile_range_note_4300_digit_full_value() -> None:
    n = _decimal_int(4300)
    note = vr.profile_range_note({"n": n})
    assert note == (
        "Asqav profile range +/-(2**53 - 1) excludes integer " f"{n} at $.n"
    )


def test_profile_range_note_4301_digit_count_message() -> None:
    note = vr.profile_range_note({"n": _decimal_int(4301)})
    assert note == (
        "Asqav profile range +/-(2**53 - 1) excludes a 4301-digit integer at $.n"
    )


def test_profile_range_note_4400_digit_count_message() -> None:
    note = vr.profile_range_note({"n": _decimal_int(4400)})
    assert note == (
        "Asqav profile range +/-(2**53 - 1) excludes a 4400-digit integer at $.n"
    )
    neg = vr.profile_range_note({"n": -_decimal_int(4400)})
    assert neg == (
        "Asqav profile range +/-(2**53 - 1) excludes a 4400-digit integer at $.n"
    )


def test_run_huge_integer_refuses_unverifiable(capsys: pytest.CaptureFixture) -> None:
    payload = _profile_payload(score=1)
    payload["ext"] = {"n": _decimal_int(4400)}
    code = vr.run(_profile_envelope(payload), {"keys": []}, None)
    captured = capsys.readouterr()
    assert code == 2
    assert "failure_class: unverifiable" in captured.out
    assert "excludes a 4400-digit integer at $.ext.n" in captured.out
    assert "Traceback" not in captured.out + captured.err
    assert "Exceeds the limit" not in captured.out + captured.err


def test_run_structured_huge_integer_refuses_unverifiable() -> None:
    payload = _profile_payload(score=1)
    payload["ext"] = {"n": _decimal_int(4400)}
    out = vr.run_structured(_profile_envelope(payload), {"keys": []}, None)
    assert out["verdict"] == "unverified"
    assert out["failure_class"] == "unverifiable"
    assert out["axes"][0]["note"] == (
        "Asqav profile range +/-(2**53 - 1) excludes a 4400-digit integer at $.ext.n"
    )


def test_reject_unsafe_integer_huge_literal() -> None:
    with pytest.raises(vr.UnsafeIntegerError, match="4400-digit integer"):
        vr._reject_unsafe_integer("9" * 4400)
    with pytest.raises(vr.UnsafeIntegerError, match="4400-digit integer"):
        vr._reject_unsafe_integer("-" + "9" * 4400)


def test_reject_unsafe_integer_zero_padded_huge_literal() -> None:
    assert vr._reject_unsafe_integer("0" * 4400 + "1") == 1
    assert vr._reject_unsafe_integer("-" + "0" * 4400) == 0


def test_end_to_end_huge_literal_exits_unverifiable(tmp_path, monkeypatch, capsys) -> None:
    import json

    payload = _profile_payload(score=1)
    text = json.dumps(
        {"payload": payload, "signature": _sig(), "anchors": [], "big": "@BIG@"}
    ).replace('"@BIG@"', "9" * 4400)
    rp = tmp_path / "receipt.json"
    jp = tmp_path / "jwks.json"
    rp.write_text(text)
    jp.write_text('{"keys": []}')
    monkeypatch.setattr(
        sys, "argv", ["verify_receipt.py", "--receipt", str(rp), "--jwks", str(jp), "--offline"]
    )
    assert vr.main() == 2
    err = capsys.readouterr().err
    assert "4400-digit integer" in err
    assert "Traceback" not in err


class _FabricatedChild(dict):
    def __init__(self, n: int) -> None:
        super().__init__()
        self._n = n

    def items(self):  # fresh child every call; nothing stored
        return [("v", {"n": self._n})]


def test_profile_range_note_fabricated_child_detected() -> None:
    for _ in range(5):
        note = vr.profile_range_note(
            {"bad": _FabricatedChild(2**53), "clean": _FabricatedChild(1)}
        )
        assert note is not None
        assert "excludes integer 9007199254740992" in note
