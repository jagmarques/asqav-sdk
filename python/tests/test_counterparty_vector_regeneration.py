"""Regeneration preserves authored negative recipes and signed fixture bytes."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

from asqav.counterparty import verify_counterparty_binding
from asqav.verifier.verify_receipt import check_counterparty_binding

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "verifier" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_fingerprint_regeneration_preserves_negative_scope_and_digest_recipes():
    regen = module("regen_fingerprint_vectors")
    original = json.loads((ROOT / "conformance/vectors.json").read_text())
    doc = copy.deepcopy(original)
    assert regen.regenerate(doc) == []
    assert doc == original
    vectors = {v["name"]: v for v in doc["vectors"]}
    checked = []
    for vector in vectors.values():
        expected = vector.get("expected", {})
        if "binding_result" not in expected:
            continue
        origin = vectors[expected["originating_envelope_ref"]]["input"]
        ack = {"payload": vector["input"], "signature": {"kid": vector["input"]["issuer_id"]}}
        result = verify_counterparty_binding(ack, origin)
        assert result.valid is expected["binding_result"]
        assert result.label == expected["binding_label"]
        state, note = check_counterparty_binding(vector["input"], origin)
        assert state == {True: "PASS", False: "FAIL", None: "SKIPPED"}[result.valid], note
        checked.append(vector["name"])
    assert set(checked) == {"counterparty_binding_scope_minus_anchors", "counterparty_binding_anchors_included_rejected",
                            "counterparty_binding_legacy_scope_absent", "counterparty_binding_unrecognised_scope"}


def test_signed_generator_is_deterministic_and_check_never_writes(tmp_path, monkeypatch):
    generator = module("generate_counterparty_vectors")
    files = generator.generated_files()
    assert files == generator.generated_files()
    for path, text in files.items():
        assert path.read_text() == text, path
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    monkeypatch.setattr(generator, "CORPUS", tmp_path / "verifier" / "conformance-vectors")
    monkeypatch.setattr(sys, "argv", ["generate_counterparty_vectors.py"])
    assert generator.main() == 0
    target = next(path for path in generator.generated_files() if path.name == "receipt.json")
    target.write_text("{\"drift\":true}\n")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(sys, "argv", ["generate_counterparty_vectors.py", "--check"])
    assert generator.main() == 1
    assert {p: p.read_bytes() for p in before} == before
    monkeypatch.setattr(sys, "argv", ["generate_counterparty_vectors.py"])
    assert generator.main() == 0
    monkeypatch.setattr(sys, "argv", ["generate_counterparty_vectors.py", "--check"])
    assert generator.main() == 0
