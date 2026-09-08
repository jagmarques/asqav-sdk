#!/usr/bin/env python3
"""Validate reference data shape and relationships, without validating receipts."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DIRECTORY = Path(__file__).resolve().parent


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON members."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def load(path: Path) -> Any:
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=unique_object, parse_constant=reject_constant,
    )


def check_schema(document: Any) -> None:
    """Require schema validation before accessing any document members."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise ValueError("jsonschema is required; install it with python3 -m pip install jsonschema") from exc
    schema = load(DIRECTORY / "wire.schema.json")
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        raise ValueError("; ".join(f"{list(e.path)}: {e.message}" for e in errors))


def check_decisions(document: dict[str, Any]) -> None:
    decisions = set(document["decisions"])
    policies = set(document["policy_decisions"])
    mapping = dict(document["decision_map"])
    if len(mapping) != len(document["decision_map"]):
        raise ValueError("decision_map has duplicate source tokens")
    if not policies <= mapping.keys():
        raise ValueError("decision_map omits a policy decision")
    if not mapping.keys() <= policies | decisions:
        raise ValueError("decision_map has an unknown source token")
    if not set(mapping.values()) <= decisions:
        raise ValueError("decision_map has an unknown target decision")


def check_profiles(document: dict[str, Any]) -> None:
    if set(document["receipt_types"]) & set(document["request_extra_types"]):
        raise ValueError("request_extra_types overlaps receipt_types")
    expected = {
        "receipt_profile": ["receipt_types"],
        "receipt_request": ["receipt_types", "request_extra_types"],
        "sdk_incident": ["dora_incident_classes", "hipaa_incident_classes"],
    }
    for name, groups in document["profiles"].items():
        if groups != expected[name]:
            raise ValueError(f"profiles.{name} must reference {expected[name]}")
    if set(document["dora_incident_classes"]) & set(document["hipaa_incident_classes"]):
        raise ValueError("sdk_incident groups overlap")


def check_taxonomies(document: dict[str, Any]) -> None:
    if set(document["taxonomy_order"]) != document["fields"].keys():
        raise ValueError("taxonomy_order must reference every field exactly once")
    options = [field["typescript_option"] for field in document["fields"].values()]
    if len(set(options)) != len(options):
        raise ValueError("fields have duplicate typescript_option names")
    for name, field in document["fields"].items():
        if field["metadata_ref"] != name or name not in document["metadata"]:
            raise ValueError(f"fields.{name}.metadata_ref must resolve to its own metadata")


def check_metadata(document: dict[str, Any]) -> None:
    if document["metadata_order"] != list(document["metadata"]):
        raise ValueError("metadata_order must match every metadata key in order")


def validate(document: Any) -> None:
    check_schema(document)
    check_decisions(document)
    check_profiles(document)
    check_taxonomies(document)
    check_metadata(document)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DIRECTORY / "wire.json")
    args = parser.parse_args(argv)
    try:
        validate(load(args.path))
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("OK: wire vocabulary reference shape and relationships")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
