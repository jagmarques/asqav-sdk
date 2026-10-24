#!/usr/bin/env python3
"""Derive the vector-to-requirement mapping, and the list of requirements no vector reaches.

Publishing a mapping is only worth something if the unmapped half is published
with it. A corpus can be grown until every vector maps to something while the
requirements nobody wrote a vector for stay invisible, so this writes both halves
from the same run and refuses to let the second one be omitted.

The mapping is derived rather than declared: each asqav-native vector is run
through the offline verifier and the axes it actually exercises are read off the
result. An axis the verifier SKIPPED did not exercise its requirement, whatever
the vector's notes claim. Vectors in the other format families are recorded as
interoperability fixtures against their own specifications; they are not evidence
about this profile's normative requirements and are not counted as covering any.

Run: python3 verifier/build_requirement_map.py
Writes: verifier/conformance-vectors/requirement-map.json
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
VECTORS = ROOT / "conformance-vectors"
OUT = VECTORS / "requirement-map.json"
sys.path.insert(0, str(ROOT.parent / "python" / "src"))

from asqav.verifier.oracle.runner import (  # noqa: E402
    ANCHOR_REQUIREMENT,
    AXIS_SCOPE_EXERCISED,
    AXIS_SCOPE_KEY,
    AXIS_SCOPE_NOT_APPLICABLE,
    anchors_axis,
    load_inputs,
    verify_inputs,
)
from asqav.verifier.verify_receipt import (  # noqa: E402
    _is_flat_hash_receipt,
    run_structured,
)

#: The requirement a verified vector never leaves unexercised.
SIGNATURE_REQUIREMENT = "REQ-SIGNATURE"

#: The normative requirements of the profile that a receipt vector can exercise,
#: each tied to the section that states it. The verifier axis named here is the
#: one whose non-skipped result is evidence the vector reached the requirement.
REQUIREMENTS = {
    SIGNATURE_REQUIREMENT: {
        "section": "mandatory-checks",
        "statement": "Verify the signature over the canonical signed bytes under the declared algorithm.",
        "axis": "signature",
    },
    "REQ-KEY-RESOLUTION": {
        "section": "mandatory-checks",
        "statement": "Resolve the verification key by kid to exactly one key, never from the envelope.",
        "axis": "issuer_key",
    },
    "REQ-ISSUER-BIND": {
        "section": "issuer-id",
        "statement": "The resolved key is published under the issuer_id the receipt claims.",
        "axis": "issuer_bind",
    },
    "REQ-KEY-STATUS": {
        "section": "mandatory-checks",
        "statement": "A receipt signed by a revoked key is not reported verified on key status alone.",
        "axis": "key_status",
    },
    "REQ-KEY-THUMBPRINT": {
        "section": "key-thumbprint",
        "statement": "Recompute the RFC 7638 thumbprint of the resolved key and require equality.",
        "axis": "key_binding",
    },
    "REQ-CHAIN": {
        "section": "hash-chain",
        "statement": "Re-walk the SHA-256 link to the predecessor receipt.",
        "axis": "chain",
    },
    ANCHOR_REQUIREMENT: {
        "section": "anchoring",
        "statement": "Cryptographically re-verify each anchor against the committed bytes; presence never suffices.",
        "axis": "anchors",
    },
    "REQ-SKEW": {
        "section": "issued-at",
        "statement": "Enforce the future-skew bound on issued_at; past skew is not a failure.",
        "axis": "skew",
    },
    "REQ-EXPIRY": {
        "section": "result-bound",
        "statement": "Report expires_at on its own axis without folding the receipt-level verdict.",
        "axis": "expiry",
    },
    "REQ-NONCE": {
        "section": "result-bound",
        "statement": "Flag a duplicate nonce under one issuer_id as a replay candidate.",
        "axis": "nonce",
    },
    "REQ-PAYLOAD-DIGEST": {
        "section": "mandatory-checks",
        "statement": "Where context and payload_digest are both carried, recompute the digest and the size.",
        "axis": "payload_digest",
    },
    "REQ-KEYED-DIGEST": {
        "section": "payload-digest",
        "statement": "Use the defined keyed construction when hash_algo selects it; a keyed digest without holder key material is unverifiable, never a mismatch.",
        "axis": "payload_digest",
    },
    "REQ-OUTER-COPIES": {
        "section": "audit-pack-export",
        "statement": "An exported entry's plain copies of signed values must equal the signed payload; a copy that disagrees fails the verdict, because no signature covers it.",
        "axis": "outer_copies",
    },
    "REQ-COUNTERPARTY": {
        "section": "counterparty-binding",
        "statement": "Weigh a counterparty_binding rather than letting the corroboration claim ride unchecked.",
        "axis": "counterparty",
    },
    "REQ-STRUCTURE": {
        "section": "field-profile",
        "statement": "Required members are present and well-formed, and the type is a registered namespace.",
        "axis": "structure",
    },
}

#: An axis result that is evidence the vector reached the requirement. SKIPPED
#: means the check did not run, so it is not coverage.
_EXERCISED = {"PASS", "FAIL"}

#: The format a vector maps against the profile; every other is interop.
NATIVE_FORMAT = "asqav-native"

#: axis_scope (AXIS_SCOPE_KEY, from the runner) is written from this map's
#: exercises and not_exercised, never typed by hand.

#: Names observed_verdict's scope: only the axes in axis_scope's exercised
#: list, an axis-scoped result, never a full-profile verdict.
OBSERVED_VERDICT_SCOPE_KEY = "observed_verdict_scope"
OBSERVED_VERDICT_SCOPE = "axis_scope.exercised"

#: Anchor material from outside the vector. Vector files ship beside the
#: receipt, so they are corpus-supplied and listed apart, never trusted.
TRUSTED_ANCHOR_MATERIAL_KEY = "trusted_anchor_material_supplied"
CORPUS_ANCHOR_MATERIAL_KEY = "corpus_anchor_material"

#: Which verifier a vector's axes were read from. run_structured refuses a
#: flat hash-mode receipt, so its axes come from the oracle that verifies it.
DERIVED_BY_KEY = "derived_by"
DERIVED_BY_VERIFIER = "verify_receipt.run_structured"
DERIVED_BY_ORACLE = "oracle"


def _load(directory: pathlib.Path, name: str):
    path = directory / name
    return json.loads(path.read_text()) if path.exists() else None


def _oracle_axes(directory: pathlib.Path) -> tuple:
    """The oracle's verdict and (axis, result, note) list, anchors included."""
    inputs = load_inputs(directory, NATIVE_FORMAT)
    oracle = verify_inputs(*inputs)
    axes = [*oracle.axes, anchors_axis(inputs[0])]
    return oracle.verdict, [(a.axis, a.result, a.note) for a in axes]


def write_axis_scopes(vectors: dict) -> int:
    """Write each native expected.json's axis_scope; return how many moved."""
    moved = 0
    for name, entry in vectors.items():
        path = VECTORS / name / "expected.json"
        expected = json.loads(path.read_text())
        expected[AXIS_SCOPE_KEY] = {
            AXIS_SCOPE_EXERCISED: entry["exercises"],
            AXIS_SCOPE_NOT_APPLICABLE: entry["not_exercised"],
        }
        text = json.dumps(expected, indent=2) + "\n"
        if path.read_text() != text:
            path.write_text(text)
            moved += 1
    return moved


def main() -> int:
    by_vector = {}
    interop = {}
    covered: dict[str, list[str]] = {req: [] for req in REQUIREMENTS}
    axis_results: dict[str, dict] = {}
    axis_notes: dict[str, dict] = {}

    for directory in sorted(p for p in VECTORS.iterdir() if p.is_dir()):
        expected = _load(directory, "expected.json")
        if expected is None:
            continue
        fmt = expected.get("format", "unknown")
        if fmt != NATIVE_FORMAT:
            interop[directory.name] = {
                "format": fmt,
                "exercises_profile_requirements": False,
                "note": (
                    "interoperability fixture for another specification; it is not "
                    "evidence about this profile's normative requirements"
                ),
            }
            continue

        receipt = _load(directory, "receipt.json")
        jwks = _load(directory, "jwks.json")
        if receipt is None or jwks is None:
            by_vector[directory.name] = {
                "exercises": [],
                "not_exercised": sorted(REQUIREMENTS),
                "note": "no receipt/jwks pair to run; nothing derived",
            }
            continue

        predecessor = _load(directory, "predecessor.json")
        # run_structured hashes what it is given as the predecessor PAYLOAD, but the
        # corpus ships full envelopes; unwrap to the payload member. A bare payload
        # passes through unchanged.
        if isinstance(predecessor, dict) and isinstance(predecessor.get("payload"), dict):
            predecessor = predecessor["payload"]
        # Per-vector anchor material, when the vector ships it: public trust
        # material (pinned TSA certificates, bitcoin block headers) that lets the
        # anchors axis complete offline. A vector without them keeps it SKIPPED.
        tsa_trust = directory / "tsa_trust.pem"
        trusted_tsa_keys = [tsa_trust.read_bytes()] if tsa_trust.exists() else None
        bitcoin_headers = _load(directory, "bitcoin_headers.json")
        result = run_structured(
            receipt,
            jwks,
            predecessor_payload=predecessor,
            counterparty=_load(directory, "originating_envelope.json"),
            trusted_tsa_keys=trusted_tsa_keys,
            bitcoin_headers=bitcoin_headers,
        )
        derived_by = DERIVED_BY_VERIFIER
        axes = [(a["name"], a["result"], a.get("note", "")) for a in result["axes"]]
        verdict = result["verdict"]
        if _is_flat_hash_receipt(receipt):
            derived_by, verdict, axes = DERIVED_BY_ORACLE, *_oracle_axes(directory)
            # the oracle run reads no corpus anchor material
            trusted_tsa_keys = bitcoin_headers = None
        results = {name: outcome for name, outcome, _note in axes}
        axis_results[directory.name] = results
        axis_notes[directory.name] = {name: note for name, _outcome, note in axes}
        exercised = sorted(
            req
            for req, meta in REQUIREMENTS.items()
            if results.get(meta["axis"]) in _EXERCISED
        )
        for req in exercised:
            covered[req].append(directory.name)
        corpus_material = sorted(
            name
            for name, given in (
                (tsa_trust.name, trusted_tsa_keys),
                ("bitcoin_headers.json", bitcoin_headers),
            )
            if given is not None
        )
        by_vector[directory.name] = {
            "declared_outcome": expected.get("outcome"),
            "observed_verdict": verdict,
            DERIVED_BY_KEY: derived_by,
            OBSERVED_VERDICT_SCOPE_KEY: OBSERVED_VERDICT_SCOPE,
            TRUSTED_ANCHOR_MATERIAL_KEY: False,
            CORPUS_ANCHOR_MATERIAL_KEY: corpus_material,
            "exercises": exercised,
            "not_exercised": sorted(
                req
                for req, meta in REQUIREMENTS.items()
                if results.get(meta["axis"]) not in _EXERCISED
            ),
        }

    unmapped = sorted(req for req, hits in covered.items() if not hits)
    # Say WHY each gap exists, read off the axis notes rather than guessed at, so
    # the list is actionable instead of merely honest.
    unmapped_detail = {}
    for req in unmapped:
        axis = REQUIREMENTS[req]["axis"]
        seen = sorted({observed.get(axis, "(absent)") for observed in axis_results.values()})
        notes = sorted(
            {
                note
                for observed_notes in axis_notes.values()
                for name, note in observed_notes.items()
                if name == axis and note
            }
        )
        unmapped_detail[req] = {
            "axis": axis,
            "results_seen_across_vectors": seen,
            "sample_notes": notes[:3],
        }
    document = {
        "generated_by": "verifier/build_requirement_map.py",
        "how": (
            "Each asqav-native vector is run through the offline verifier and the "
            "axes it actually exercises are read off the result. A SKIPPED axis is "
            "not coverage, whatever a vector's notes claim. run_structured refuses "
            "a flat hash-mode receipt, so such a vector's axes and verdict are read "
            "off the oracle instead, and derived_by names which one ran."
        ),
        "observed_verdict_note": (
            "observed_verdict is an axis-scoped result: it covers only the axes in "
            "the vector's axis_scope.exercised, not a full-profile verdict. The "
            "anchor material this run reads ships in the vector beside the receipt "
            "(corpus_anchor_material); none is trusted material supplied apart from "
            "the corpus, so trusted_anchor_material_supplied is false and an anchors "
            "PASS shows the check ran, not that a trusted root confirmed it."
        ),
        # The per-vector axis evidence the coverage lists are derived from, published
        # so a reader can check the derivation rather than trust it. Results only:
        # the notes embed wall-clock seconds (skew, expiry), which never reproduce.
        "axis_results": axis_results,
        "requirements": REQUIREMENTS,
        "vectors": by_vector,
        "interop_fixtures": interop,
        "coverage": {req: sorted(hits) for req, hits in covered.items()},
        "unmapped_requirements": unmapped,
        "unmapped_detail": unmapped_detail,
        "unmapped_note": (
            "These are normative requirements no vector in this corpus exercises. "
            "They are published because an unmapped requirement stays unmapped "
            "however many vectors exist, so growing the corpus does not remove one "
            "from this list; writing a vector that reaches it does."
        ),
        "counts": {
            "asqav_native_vectors": len(by_vector),
            "interop_fixtures": len(interop),
            "requirements": len(REQUIREMENTS),
            "requirements_covered": len(REQUIREMENTS) - len(unmapped),
            "requirements_unmapped": len(unmapped),
        },
    }
    OUT.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT.parent)}")
    print(f"  axis_scope rewritten : {write_axis_scopes(by_vector)} expected.json")
    print(f"  asqav-native vectors : {len(by_vector)}")
    print(f"  interop fixtures     : {len(interop)}")
    print(f"  requirements covered : {len(REQUIREMENTS) - len(unmapped)}/{len(REQUIREMENTS)}")
    if unmapped:
        print(f"  UNMAPPED             : {', '.join(unmapped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
