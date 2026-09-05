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

from asqav.verifier.verify_receipt import run_structured  # noqa: E402

#: The normative requirements of the profile that a receipt vector can exercise,
#: each tied to the section that states it. The verifier axis named here is the
#: one whose non-skipped result is evidence the vector reached the requirement.
REQUIREMENTS = {
    "REQ-SIGNATURE": {
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
    "REQ-ANCHOR": {
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


def _load(directory: pathlib.Path, name: str):
    path = directory / name
    return json.loads(path.read_text()) if path.exists() else None


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
        if fmt != "asqav-native":
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
            trusted_tsa_keys=trusted_tsa_keys,
            bitcoin_headers=bitcoin_headers,
        )
        results = {axis["name"]: axis["result"] for axis in result["axes"]}
        axis_results[directory.name] = results
        axis_notes[directory.name] = {a["name"]: a.get("note", "") for a in result["axes"]}
        exercised = sorted(
            req
            for req, meta in REQUIREMENTS.items()
            if results.get(meta["axis"]) in _EXERCISED
        )
        for req in exercised:
            covered[req].append(directory.name)
        by_vector[directory.name] = {
            "declared_outcome": expected.get("outcome"),
            "observed_verdict": result["verdict"],
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
            "not coverage, whatever a vector's notes claim."
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
    print(f"  asqav-native vectors : {len(by_vector)}")
    print(f"  interop fixtures     : {len(interop)}")
    print(f"  requirements covered : {len(REQUIREMENTS) - len(unmapped)}/{len(REQUIREMENTS)}")
    if unmapped:
        print(f"  UNMAPPED             : {', '.join(unmapped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
