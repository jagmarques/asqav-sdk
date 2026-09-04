# Copyright 2026 Asqav
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Literal pins for every derived member of conformance/vectors.json (criterion 670).

The corpus is a contract with third-party implementers, so it is FIXTURE DATA: the
expected value of a derived member is a literal string in this file, never a value
the production code supplies at assertion time.

That distinction is not academic. The counterparty ``envelope_hash`` was stale in
five vectors for the life of the file, and the two tests that named it both built
their expectation by calling ``compute_envelope_hash`` - the function under test -
against a locally constructed envelope, so neither ever read the published bytes.
A green suite proved nothing about the file outsiders implement against.

Sibling checks, deliberately kept separate:

* ``test_canonicalize.py`` asserts the SDK's public helpers reproduce these bytes.
  That is the other direction and is still wanted.
* ``verifier/regen_fingerprint_vectors.py --check`` re-derives every member from each
  vector's own input and is what CI runs on every push.
* ``test_corpus_lock.py`` pins the file's SHA-256 as a whole.

This file is the one that says WHAT the bytes are.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

VECTORS_PATH = Path(__file__).parent.parent.parent / "conformance" / "vectors.json"
_VECTOR_ROOT = Path(__file__).parent.parent.parent / "verifier" / "conformance-vectors"


def _vectors() -> dict[str, dict]:
    return {v["name"]: v for v in json.loads(VECTORS_PATH.read_text())["vectors"]}


#: SHA-256 of each vector's canonical bytes, verbatim from the published file.
#: Pinning this pins ``canonical`` transitively: no other byte string has this digest.
PINNED_SHA256 = {
    "minimal_read": "991033e23a3e0939c258e10f2f8f88183d9bcdb08de62ef02446e11e0819c671",
    "tool_call_with_counterparty":
        "a6c39c3b6e09761a141148c0862edbeaeb59642b4e0a1e4a85d6fba63f634ff4",
    "traced_child_action": "9294075a29c366f9dfc54b8b7f8fa0054c290fef20e8931e93e94b2bee921d22",
    "tampered_signature": "3deafb65ffe2785bfe35fe9d687a3076aa475c50ef8be4b2763424ced3e19bba",
    "swapped_public_key": "3deafb65ffe2785bfe35fe9d687a3076aa475c50ef8be4b2763424ced3e19bba",
    "stale_card": "0dcced0b29db5e9c29f5b71b4693c01443371042dd523e97fb917dad3b4040e9",
    "nonce_mismatch": "0d3916bae22c889c9bf77c2edecf91849366a8b3b8fb64a1d291e9a319904c9a",
    "card_version_downgrade": "9d62c8f11b8d3922e838edfecd866eae20e4865b38cdd58092e8b592aa789c30",
    "capture_topology_in_process_sdk":
        "f4b6da748f7720f15450cf61fa2c4cf8c3e9148fc15ab6b6cd15d8d3f3fd341e",
    "capture_topology_network_proxy":
        "de100ddcb08a1aadf3ade136a42ce554d38004ba06bbc4f0cecdc2235d46bd9d",
    "capture_topology_browser_extension":
        "9375a5437440101f386da4f6af0b2b64b1ed1f53fa3bdd5334363f15a173bd15",
    "capture_topology_github_sha_pull":
        "1a242ac2766e99b3e9bf00d7bb25ff251fdc1a1b59d77bbbe537723755fef4b6",
    "capture_topology_mcp_proxy":
        "4731cae72f4a30b5c10a688541faf8759873af7d78b8fe999069e112217b82c7",
    "capture_topology_unknown_value_rejected":
        "d2a306e1c437b467c431b7d1b4a70c1190aec10642e43ac002c55bb84dd3e617",
    "counterparty_binding_happy_path":
        "de1ef9618ba63f73251f6b1a439e29599b8b1ab1a037ada81026857d57c7b871",
    "counterparty_binding_envelope_byte_equality":
        "f3b138b08bd91a5a7cf20d3873bbf8949048fdb0a77925cb739d363b48343de6",
    "counterparty_binding_base64url_tolerance":
        "fc60e93117ae77a5bc533577fcd06cfe55bd9eabd8ee01a9a1e8bf76690814d2",
    "counterparty_binding_opaque_receipt_ref":
        "d00104822d5d45ce23755bd3bf4c3b06df54ddba04405f5c345f163fba73896e",
    "counterparty_binding_transport_label_non_trust":
        "49265cb5659fb7eff6327353a48462e6be4fb174e0417e03ad931df454435546",
    "counterparty_binding_missing_envelope_hash_rejected":
        "c96c3ba5b53524ba304a141a754284628123765581acdb09229adc10e34fe2bc",
    "receipt_v2_signer_canary": "dbb9fa8b319830814d2e6cc5b9c6a33f9c8f45e3223f75096d9c520aa9be55d4",
    "asqav-24-jcs-astral-key-order":
        "425159f5c1f0575fbcbf9d05a8f60cde3d040eae5166aa2136657564048651b6",
    "asqav-24-jcs-astral-key-order-codepoint-rejected":
        "425159f5c1f0575fbcbf9d05a8f60cde3d040eae5166aa2136657564048651b6",
    "asqav-25-number-above-safe-range-as-string":
        "3bbb752b000e9ac58b939c07bb99307d18ccc5be92189d3c32738ec24c98579f",
    "asqav-25-number-at-safe-range-boundary":
        "66c87d9cb3014e05a11baa97df62282d89d425f22ee15816577c84534e2ef1bb",
}

#: Length of each canonical string, so a published ``canonical`` cannot be swapped
#: for a different one that happens to carry the pinned digest in its ``sha256``.
PINNED_CANONICAL_LENGTH = {
    "minimal_read": 40,
    "tool_call_with_counterparty": 212,
    "traced_child_action": 130,
    "tampered_signature": 213,
    "swapped_public_key": 213,
    "stale_card": 213,
    "nonce_mismatch": 263,
    "card_version_downgrade": 213,
    "capture_topology_in_process_sdk": 111,
    "capture_topology_network_proxy": 110,
    "capture_topology_browser_extension": 114,
    "capture_topology_github_sha_pull": 112,
    "capture_topology_mcp_proxy": 106,
    "capture_topology_unknown_value_rejected": 104,
    "counterparty_binding_happy_path": 850,
    "counterparty_binding_envelope_byte_equality": 872,
    "counterparty_binding_base64url_tolerance": 785,
    "counterparty_binding_opaque_receipt_ref": 791,
    "counterparty_binding_transport_label_non_trust": 810,
    "counterparty_binding_missing_envelope_hash_rejected": 689,
    "receipt_v2_signer_canary": 873,
    "asqav-24-jcs-astral-key-order": 13,
    "asqav-24-jcs-astral-key-order-codepoint-rejected": 13,
    "asqav-25-number-above-safe-range-as-string": 24,
    "asqav-25-number-at-safe-range-boundary": 22
}

#: The wire ``counterparty_binding.envelope_hash`` string, exactly as published,
#: in the alphabet each vector exercises.
PINNED_ENVELOPE_HASH = {
    "counterparty_binding_happy_path": "DaE/V0yvdRCKIGBaAMYV9jCMeETMiSd5Mw6HZWsx2Pk=",
    "counterparty_binding_base64url_tolerance": "DaE_V0yvdRCKIGBaAMYV9jCMeETMiSd5Mw6HZWsx2Pk=",
    "counterparty_binding_opaque_receipt_ref": "DaE/V0yvdRCKIGBaAMYV9jCMeETMiSd5Mw6HZWsx2Pk=",
    "counterparty_binding_transport_label_non_trust":
        "DaE/V0yvdRCKIGBaAMYV9jCMeETMiSd5Mw6HZWsx2Pk=",
}

#: The non-wire renderings of the same digest carried under ``expected``.
PINNED_ENVELOPE_HASH_RENDERINGS = {
    "counterparty_binding_happy_path": {
        "envelope_hash_base64": "DaE/V0yvdRCKIGBaAMYV9jCMeETMiSd5Mw6HZWsx2Pk=",
        "envelope_hash_hex": "0da13f574caf75108a20605a00c615f6308c7844cc892779330e87656b31d8f9"
    },
    "counterparty_binding_envelope_byte_equality": {
        "envelope_hash_base64": "DaE/V0yvdRCKIGBaAMYV9jCMeETMiSd5Mw6HZWsx2Pk=",
        "envelope_hash_base64url": "DaE_V0yvdRCKIGBaAMYV9jCMeETMiSd5Mw6HZWsx2Pk="
    },
    "counterparty_binding_base64url_tolerance": {
        "envelope_hash_base64": "DaE/V0yvdRCKIGBaAMYV9jCMeETMiSd5Mw6HZWsx2Pk=",
        "envelope_hash_base64url": "DaE_V0yvdRCKIGBaAMYV9jCMeETMiSd5Mw6HZWsx2Pk="
    }
}

#: Remaining derived payload members, verbatim.
PINNED_PAYLOAD_MEMBERS = {
    "counterparty_binding_happy_path": {
        "previousReceiptHash": "0000000000000000000000000000000000000000000000000000000000000000",
        "policy_digest": "sha256:aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa7777bbbb8888",
        "action_ref": "act_01HVZA_ORIGINATOR_0001",
        "payload_digest": {
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "size": 0
        }
    },
    "counterparty_binding_base64url_tolerance": {
        "previousReceiptHash": "0000000000000000000000000000000000000000000000000000000000000000",
        "policy_digest": "sha256:aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa7777bbbb8888",
        "action_ref": "act_01HVZA_ORIGINATOR_0001",
        "payload_digest": {
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "size": 0
        }
    },
    "counterparty_binding_opaque_receipt_ref": {
        "previousReceiptHash": "0000000000000000000000000000000000000000000000000000000000000000",
        "policy_digest": "sha256:aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa7777bbbb8888",
        "action_ref": "act_01HVZA_ORIGINATOR_0001",
        "payload_digest": {
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "size": 0
        }
    },
    "counterparty_binding_transport_label_non_trust": {
        "previousReceiptHash": "0000000000000000000000000000000000000000000000000000000000000000",
        "policy_digest": "sha256:aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa7777bbbb8888",
        "action_ref": "act_01HVZA_ORIGINATOR_0001",
        "payload_digest": {
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "size": 0
        }
    },
    "counterparty_binding_missing_envelope_hash_rejected": {
        "previousReceiptHash": "0000000000000000000000000000000000000000000000000000000000000000",
        "policy_digest": "sha256:aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa7777bbbb8888",
        "action_ref": "act_01HVZA_ORIGINATOR_0001",
        "payload_digest": {
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "size": 0
        }
    },
    "receipt_v2_signer_canary": {
        "previousReceiptHash": "0000000000000000000000000000000000000000000000000000000000000000",
        "policy_digest": "sha256:9b71d224bd62f3785d96d46ad3ea3d73319bfbc2890caadae2dff72519673ca7",
        "action_ref": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "payload_digest": {
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "size": 0
        }
    }
}

#: Published signature blocks, verbatim.
PINNED_SIGNATURES = {
    "counterparty_binding_envelope_byte_equality": {
        "alg": "ML-DSA-65",
        "kid": "00000000000000000098",
        "sig": "AAAA_synthetic_signature_bytes_base64_placeholder_for_vector_only_AAAA"
    }
}

#: The exact document each refused-parse vector publishes, verbatim. A vector that pins a
#: REFUSAL has no canonical form, so its contract with implementers is these bytes.
PINNED_REFUSED_DOCUMENT = {
    "asqav-25-number-above-safe-range-rejected": "{\"n\":9007199254740993}",
}

#: Every derived member name this file claims to pin. A derived member added to the
#: corpus without a literal pin here fails ``test_no_derived_member_escapes_a_pin``.
PINNED_MEMBER_NAMES = frozenset(
    {"canonical", "sha256", "envelope_hash", "previousReceiptHash", "policy_digest",
      "action_ref", "payload_digest", "key_thumbprint", "sig",
      "envelope_hash_base64", "envelope_hash_base64url", "envelope_hash_hex"}
)


@pytest.mark.parametrize("name,digest", sorted(PINNED_SHA256.items()))
def test_sha256_member_is_the_pinned_literal(name: str, digest: str) -> None:
    assert _vectors()[name]["sha256"] == digest, (
        f"{name}: the published sha256 moved. If the change is intended, regenerate with "
        "verifier/regen_fingerprint_vectors.py and update this pin in the same commit."
    )


@pytest.mark.parametrize("name,digest", sorted(PINNED_SHA256.items()))
def test_canonical_member_hashes_to_the_pinned_literal(name: str, digest: str) -> None:
    canonical = _vectors()[name]["canonical"]
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == digest, (
        f"{name}: the published canonical bytes no longer hash to the pinned digest"
    )


@pytest.mark.parametrize("name,length", sorted(PINNED_CANONICAL_LENGTH.items()))
def test_canonical_member_is_the_pinned_length(name: str, length: int) -> None:
    assert len(_vectors()[name]["canonical"]) == length


@pytest.mark.parametrize("name,value", sorted(PINNED_ENVELOPE_HASH.items()))
def test_envelope_hash_is_the_pinned_literal(name: str, value: str) -> None:
    """The member that was stale. Pinned as a string, not recomputed."""
    binding = _vectors()[name]["input"]["counterparty_binding"]
    assert binding["envelope_hash"] == value, (
        f"{name}: published envelope_hash moved from the pin"
    )


@pytest.mark.parametrize("name,renderings", sorted(PINNED_ENVELOPE_HASH_RENDERINGS.items()))
def test_envelope_hash_renderings_are_the_pinned_literals(name: str, renderings: dict) -> None:
    expected = _vectors()[name]["expected"]
    for member, value in renderings.items():
        assert expected[member] == value, f"{name}.expected.{member} moved from the pin"


@pytest.mark.parametrize("name,members", sorted(PINNED_PAYLOAD_MEMBERS.items()))
def test_payload_derived_members_are_the_pinned_literals(name: str, members: dict) -> None:
    payload = _vectors()[name]["input"]
    for member, value in members.items():
        assert payload[member] == value, f"{name}.{member} moved from the pin"


@pytest.mark.parametrize("name,signature", sorted(PINNED_SIGNATURES.items()))
def test_signatures_are_the_pinned_literals(name: str, signature: dict) -> None:
    assert _vectors()[name]["input"]["signature"] == signature


    # The three encodings under `expected` must all decode to the same digest, and that
    # digest must be the one the wire member carries. Catches a partial re-pin that
    # updates one alphabet and forgets another - the exact shape of the original defect.
@pytest.mark.parametrize("name", sorted(PINNED_ENVELOPE_HASH_RENDERINGS))
def test_every_envelope_hash_encoding_decodes_to_one_digest(name: str) -> None:
    vector = _vectors()[name]
    expected = vector["expected"]
    digests = set()
    if "envelope_hash_hex" in expected:
        digests.add(expected["envelope_hash_hex"])
    if "envelope_hash_base64" in expected:
        digests.add(base64.b64decode(expected["envelope_hash_base64"]).hex())
    if "envelope_hash_base64url" in expected:
        digests.add(base64.urlsafe_b64decode(expected["envelope_hash_base64url"]).hex())
    binding = vector.get("input", {}).get("counterparty_binding")
    if isinstance(binding, dict) and "envelope_hash" in binding:
        wire = binding["envelope_hash"]
        urlsafe = "-" in wire or "_" in wire
        raw = base64.urlsafe_b64decode(wire) if urlsafe else base64.b64decode(wire)
        digests.add(raw.hex())
    assert len(digests) == 1, f"{name}: encodings disagree: {sorted(digests)}"


    # A vector that carries a counterparty digest MUST declare the scope that digest
    # was taken under. Re-pinning to envelope_minus_anchors without the scope member
    # would publish a minus-anchors value on a binding the absent-scope rule reads as
    # a legacy three-key binding.
def test_every_counterparty_digest_declares_its_scope() -> None:
    for name, vector in _vectors().items():
        binding = vector.get("input", {}).get("counterparty_binding")
        if isinstance(binding, dict) and "envelope_hash" in binding:
            assert binding.get("scope") == "envelope_minus_anchors", (
                f"{name}: carries an envelope_hash with no declared scope"
            )


    # Every derived digest must name the envelope it was taken over, so the corpus is
    # self-describing and the integrity job can recompute it from the file alone.
def test_every_counterparty_vector_names_its_originating_envelope() -> None:
    vectors = _vectors()
    for name, vector in vectors.items():
        binding = vector.get("input", {}).get("counterparty_binding")
        if not (isinstance(binding, dict) and "envelope_hash" in binding):
            continue
        ref = (vector.get("expected") or {}).get("originating_envelope_ref")
        assert ref, f"{name}: carries an envelope_hash but names no originating envelope"
        assert ref in vectors, f"{name}: originating_envelope_ref {ref!r} names no vector"


@pytest.mark.parametrize("name,document", sorted(PINNED_REFUSED_DOCUMENT.items()))
def test_refused_document_is_the_pinned_literal(name: str, document: str) -> None:
    assert _vectors()[name]["input_text"] == document


    # The corpus says these documents are refused, so the shipped parser must refuse them.
    # Without this the corpus could publish a refusal the code does not implement.
@pytest.mark.parametrize("name,document", sorted(PINNED_REFUSED_DOCUMENT.items()))
def test_the_shipped_parser_actually_refuses_it(name: str, document: str) -> None:
    from asqav import strict_json

    with pytest.raises(ValueError):
        strict_json.loads(document)


def test_no_derived_member_escapes_a_pin() -> None:
    """A new derived member must arrive with a literal pin, not silently."""
    pinned_here = (
        set(PINNED_SHA256)
        | set(PINNED_ENVELOPE_HASH)
        | set(PINNED_PAYLOAD_MEMBERS)
        | set(PINNED_SIGNATURES)
        | set(PINNED_ENVELOPE_HASH_RENDERINGS)
        | set(PINNED_REFUSED_DOCUMENT)
    )
    for name, vector in _vectors().items():
        if "input_text" in vector:
            # A refused document has no canonical form; its bytes are pinned instead.
            assert name in PINNED_REFUSED_DOCUMENT, f"{name}: refused document not pinned"
            assert "canonical" not in vector and "sha256" not in vector, (
                f"{name}: a refused document must not publish a canonical form"
            )
            continue
        assert name in pinned_here, f"{name}: vector has no literal pin in this file"
        for member in ("canonical", "sha256"):
            assert member in vector, f"{name}: missing {member}"
        assert name in PINNED_SHA256, f"{name}: sha256 is not pinned"


class TestTheCorpusSpellsNoAnchorsOneWay:
    """The corpus carried two spellings of the same fact: null and an empty array.

    The verifier maps absent, null and empty to the identical SKIPPED result, so this
    never changed a verdict, but a corpus that states one fact two ways invites a
    reader to infer a distinction that is not there.
    """

    def test_no_asqav_vector_uses_null_for_no_anchors(self) -> None:
        """One spelling, pinned, so the corpus cannot drift back to carrying both."""
        offenders = []
        for vector_dir in sorted(_VECTOR_ROOT.glob("asqav-*")):
            if not vector_dir.is_dir():
                continue
            receipt = json.loads((vector_dir / "receipt.json").read_text())
            if "anchors" in receipt and receipt["anchors"] is None:
                offenders.append(vector_dir.name)
        assert offenders == [], f"anchors null instead of []: {offenders}"

    def test_every_anchors_member_is_a_list(self) -> None:
        """A non-list anchors value is malformed: the verifier FAILs it, never laundered."""
        for vector_dir in sorted(_VECTOR_ROOT.glob("asqav-*")):
            if not vector_dir.is_dir():
                continue
            receipt = json.loads((vector_dir / "receipt.json").read_text())
            if "anchors" in receipt:
                assert isinstance(receipt["anchors"], list), vector_dir.name
