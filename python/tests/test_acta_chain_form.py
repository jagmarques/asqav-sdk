# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""The ACTA chain recompute follows the form of the carried link (ACTA -03 §6.7).

The corpus vectors acta-06/07 pin the end-to-end outcomes; these unit tests pin
the adapter mechanics: prefixed carried value gets a prefixed recompute, bare
gets bare, and an unknown prefix is never normalised into a pass.
"""

from __future__ import annotations

import json
from pathlib import Path

from asqav.verifier.oracle.adapters.acta import ActaAdapter
from asqav.verifier.oracle.core import verify

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"


def _acta_06():
    d = _CORPUS / "acta-06-chain-link-03-prefixed"
    return (
        json.loads((d / "receipt.json").read_text()),
        json.loads((d / "predecessor.json").read_text()),
        json.loads((d / "acta-keys.json").read_text()),
    )


def test_chain_recompute_matches_the_carried_form() -> None:
    receipt, pred, _keys = _acta_06()
    adapter = ActaAdapter()
    step = adapter.chain_step(receipt)

    prefixed = step.recompute(pred)
    assert prefixed.startswith("sha256:")

    bare = dict(receipt["payload"])
    bare["previousReceiptHash"] = prefixed[len("sha256:"):]
    bare_step = adapter.chain_step({"payload": bare, "signature": receipt["signature"]})
    bare_recompute = bare_step.recompute(pred)
    assert not bare_recompute.startswith("sha256:")
    assert bare_recompute == prefixed[len("sha256:"):]


def test_unknown_prefix_fails_the_chain_axis() -> None:
    """A carried `sha512:` link is compared, not normalised: it must FAIL."""
    receipt, pred, keys = _acta_06()
    mutated = json.loads(json.dumps(receipt))
    carried = mutated["payload"]["previousReceiptHash"]
    assert carried.startswith("sha256:")
    mutated["payload"]["previousReceiptHash"] = "sha512:" + carried[len("sha256:"):]

    result = verify(mutated, [ActaAdapter()], key_provider=keys, predecessor=pred)
    chain = next(a for a in result.axes if a.axis == "chain")
    assert chain.result == "FAIL"
    assert result.verdict == "unverified"
