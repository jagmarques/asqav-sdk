"""Tests for the legacy framework alias map in compliance.export_bundle.

The keys ``soc2``, ``eu_ai_act_art12``, ``eu_ai_act_art14`` and
``dora_ict`` were dropped from FRAMEWORKS in 0.4.4. To keep older bundles
and scripts working we accept them with a DeprecationWarning that maps
each to a current key. This test file pins that contract.
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from asqav.compliance import (
    FRAMEWORKS,
    LEGACY_FRAMEWORK_ALIASES,
    export_bundle,
)


def _signature_dict(idx: int = 0) -> dict:
    return {
        "signature_id": f"sid_{idx}",
        "signature": f"sig_{idx}",
        "action_id": f"act_{idx}",
        "timestamp": 1700000000.0 + idx,
        "verification_url": f"https://asqav.com/verify/sid_{idx}",
    }


def test_alias_map_keys_match_known_legacy_set() -> None:
    """Catch silent additions or removals: the alias set is a stable contract."""
    assert set(LEGACY_FRAMEWORK_ALIASES) == {
        "soc2",
        "eu_ai_act_art12",
        "eu_ai_act_art14",
        "dora_ict",
    }


def test_alias_targets_are_current_framework_keys() -> None:
    """Every alias must point to a key that still exists in FRAMEWORKS."""
    for legacy, current in LEGACY_FRAMEWORK_ALIASES.items():
        assert current in FRAMEWORKS, (
            f"alias {legacy!r} points to {current!r} which is not in FRAMEWORKS"
        )


@pytest.mark.parametrize("legacy", sorted(LEGACY_FRAMEWORK_ALIASES))
def test_legacy_key_emits_deprecation_warning_and_succeeds(legacy: str) -> None:
    """Legacy key still produces a usable bundle but warns the caller."""
    target = LEGACY_FRAMEWORK_ALIASES[legacy]
    with pytest.warns(DeprecationWarning, match=legacy):
        bundle = export_bundle([_signature_dict()], framework=legacy)

    assert bundle.framework == target
    assert bundle.framework_metadata == FRAMEWORKS[target]
    assert bundle.receipt_count == 1


def test_current_key_does_not_warn() -> None:
    """eu_ai_act is a real key, so it must not trip the deprecation path."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        bundle = export_bundle([_signature_dict()], framework="eu_ai_act")
    assert bundle.framework == "eu_ai_act"


def test_unknown_key_still_raises_value_error() -> None:
    """A truly unknown key must keep raising; the alias map is closed."""
    with pytest.raises(ValueError, match="nonexistent"):
        export_bundle([], framework="nonexistent")
