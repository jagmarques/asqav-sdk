"""SDK parity for the threat-framework taxonomy wire fields.

The cloud accepts seven optional fields on the signed Compliance
Receipt: ``mitre_techniques``, ``mitre_atlas``, ``owasp_llm_top10``,
``nist_ai_rmf``, ``iso_42001``, ``eu_ai_act_articles``, and
``rfc3161_timestamp``. The SDK forwards each verbatim, validates each
list field as a non-empty list of strings (each entry <= 128 chars),
and validates ``rfc3161_timestamp`` as base64. The cloud auto-flips
``framework_mappings_self_declared`` to ``true`` whenever any of the
six list fields is populated; the SDK does not need to set the flag.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import Agent

GOOD_MITRE_TECHNIQUES = ["T1059", "T1078"]
GOOD_MITRE_ATLAS = ["AML.T0051", "AML.T0043"]
GOOD_OWASP_LLM = ["LLM01", "LLM02"]
GOOD_NIST_AI_RMF = ["GOVERN-1.1", "MEASURE-2.7"]
GOOD_ISO_42001 = ["A.6.2.6"]
GOOD_EU_AI_ACT = ["Article-12", "Article-15"]
# Synthetic base64 stand-in for a TimeStampResp DER; the SDK only
# validates base64 shape, the cloud preserves the bytes verbatim.
GOOD_RFC3161_B64 = (
    "MIIGgzADAgEAMIIGegYJKoZIhvcNAQcCoIIGazCCBmcCAQMxDTALBglghkgBZQMEAgE="
)


def _agent() -> Agent:
    return Agent(
        agent_id="agent_threat_fw",
        name="threat-fw-agent",
        public_key="pk_xxx",
        key_id="key_threat_fw",
        algorithm="ML-DSA-65",
        capabilities=["api:*"],
        created_at=1700000000.0,
    )


def _ok_response() -> dict:
    return {
        "signature": "sig_b64",
        "signature_id": "sig_abc",
        "action_id": "act_abc",
        "timestamp": 1700000000.0,
        "verification_url": "https://asqav.com/verify/sig_abc",
        "algorithm": "ML-DSA-65",
        "chain_hash": "deadbeef",
    }


@pytest.mark.parametrize(
    "kwarg,value",
    [
        ("mitre_techniques", GOOD_MITRE_TECHNIQUES),
        ("mitre_atlas", GOOD_MITRE_ATLAS),
        ("owasp_llm_top10", GOOD_OWASP_LLM),
        ("nist_ai_rmf", GOOD_NIST_AI_RMF),
        ("iso_42001", GOOD_ISO_42001),
        ("eu_ai_act_articles", GOOD_EU_AI_ACT),
    ],
)
def test_taxonomy_list_forwarded_to_wire(kwarg: str, value: list[str]) -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            **{kwarg: value},
        )
    assert captured["body"][kwarg] == value


def test_rfc3161_timestamp_forwarded_to_wire() -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            rfc3161_timestamp=GOOD_RFC3161_B64,
        )
    assert captured["body"]["rfc3161_timestamp"] == GOOD_RFC3161_B64


def test_all_seven_fields_forwarded_together() -> None:
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            mitre_techniques=GOOD_MITRE_TECHNIQUES,
            mitre_atlas=GOOD_MITRE_ATLAS,
            owasp_llm_top10=GOOD_OWASP_LLM,
            nist_ai_rmf=GOOD_NIST_AI_RMF,
            iso_42001=GOOD_ISO_42001,
            eu_ai_act_articles=GOOD_EU_AI_ACT,
            rfc3161_timestamp=GOOD_RFC3161_B64,
        )
    body = captured["body"]
    assert body["mitre_techniques"] == GOOD_MITRE_TECHNIQUES
    assert body["mitre_atlas"] == GOOD_MITRE_ATLAS
    assert body["owasp_llm_top10"] == GOOD_OWASP_LLM
    assert body["nist_ai_rmf"] == GOOD_NIST_AI_RMF
    assert body["iso_42001"] == GOOD_ISO_42001
    assert body["eu_ai_act_articles"] == GOOD_EU_AI_ACT
    assert body["rfc3161_timestamp"] == GOOD_RFC3161_B64


@pytest.mark.parametrize(
    "kwarg",
    [
        "mitre_techniques",
        "mitre_atlas",
        "owasp_llm_top10",
        "nist_ai_rmf",
        "iso_42001",
        "eu_ai_act_articles",
    ],
)
def test_taxonomy_empty_list_rejected(kwarg: str) -> None:
    with pytest.raises(ValueError, match=f"{kwarg}_must_be_non_empty_list"):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            **{kwarg: []},
        )


def test_taxonomy_non_string_entry_rejected() -> None:
    with pytest.raises(ValueError, match="owasp_llm_top10_entry_invalid"):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            owasp_llm_top10=["LLM01", 42],
        )


def test_taxonomy_oversize_entry_rejected() -> None:
    with pytest.raises(ValueError, match="iso_42001_entry_invalid"):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            iso_42001=["A" * 129],
        )


def test_rfc3161_timestamp_empty_string_rejected() -> None:
    with pytest.raises(ValueError, match="rfc3161_timestamp_not_base64"):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            rfc3161_timestamp="",
        )


def test_rfc3161_timestamp_invalid_base64_rejected() -> None:
    with pytest.raises(ValueError, match="rfc3161_timestamp_not_base64"):
        _agent().sign(
            "api:call",
            {"k": "v"},
            compliance_mode=True,
            rfc3161_timestamp="not!!base64@@",
        )


def test_no_taxonomy_fields_omitted_from_wire() -> None:
    """The cloud must not see the new fields at all when caller omits them."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["body"] = body
        return _ok_response()

    with patch("asqav.client._post", side_effect=fake_post):
        _agent().sign("api:call", {"k": "v"}, compliance_mode=True)
    body = captured["body"]
    for k in (
        "mitre_techniques",
        "mitre_atlas",
        "owasp_llm_top10",
        "nist_ai_rmf",
        "iso_42001",
        "eu_ai_act_articles",
        "rfc3161_timestamp",
    ):
        assert k not in body, f"absent field {k} must not project onto wire"
