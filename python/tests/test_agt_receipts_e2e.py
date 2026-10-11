"""Asqav signs Microsoft AGT receipts and AGT's own verifier consumes the link (criterion 763).

Runs the real AGT package against a local stand-in for the Asqav agent and sign
endpoints on 127.0.0.1 that mints ML-DSA-65 receipts. Nothing reaches api.asqav.com.
ASQAV_AGT_REQUIRED=1 turns a missing AGT into a failure; ASQAV_AGT_PROFILE_REQUIRED=1
does the same for AGT's external-authorization profile.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import importlib
import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from asqav import client as asqav_client
from asqav.verifier import verify_receipt as vr

REQUIRED = os.environ.get("ASQAV_AGT_REQUIRED") == "1"
PROFILE_REQUIRED = os.environ.get("ASQAV_AGT_PROFILE_REQUIRED") == "1"


def _need(module: str) -> Any:
    return importlib.import_module(module) if REQUIRED else pytest.importorskip(module)


agt = _need("mcp_receipt_governed")
ML_DSA_65 = _need("dilithium_py.ml_dsa").ML_DSA_65
ed25519 = _need("cryptography.hazmat.primitives.asymmetric.ed25519")

agt_receipts = _need("asqav.agt_receipts")
AUTHORIZE_ACTION = agt_receipts.AUTHORIZE_ACTION
AUTHORIZER_PREFIX = agt_receipts.AUTHORIZER_PREFIX
COUNTERSIGN_ACTION = agt_receipts.COUNTERSIGN_ACTION
AgtSigningError = agt_receipts.AgtSigningError
AsqavAgtSigner = agt_receipts.AsqavAgtSigner
verify_asqav_link = agt_receipts.verify_asqav_link

ORG = "33333333-3333-4333-8333-333333333333"
LEI = "5493001KJTIIGC8Y1R12"
AGENT_ID = "agt_763"
KID = "key-763"
CEDAR = """
permit(principal, action == Action::"read_file", resource);
forbid(principal, action == Action::"delete_file", resource);
"""


def _seed() -> tuple[str, str]:
    key = ed25519.Ed25519PrivateKey.generate()
    seed = key.private_bytes_raw().hex()
    return seed, key.public_key().public_bytes_raw().hex()


class _AsqavStub:
    """Local stand-in for the Asqav agent-create and sign endpoints."""

    def __init__(self, legal_entity: str | None = None) -> None:
        self.issuer_id = legal_entity or ORG
        self.org_id_signed = True
        self.pk, self.sk = ML_DSA_65.keygen()
        self.thumb = vr.thumbprint_for_key(alg="ML-DSA-65", public_key=self.pk)
        self.sign_posts: list[dict[str, Any]] = []
        self.creates = 0
        self.fail_status: int | None = None
        self.decision = "allow"
        self.override_ref: str | None = None
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a: Any) -> None:
                pass

            def _send(self, doc: dict[str, Any], status: int = 200) -> None:
                raw = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if self.path == "/api/v1/agents/create":
                    stub.creates += 1
                    self._send(stub.agent_doc())
                elif self.path == f"/api/v1/agents/{AGENT_ID}/sign":
                    stub.sign_posts.append(body)
                    if stub.fail_status is not None:
                        self._send({"detail": "refused"}, stub.fail_status)
                    else:
                        self._send(stub.mint(body))
                else:
                    self._send({"detail": "not found"}, 404)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/api/v1"

    def jwks(self) -> dict[str, Any]:
        row = {
            "kid": KID,
            "agent_id": AGENT_ID,
            "issuer_id": self.issuer_id,
            "org_id": ORG,
            "alg": "ML-DSA-65",
            "kty": "AKP",
            "public_key": base64.b64encode(self.pk).decode(),
            "status": "active",
            "revoked_at": None,
            "key_thumbprint": self.thumb,
        }
        return {"keys": [row]}

    def agent_doc(self) -> dict[str, Any]:
        return {
            "agent_id": AGENT_ID,
            "name": "asqav-agt-signer",
            "public_key": base64.b64encode(self.pk).decode(),
            "key_id": KID,
            "algorithm": "ml-dsa-65",
            "capabilities": [],
            "created_at": "2026-10-11T00:00:00+00:00",
        }

    def mint(self, body: dict[str, Any]) -> dict[str, Any]:
        """Shape the payload like the cloud: org_id always, issuer_id is the legal entity if set."""
        seq = len(self.sign_posts)
        if "hash" in body:
            digest = {"hash": body["hash"][len("sha256:"):], "size": body["payload_size"]}
        else:
            encoded = vr.canonical_json(body["context"])
            digest = {"hash": hashlib.sha256(encoded).hexdigest(), "size": len(encoded)}
        issued_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "v": 1,
            "action_id": f"act_763_{seq}",
            "agent_id": AGENT_ID,
            "org_id": ORG,
            "policy_digest": "sha256:" + "3" * 64,
            "type": "protectmcp:decision",
            "issued_at": issued_at,
            "issuer_id": self.issuer_id,
            "action_ref": self.override_ref or body["action_ref"],
            "payload_digest": digest,
            "previousReceiptHash": "0" * 64,
            "decision": self.decision,
            "tool_name": body["action_type"],
            "key_thumbprint": self.thumb,
        }
        if "hash" in body:
            payload.update(mode="hash", hash=body["hash"], hash_algo="sha256")
            payload.update(metadata=body["metadata"], server_timestamp=issued_at)
        else:
            payload.update(mode="payload", action_type=body["action_type"])
            payload.update(context=body["context"], timestamp=issued_at)
        if not self.org_id_signed:
            del payload["org_id"]
        sig = ML_DSA_65.sign(self.sk, vr.canonical_json(payload))
        return {
            "signature": {"alg": "ML-DSA-65", "kid": KID, "sig": base64.b64encode(sig).decode()},
            "signature_id": f"sig_763_{seq}",
            "action_id": f"act_763_{seq}",
            "timestamp": payload["issued_at"],
            "verification_url": f"https://example.invalid/verify/sig_763_{seq}",
            "payload": payload,
            "anchors": [],
            "decision": self.decision,
            "compliance_mode": True,
        }


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch):
    yield from _serve(monkeypatch, _AsqavStub())


@pytest.fixture
def lei_stub(monkeypatch: pytest.MonkeyPatch):
    yield from _serve(monkeypatch, _AsqavStub(legal_entity=LEI))


def _serve(monkeypatch: pytest.MonkeyPatch, server: _AsqavStub):
    server.thread.start()
    for name in ("_api_key", "_api_base", "_client", "_mode", "_org_salt", "_timeout"):
        monkeypatch.setattr(asqav_client, name, getattr(asqav_client, name))
    asqav_client.init(api_key="sk_test_agt763", base_url=server.base)
    yield server
    server.httpd.shutdown()
    server.httpd.server_close()


@pytest.fixture
def profile() -> None:
    if agt_receipts.supports_external_authorization():
        return
    if PROFILE_REQUIRED:
        pytest.fail("the installed AGT lacks its external-authorization profile")
    pytest.skip("the installed AGT lacks its external-authorization profile")


def _agt_receipts(seed: str, *tools: str) -> list[Any]:
    adapter = agt.McpReceiptAdapter(
        cedar_policy=CEDAR, cedar_policy_id="p-763", signing_key_hex=seed
    )
    return [adapter.govern_tool_call("did:example:agent", t, {"path": "/tmp/a"}) for t in tools]


def _reload(receipt: Any) -> Any:
    """Round-trip through the JSON AGT exports, as an offline verifier would read it."""
    names = {f.name for f in dataclasses.fields(agt.GovernanceReceipt)}
    data = json.loads(json.dumps(receipt.to_dict()))
    return agt.GovernanceReceipt(**{k: v for k, v in data.items() if k in names})


# === Countersign: every AGT release ===


@pytest.mark.parametrize("mode", ["full-payload", "hash-only"])
def test_countersign_links_an_agt_verified_receipt(stub: _AsqavStub, mode: str) -> None:
    asqav_client.init(api_key="sk_test_agt763", base_url=stub.base, mode=mode)
    seed, agt_pub = _seed()
    receipts = _agt_receipts(seed, "read_file", "delete_file")
    assert agt.verify_receipt_chain(receipts, trusted_keys=[agt_pub]) == []

    signer = AsqavAgtSigner()
    links = [signer.countersign(r) for r in receipts]

    assert [p["action_ref"] for p in stub.sign_posts] == [link.action_ref for link in links]
    assert ("context" in stub.sign_posts[0]) is (mode == "full-payload")
    for receipt, link in zip(receipts, links):
        assert link.action_type == COUNTERSIGN_ACTION
        assert signer.links[receipt.receipt_id] is link
        offline = _reload(receipt)
        assert agt.verify_receipt(offline) is True
        assert verify_asqav_link(offline, link.asqav_receipt, jwks=stub.jwks()) == []
    assert agt.verify_receipt_chain([_reload(r) for r in receipts], trusted_keys=[agt_pub]) == []


def test_countersign_refuses_a_receipt_agt_rejects(stub: _AsqavStub) -> None:
    seed, _ = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    receipt.tool_name = "delete_file"
    assert agt.verify_receipt(receipt) is False
    with pytest.raises(AgtSigningError, match="agt_receipt_unverified"):
        AsqavAgtSigner().countersign(receipt)
    assert stub.sign_posts == []


def test_countersign_refuses_an_unsigned_receipt(stub: _AsqavStub) -> None:
    receipt = agt.GovernanceReceipt(tool_name="read_file", cedar_decision="allow")
    with pytest.raises(AgtSigningError, match="agt_receipt_unverified"):
        AsqavAgtSigner().countersign(receipt)
    assert stub.sign_posts == []


def test_link_does_not_transfer_to_another_receipt(stub: _AsqavStub) -> None:
    seed, _ = _seed()
    first, second = _agt_receipts(seed, "read_file", "read_file")
    link = AsqavAgtSigner().countersign(first)
    errors = verify_asqav_link(second, link.asqav_receipt, jwks=stub.jwks())
    assert errors == ["Asqav receipt does not bind this AGT receipt (action_ref mismatch)."]


def test_operator_rewrite_passes_agt_but_breaks_the_asqav_link(stub: _AsqavStub) -> None:
    seed, agt_pub = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    link = AsqavAgtSigner().countersign(receipt)
    rewritten = _reload(receipt)
    rewritten.args_hash = agt.hash_tool_args({"path": "/etc/shadow"})
    agt.sign_receipt(rewritten, seed)
    assert agt.verify_receipt_chain([rewritten], trusted_keys=[agt_pub]) == []
    assert verify_asqav_link(rewritten, link.asqav_receipt, jwks=stub.jwks()) == [
        "Asqav receipt does not bind this AGT receipt (action_ref mismatch)."
    ]


def test_forged_asqav_receipt_fails_the_signature_axis(stub: _AsqavStub) -> None:
    seed, _ = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    link = AsqavAgtSigner().countersign(receipt)
    forged = json.loads(json.dumps(link.asqav_receipt))
    forged["payload"]["policy_digest"] = "sha256:" + "4" * 64
    errors = verify_asqav_link(receipt, forged, jwks=stub.jwks())
    assert errors == ["Asqav receipt signature axis FAIL: signature mismatch"]
    other = _AsqavStub()
    errors = verify_asqav_link(receipt, link.asqav_receipt, jwks=other.jwks())
    assert "Asqav receipt signature axis FAIL: signature mismatch" in errors
    other.httpd.server_close()
    assert verify_asqav_link(receipt, link.asqav_receipt, jwks={"keys": []}) != []


def test_unbound_asqav_response_fails_closed(stub: _AsqavStub) -> None:
    seed, _ = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    stub.override_ref = "sha256:" + "0" * 64
    with pytest.raises(AgtSigningError, match="asqav_receipt_unbound"):
        AsqavAgtSigner().countersign(receipt)


@pytest.mark.parametrize("decision", ["deny", "rate_limit", "unknown"])
def test_refused_asqav_decision_fails_closed(stub: _AsqavStub, decision: str) -> None:
    seed, _ = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    stub.decision = decision
    with pytest.raises(AgtSigningError, match="asqav_decision_refused"):
        AsqavAgtSigner().countersign(receipt)


def test_observation_is_an_accepted_countersign(stub: _AsqavStub) -> None:
    seed, _ = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    stub.decision = "observation"
    link = AsqavAgtSigner().countersign(receipt)
    assert verify_asqav_link(receipt, link.asqav_receipt, jwks=stub.jwks()) == []


def test_verify_refuses_without_key_material(stub: _AsqavStub) -> None:
    seed, _ = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    link = AsqavAgtSigner().countersign(receipt)
    refused = ["Asqav signature not checked: pass the Asqav JWKS (asqav.fetch_jwks())."]
    assert verify_asqav_link(receipt, link.asqav_receipt) == refused
    assert verify_asqav_link(receipt, link.asqav_receipt, jwks=None) == refused
    unsigned = {"payload": dict(link.asqav_receipt["payload"])}
    assert verify_asqav_link(receipt, unsigned) == refused
    errors = verify_asqav_link(receipt, unsigned, jwks=stub.jwks())
    assert errors and all("action_ref" not in e for e in errors)


@pytest.mark.parametrize("decision", ["deny", "rate_limit"])
def test_verify_rejects_a_signed_refusal(stub: _AsqavStub, decision: str) -> None:
    seed, _ = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    AsqavAgtSigner().countersign(receipt)
    stub.decision = decision
    refusal = stub.mint(stub.sign_posts[0])
    assert verify_asqav_link(receipt, refusal, jwks=stub.jwks()) == [
        f"Asqav decision {decision!r} does not support {COUNTERSIGN_ACTION}."
    ]


def test_expected_org_pins_the_asqav_org(stub: _AsqavStub) -> None:
    seed, _ = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    link = AsqavAgtSigner().countersign(receipt)
    jwks = stub.jwks()
    assert verify_asqav_link(receipt, link.asqav_receipt, jwks=jwks, expected_org=ORG) == []
    other = "44444444-4444-4444-8444-444444444444"
    errors = verify_asqav_link(receipt, link.asqav_receipt, jwks=jwks, expected_org=other)
    assert f"Asqav receipt is not from org {other} (got {ORG!r})." in errors


@pytest.mark.parametrize("mode", ["full-payload", "hash-only"])
def test_expected_org_pins_a_legal_entity_org_by_its_signed_org_id(
    lei_stub: _AsqavStub, mode: str
) -> None:
    asqav_client.init(api_key="sk_test_agt763", base_url=lei_stub.base, mode=mode)
    seed, _ = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    link = AsqavAgtSigner().countersign(receipt)
    payload, jwks = link.asqav_receipt["payload"], lei_stub.jwks()
    assert (payload["issuer_id"], payload["org_id"]) == (LEI, ORG)
    assert verify_asqav_link(receipt, link.asqav_receipt, jwks=jwks, expected_org=ORG) == []
    other = "44444444-4444-4444-8444-444444444444"
    for pin in (other, LEI):
        errors = verify_asqav_link(receipt, link.asqav_receipt, jwks=jwks, expected_org=pin)
        assert f"Asqav receipt is not from org {pin} (got {ORG!r})." in errors


def test_expected_org_falls_back_to_issuer_id_without_org_id(stub: _AsqavStub) -> None:
    stub.org_id_signed = False
    seed, _ = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    link = AsqavAgtSigner().countersign(receipt)
    assert "org_id" not in link.asqav_receipt["payload"]
    jwks = stub.jwks()
    assert verify_asqav_link(receipt, link.asqav_receipt, jwks=jwks, expected_org=ORG) == []
    errors = verify_asqav_link(receipt, link.asqav_receipt, jwks=jwks, expected_org=LEI)
    assert f"Asqav receipt is not from org {LEI} (got {ORG!r})." in errors


VICTIM_ORG = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


def _resign(server: _AsqavStub, envelope: dict[str, Any], **edits: Any) -> dict[str, Any]:
    """Re-sign an Asqav envelope with the stand-in's own key after editing its payload."""
    forged = json.loads(json.dumps(envelope))
    for key, value in edits.items():
        if value is None:
            forged["payload"].pop(key, None)
        else:
            forged["payload"][key] = value
    sig = ML_DSA_65.sign(server.sk, vr.canonical_json(forged["payload"]))
    forged["signature"]["sig"] = base64.b64encode(sig).decode()
    return forged


@pytest.mark.parametrize(
    ("legal_entity", "edits", "key_org"),
    [
        pytest.param(VICTIM_ORG, {"org_id": None}, None, id="B2-no-org-id-legal-entity-is-victim"),
        pytest.param(LEI, {"org_id": VICTIM_ORG}, None, id="B3-org-id-claims-victim-under-lei"),
        pytest.param(None, {"org_id": VICTIM_ORG}, None, id="B4-org-id-claims-victim"),
        pytest.param(LEI, {"org_id": VICTIM_ORG}, VICTIM_ORG, id="B5-control-key-for-victim"),
    ],
)
def test_expected_org_needs_the_signing_key_published_for_that_org(
    monkeypatch: pytest.MonkeyPatch,
    legal_entity: str | None,
    edits: dict[str, Any],
    key_org: str | None,
) -> None:
    for server in _serve(monkeypatch, _AsqavStub(legal_entity=legal_entity)):
        seed, _ = _seed()
        (receipt,) = _agt_receipts(seed, "read_file")
        link = AsqavAgtSigner().countersign(receipt)
        forged = _resign(server, link.asqav_receipt, **edits)
        jwks = server.jwks()
        if key_org is not None:
            jwks["keys"][0]["org_id"] = key_org
        errors = verify_asqav_link(receipt, forged, jwks=jwks, expected_org=VICTIM_ORG)
        if key_org is not None:
            assert errors == []
        else:
            assert f"Asqav signing key is published under org {ORG}, not {VICTIM_ORG}." in errors


def test_operator_recountersign_shows_as_a_second_later_receipt(stub: _AsqavStub) -> None:
    seed, agt_pub = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    signer = AsqavAgtSigner()
    first = signer.countersign(receipt)
    rewritten = _reload(receipt)
    rewritten.args_hash = agt.hash_tool_args({"path": "/etc/shadow"})
    agt.sign_receipt(rewritten, seed)
    second = signer.countersign(rewritten)
    assert verify_asqav_link(rewritten, second.asqav_receipt, jwks=stub.jwks()) == []
    assert first.agt_receipt_id == second.agt_receipt_id
    assert first.signature_id != second.signature_id
    issued = [link.asqav_receipt["payload"]["issued_at"] for link in (first, second)]
    assert issued[0] <= issued[1]
    assert verify_asqav_link(receipt, second.asqav_receipt, jwks=stub.jwks()) != []


def test_asqav_sign_error_fails_closed(stub: _AsqavStub) -> None:
    seed, _ = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    stub.fail_status = 403
    signer = AsqavAgtSigner()
    with pytest.raises(AgtSigningError, match="asqav_sign_failed"):
        signer.countersign(receipt)
    assert signer.links == {}


@pytest.mark.parametrize("bad", [None, [], "x", {}, {"payload": "x"}, {"payload": None}])
def test_verify_link_fails_clean_on_malformed_input(stub: _AsqavStub, bad: Any) -> None:
    seed, _ = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    assert verify_asqav_link(receipt, bad, jwks=stub.jwks()) == [
        "Asqav receipt has no signed payload."
    ]
    errors = verify_asqav_link(object(), {"payload": {}}, jwks=stub.jwks())
    assert any(e.startswith("AGT receipt cannot be bound") for e in errors)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"authorizer_key_hex": "zz"}, "32-byte Ed25519 seed"),
        ({"authorizer_key_hex": "00" * 31}, "32-byte Ed25519 seed"),
        ({"authorization_ttl_seconds": 0}, "positive"),
        ({"authorization_ttl_seconds": float("nan")}, "finite"),
        ({"authorization_ttl_seconds": True}, "finite"),
    ],
)
def test_bad_configuration_fails_before_any_call(
    stub: _AsqavStub, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AsqavAgtSigner(**kwargs)
    assert stub.creates == 0


def test_hook_needs_the_profile_on_older_agt(stub: _AsqavStub) -> None:
    if agt_receipts.supports_external_authorization():
        pytest.skip("the installed AGT ships the external-authorization profile")
    seed, _ = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    auth_seed, _ = _seed()
    with pytest.raises(AgtSigningError, match="agt_external_authorization_unavailable"):
        AsqavAgtSigner(authorizer_key_hex=auth_seed)(receipt)
    assert stub.sign_posts == []


# === External authorization: AGT releases that ship authorize_receipt ===


def _authorized_adapter(signer: AsqavAgtSigner, seed: str) -> Any:
    return agt.McpReceiptAdapter(
        cedar_policy=CEDAR,
        cedar_policy_id="p-763",
        signing_key_hex=seed,
        external_authorizer=signer,
        trusted_authorizer_keys=[signer.authorizer_public_key],
    )


def test_agt_external_authorizer_carries_the_asqav_reference(stub: _AsqavStub, profile) -> None:
    seed, agt_pub = _seed()
    auth_seed, auth_pub = _seed()
    signer = AsqavAgtSigner(authorizer_key_hex=auth_seed)
    assert signer.authorizer_public_key == auth_pub
    adapter = _authorized_adapter(signer, seed)
    receipts = [
        adapter.govern_tool_call("did:example:agent", tool, {"path": "/tmp/a"})
        for tool in ("read_file", "delete_file")
    ]

    offline = [_reload(r) for r in receipts]
    assert agt.verify_receipt_chain(
        offline,
        trusted_keys=[agt_pub],
        trusted_authorizer_keys=[auth_pub],
        require_external_authorization=True,
    ) == []
    for receipt, post in zip(offline, stub.sign_posts):
        link = signer.links[receipt.receipt_id]
        assert receipt.assurance_level == "externally_authorized"
        assert receipt.authorizer_id == AUTHORIZER_PREFIX + link.signature_id
        assert link.action_type == AUTHORIZE_ACTION
        assert post["action_ref"] == link.action_ref
        assert verify_asqav_link(receipt, link.asqav_receipt, jwks=stub.jwks()) == []


def test_agt_verifier_rejects_a_rewritten_asqav_reference(stub: _AsqavStub, profile) -> None:
    seed, agt_pub = _seed()
    auth_seed, auth_pub = _seed()
    signer = AsqavAgtSigner(authorizer_key_hex=auth_seed)
    (receipt,) = [_authorized_adapter(signer, seed).govern_tool_call("did:x", "read_file")]
    link = signer.links[receipt.receipt_id]
    receipt.authorizer_id = AUTHORIZER_PREFIX + "sig_other"
    errors = agt.verify_receipt_chain(
        [receipt], trusted_keys=[agt_pub], trusted_authorizer_keys=[auth_pub],
        require_external_authorization=True,
    )
    assert errors == ["[0] External authorization signature is invalid."]
    assert verify_asqav_link(receipt, link.asqav_receipt, jwks=stub.jwks()) == [
        "AGT authorizer_id does not name this Asqav receipt."
    ]


def test_reauthorization_by_the_key_holder_breaks_the_asqav_link(stub: _AsqavStub, profile) -> None:
    seed, agt_pub = _seed()
    auth_seed, auth_pub = _seed()
    signer = AsqavAgtSigner(authorizer_key_hex=auth_seed)
    receipt = _authorized_adapter(signer, seed).govern_tool_call("did:x", "read_file")
    link = signer.links[receipt.receipt_id]
    agt.authorize_receipt(
        receipt, auth_seed, authorizer_id=receipt.authorizer_id,
        expires_at=receipt.authorization_expires_at + 86400, nonce="replayed",
    )
    assert agt.verify_receipt_chain(
        [receipt], trusted_keys=[agt_pub], trusted_authorizer_keys=[auth_pub],
        require_external_authorization=True,
    ) == []
    assert verify_asqav_link(receipt, link.asqav_receipt, jwks=stub.jwks()) == [
        "Asqav receipt does not bind this AGT receipt (action_ref mismatch)."
    ]


def test_authorization_needs_an_asqav_allow(stub: _AsqavStub, profile) -> None:
    seed, _ = _seed()
    auth_seed, _ = _seed()
    signer = AsqavAgtSigner(authorizer_key_hex=auth_seed)
    adapter = _authorized_adapter(signer, seed)
    stub.decision = "observation"
    with pytest.raises(AgtSigningError, match="asqav_decision_refused: observation"):
        adapter.govern_tool_call("did:x", "read_file")
    assert adapter.store.count == 0


@pytest.mark.parametrize("decision", ["observation", "deny"])
def test_verify_rejects_a_non_allow_authorization(stub: _AsqavStub, profile, decision: str) -> None:
    seed, _ = _seed()
    auth_seed, _ = _seed()
    signer = AsqavAgtSigner(authorizer_key_hex=auth_seed)
    receipt = _authorized_adapter(signer, seed).govern_tool_call("did:x", "read_file")
    link = signer.links[receipt.receipt_id]
    stub.decision = decision
    weaker = stub.mint(stub.sign_posts[0])
    weaker["signature_id"] = link.signature_id
    assert verify_asqav_link(receipt, weaker, jwks=stub.jwks()) == [
        f"Asqav decision {decision!r} does not support {AUTHORIZE_ACTION}."
    ]


def test_agt_verifier_rejects_an_untrusted_authorizer(stub: _AsqavStub, profile) -> None:
    seed, agt_pub = _seed()
    auth_seed, _ = _seed()
    _, stranger = _seed()
    signer = AsqavAgtSigner(authorizer_key_hex=auth_seed)
    receipt = _authorized_adapter(signer, seed).govern_tool_call("did:x", "read_file")
    errors = agt.verify_receipt_chain(
        [receipt], trusted_keys=[agt_pub], trusted_authorizer_keys=[stranger],
        require_external_authorization=True,
    )
    assert len(errors) == 1 and "Untrusted external authorizer" in errors[0]


def test_self_attested_receipts_fail_agt_required_authorization(stub: _AsqavStub, profile) -> None:
    seed, agt_pub = _seed()
    _, auth_pub = _seed()
    receipts = _agt_receipts(seed, "read_file")
    AsqavAgtSigner().countersign(receipts[0])
    errors = agt.verify_receipt_chain(
        receipts, trusted_keys=[agt_pub], trusted_authorizer_keys=[auth_pub],
        require_external_authorization=True,
    )
    assert errors == ["[0] External authorization is required but receipt is self-attested"]


def test_agt_adapter_fails_closed_when_asqav_fails(stub: _AsqavStub, profile) -> None:
    seed, _ = _seed()
    auth_seed, _ = _seed()
    signer = AsqavAgtSigner(authorizer_key_hex=auth_seed)
    adapter = _authorized_adapter(signer, seed)
    stub.fail_status = 403
    with pytest.raises(AgtSigningError, match="asqav_sign_failed"):
        adapter.govern_tool_call("did:x", "read_file")
    assert adapter.store.count == 0 and signer.links == {}


def test_hook_refuses_without_a_distinct_authorizer_key(stub: _AsqavStub, profile) -> None:
    seed, _ = _seed()
    (receipt,) = _agt_receipts(seed, "read_file")
    with pytest.raises(AgtSigningError, match="authorizer_key_required"):
        AsqavAgtSigner()(receipt)
    with pytest.raises(AgtSigningError, match="authorizer_key_reused"):
        AsqavAgtSigner(authorizer_key_hex=seed)(receipt)
    assert stub.sign_posts == []
