"""sign() states the cloud's 409 nonce-replay refusal, never 'no uniqueness'.
NEGCTL: restoring the old nonce docstring entry fails both assertions."""

import inspect

from asqav.client import Agent


def test_sign_nonce_doc_names_the_409_refusal() -> None:
    src = inspect.getsource(Agent.sign)
    head = src[:src.index("nonce: str | None")]
    assert "409" in head.splitlines()[-2] + head.splitlines()[-3]
    assert "no uniqueness check" not in (inspect.getdoc(Agent.sign) or "")
