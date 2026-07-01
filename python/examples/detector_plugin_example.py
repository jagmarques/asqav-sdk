"""Bring-your-own-detector: a pluggable DLP/policy gate on the sign path.

Register a detector and it runs inside every ``Agent.sign()`` call, after the
context is normalized and BEFORE the network call. Its verdict is recorded into
the signed receipt (under ``_detectors``), so the audit trail proves which
detector saw the action and why. A deny raises ``DetectorBlockedError`` and no
signature is created. Detection is pluggable; asqav owns the enforce + proof.

The reference detectors ``PresidioDetector`` (PII) and ``OpaDetector`` (policy)
live in ``asqav.extras`` behind optional extras. This example uses a tiny custom
regex detector so it runs with no extra install.

Requirements:
    pip install asqav

Usage:
    ASQAV_API_KEY=sk_... python examples/detector_plugin_example.py
"""

from __future__ import annotations

import os
import re

import asqav
from asqav import DetectorBlockedError, DetectorResult, register_detector

SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


class SsnDetector:
    """Deny any sign whose context contains a US SSN pattern."""

    name = "ssn-regex"

    def inspect(self, action_type: str, context: dict) -> DetectorResult:
        hit = SSN_RE.search(str(context))
        if hit:
            return DetectorResult(
                allow=False, confidence=1.0, labels=["US_SSN"], reason="SSN in context"
            )
        return DetectorResult(allow=True)


def main() -> int:
    api_key = os.environ.get("ASQAV_API_KEY", "")
    if not api_key:
        print("Set ASQAV_API_KEY before running. Get your key at https://asqav.com")
        return 1

    asqav.init(api_key=api_key)
    agent = asqav.Agent.create("detector-demo")

    # fail_open defaults to False: a detector error would block, not bypass.
    register_detector(SsnDetector())

    # Clean context: allowed, and the receipt records the detector verdict.
    print("Signing a clean action (detector allows)...")
    sig = agent.sign("email:send", {"to": "ops@acme.com", "body": "quarterly report"})
    print(f"  signature_id : {sig.signature_id}")
    print(f"  verify       : {sig.verification_url}")

    # Sensitive context: the detector denies, sign() blocks before the network.
    print("\nSigning an action with an SSN (detector denies)...")
    try:
        agent.sign("email:send", {"to": "ops@acme.com", "body": "ssn 123-45-6789"})
    except DetectorBlockedError as exc:
        print(f"  Blocked by  : {exc.detector_name}")
        print(f"  Labels      : {exc.result.labels}")
        print(f"  Docs        : {exc.docs_url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
