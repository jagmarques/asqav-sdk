"""The AGT integration's docs and CI job state their limits (criterion 763).

Runs without AGT installed, so the main python job checks it too.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC = (REPO / "docs" / "integrations-agt.md").read_text()
CI = (REPO / ".github" / "workflows" / "ci.yml").read_text()


def _agt_job() -> str:
    start = CI.index("\n  agt:\n")
    return CI[start : CI.index("\n  ci-ok:\n", start)]


def test_docs_describe_a_witness_and_its_limits() -> None:
    assert "They cannot produce a matching" not in DOC
    assert "independent signer" not in DOC
    assert "second, later Asqav receipt" in DOC
    assert "does not evaluate" in DOC
    assert "passes without\nconsulting Asqav" in DOC


def test_docs_disclose_the_cryptography_downgrade() -> None:
    for needle in ("49.0.0", "GHSA-g6cj-pr64-35w5", "CVE-2026-69247", "--override"):
        assert needle in DOC, needle


def test_ci_job_is_least_privilege_and_hash_pinned() -> None:
    job = _agt_job()
    assert "    permissions:\n      contents: read\n" in job
    urls = re.findall(r'@ (https://\S+?)"', job)
    assert len(urls) == 1
    assert re.search(r"/archive/[0-9a-f]{40}\.tar\.gz#sha256=[0-9a-f]{64}&subdirectory=", urls[0])
