"""Tests for the `asqav verify-litellm-log` CLI command.

No network: verify_signature is patched. A tampered (unverifiable) line
must drive a non-zero exit so the rollup is non-vacuous.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typer.testing import CliRunner  # noqa: E402

from asqav.cli import app  # noqa: E402

runner = CliRunner()


def _write_log(tmp_path, sig_ids):
    path = tmp_path / "index.jsonl"
    with open(path, "w") as fh:
        for sid in sig_ids:
            fh.write(
                json.dumps(
                    {
                        "ts": 1700000000.0,
                        "model": "gpt-4",
                        "signature_id": sid,
                        "verification_url": f"https://api.asqav.com/verify/{sid}",
                        "messages_digest": "abc",
                    }
                )
                + "\n"
            )
    return str(path)


def _verify_result(verified: bool) -> MagicMock:
    r = MagicMock()
    r.verified = verified
    return r


class TestVerifyLitellmLog:
    def test_all_verified_rolls_up_and_exits_zero(self, tmp_path):
        path = _write_log(tmp_path, ["sid_1", "sid_2", "sid_3"])

        with patch("asqav.verify_signature", return_value=_verify_result(True)):
            result = runner.invoke(app, ["verify-litellm-log", path])

        assert result.exit_code == 0, result.output
        assert "verified 3/3" in result.output

    def test_tampered_line_fails_non_vacuous(self, tmp_path):
        path = _write_log(tmp_path, ["sid_ok", "sid_tampered"])

        def fake_verify(sig_id):
            return _verify_result(sig_id != "sid_tampered")

        with patch("asqav.verify_signature", side_effect=fake_verify):
            result = runner.invoke(app, ["verify-litellm-log", path])

        # One receipt fails verification -> non-zero exit, rollup shows 1/2.
        assert result.exit_code == 1, result.output
        assert "verified 1/2" in result.output
        assert "sid_tampered" in result.output

    def test_missing_file_exits_nonzero(self, tmp_path):
        result = runner.invoke(
            app, ["verify-litellm-log", str(tmp_path / "nope.jsonl")]
        )
        assert result.exit_code == 1
