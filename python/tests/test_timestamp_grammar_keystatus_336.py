"""Timestamp grammar pinned on the key-status axis (criterion 336)."""

from __future__ import annotations

from asqav import acceptor
from asqav.verifier import verify_receipt as v

REVOKED_AFTER = "2020-06-01T00:00:00Z"
BASIC = "20200101T000000Z"


def test_basic_issued_at_fails_identically() -> None:
    """Basic issued_at FAILs as unparseable instead of parsing on 3.11+."""
    res, note = v.check_key_status("revoked", BASIC, REVOKED_AFTER, False)
    assert res == "FAIL"
    assert "unparseable" in note


def test_basic_revoked_at_fails_identically() -> None:
    """Basic revoked_at FAILs as unparseable on every interpreter."""
    res, note = v.check_key_status("revoked", REVOKED_AFTER, BASIC, False)
    assert res == "FAIL"
    assert "unparseable" in note


def test_week_date_issued_at_fails() -> None:
    """A week date fromisoformat accepts on 3.11+ is outside the grammar."""
    res, _ = v.check_key_status("revoked", "2020-W01-1", REVOKED_AFTER, False)
    assert res == "FAIL"


def test_extended_pairs_keep_their_answers() -> None:
    """The pin narrows the language, never the extended-form outcomes."""
    res, _ = v.check_key_status("revoked", "2020-05-01T00:00:00Z", REVOKED_AFTER, False)
    assert res == "SKIPPED"
    res, _ = v.check_key_status("revoked", "2020-05-01T00:00:00Z", REVOKED_AFTER, True)
    assert res == "PASS"
    res, _ = v.check_key_status("revoked", "2020-07-01T00:00:00Z", REVOKED_AFTER, False)
    assert res == "FAIL"
    res, _ = v.check_key_status("active", BASIC, REVOKED_AFTER, False)
    assert res == "PASS"


def test_acceptor_expiry_uses_the_pinned_grammar() -> None:
    """Acceptor expiry refuses basic and week-date stamps like skew does."""
    assert acceptor._parse_stamp(BASIC) is None
    assert acceptor._parse_stamp("2020-W01-1") is None
    stamp = acceptor._parse_stamp(REVOKED_AFTER)
    assert stamp is not None and stamp.tzinfo is not None
