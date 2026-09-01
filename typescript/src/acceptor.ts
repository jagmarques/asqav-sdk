/**
 * Acceptor-side admission control for an inbound peer receipt.
 * A port of the Python `asqav.acceptor`, rule for rule.
 *
 * An Acceptor is the party on the receiving end of an agent-to-agent action.
 * This module answers one question: may this inbound action be admitted, given
 * the receipt the peer presented?
 *
 * It is deliberately NOT a thin wrapper over the verifier. Three of its rules do
 * not follow from the verdict alone, and each exists because a peer could
 * otherwise weaken the evidence without ever producing an unverified receipt:
 *
 *   Expiry.  The verifier reports expiry on its own axis and never folds it into
 *            the verdict, so a lapsed receipt still reads `verified`. Correct for
 *            a verifier - the signature really is good - and wrong for an
 *            acceptor, which is deciding about an action happening NOW.
 *
 *   Seq downgrade. A peer that has been emitting a counter and then stops makes
 *            contiguity uncheckable across that link. Absence has to stay legal
 *            in general (receipts predate the member), but an acceptor holding a
 *            predecessor that carried one is watching the exact transition that
 *            hides a withheld receipt.
 *
 *   Challenge. A challenge the acceptor issued but the receipt does not answer is
 *            a challenge that proved nothing. "Verify it when present" alone
 *            would let a peer skip freshness by omitting the member.
 *
 * The verification itself is the shared oracle's, not a reimplementation, so an
 * acceptor and an offline auditor cannot disagree about the same bytes.
 */

import { ADAPTERS } from "./verifier/index.js";
import {
  VERDICT_VERIFIED,
  VERDICT_VERIFIED_KEYED,
  verify,
  type FailureClass,
} from "./verifier/core.js";
import type { KeyProvider } from "./verifier/adapter.js";

/** Which acceptor rule refused, or `"verifier"` when the receipt did not verify. */
export type AcceptorRule = "verifier" | "expiry" | "seq_downgrade" | "challenge";

/** Whether to admit an inbound action, and the single reason why not. */
export interface AcceptorDecision {
  accepted: boolean;
  reason: string;
  verdict: string;
  failureClass: FailureClass | null;
  /** The earliest check that stopped the receipt, so a refusal points at one edge. */
  firstFailingEdge: string | null;
  rule: AcceptorRule | null;
}

export interface CheckPeerReceiptOptions {
  keyProvider?: KeyProvider;
  /** The last receipt this acceptor admitted from the same peer chain. */
  predecessor?: Record<string, unknown> | null;
  /** The nonce this acceptor issued for this exchange, if it issued one. */
  challenge?: string | null;
  /** Clock override; defaults to now. */
  now?: Date;
}

/**
 * `verified_keyed` is admissible because a keyed digest is internally consistent:
 * it is the peer's own hash, so it proves the same binding to the acceptor while
 * not being third-party re-derivable.
 */
const ADMISSIBLE = new Set<string>([VERDICT_VERIFIED, VERDICT_VERIFIED_KEYED]);

function parseStamp(raw: unknown): Date | null {
  if (typeof raw !== "string" || raw === "") return null;
  const d = new Date(raw);
  return Number.isNaN(d.getTime()) ? null : d;
}

function payloadOf(receipt: Record<string, unknown>): Record<string, unknown> {
  const p = receipt.payload;
  return typeof p === "object" && p !== null ? (p as Record<string, unknown>) : receipt;
}

/**
 * Decide whether an inbound action carrying `receipt` may be admitted.
 *
 * Supplying `predecessor` is what lets the seq and chain axes mean anything,
 * since neither can detect a gap against nothing.
 *
 * Refuses on the first rule that fails, in a fixed order, so the reason is
 * deterministic for the same inputs.
 */
export function checkPeerReceipt(
  receipt: Record<string, unknown>,
  options: CheckPeerReceiptOptions = {},
): AcceptorDecision {
  const { keyProvider = null, predecessor = null, challenge = null, now } = options;
  const result = verify(receipt, ADAPTERS, keyProvider, predecessor);

  if (!ADMISSIBLE.has(result.verdict)) {
    const edge = result.firstFailingEdge;
    return {
      accepted: false,
      reason: `peer receipt did not verify at ${edge ?? "an unnamed check"}`,
      verdict: result.verdict,
      failureClass: result.failureClass,
      firstFailingEdge: edge,
      rule: "verifier",
    };
  }

  const payload = payloadOf(receipt);

  // Expiry, which the verdict deliberately does not carry (criterion 426).
  const expiresAt = payload.expires_at;
  if (expiresAt !== undefined && expiresAt !== null) {
    const stamp = parseStamp(expiresAt);
    if (stamp === null) {
      return {
        accepted: false,
        reason: `unreadable expires_at ${JSON.stringify(expiresAt)}; refused rather than read as no expiry`,
        verdict: result.verdict,
        failureClass: "unverifiable",
        firstFailingEdge: "expiry",
        rule: "expiry",
      };
    }
    if ((now ?? new Date()).getTime() > stamp.getTime()) {
      return {
        accepted: false,
        reason: `peer receipt expired at ${String(expiresAt)}`,
        verdict: result.verdict,
        failureClass: "invalid",
        firstFailingEdge: "expiry",
        rule: "expiry",
      };
    }
  }

  // A peer that carried a counter and stopped is the one case where absence is
  // not the legacy case: it is the transition that makes a gap uncheckable.
  if (predecessor !== null && predecessor !== undefined) {
    const prevSeq = payloadOf(predecessor).seq;
    if (Number.isInteger(prevSeq) && !Number.isInteger(payload.seq)) {
      return {
        accepted: false,
        reason:
          `peer stopped emitting seq after ${String(prevSeq)}; contiguity cannot ` +
          "be checked across this link",
        verdict: result.verdict,
        failureClass: "unverifiable",
        firstFailingEdge: "seq",
        rule: "seq_downgrade",
      };
    }
  }

  // A challenge that goes unanswered proved nothing, so requiring it is the
  // whole point of having issued one.
  if (challenge !== null && challenge !== undefined) {
    const answered = payload.challenge_nonce;
    if (answered === undefined || answered === null) {
      return {
        accepted: false,
        reason: "acceptor issued a challenge and the receipt answers none",
        verdict: result.verdict,
        failureClass: "unverifiable",
        firstFailingEdge: "challenge_nonce",
        rule: "challenge",
      };
    }
    if (answered !== challenge) {
      return {
        accepted: false,
        reason: "receipt answers a different challenge than the one issued",
        verdict: result.verdict,
        failureClass: "invalid",
        firstFailingEdge: "challenge_nonce",
        rule: "challenge",
      };
    }
  }

  return {
    accepted: true,
    reason: "peer receipt verified and satisfies every acceptor rule",
    verdict: result.verdict,
    failureClass: null,
    firstFailingEdge: null,
    rule: null,
  };
}
