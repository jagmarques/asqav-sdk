/**
 * The non-coverage declaration every `verify()` result carries, a mirror of the Python
 * verifier's `NOT_CHECKED`. A verifier that reports only what it checked lets a reader
 * mistake silence for coverage.
 *
 * `condition` is null when the check is never performed at any invocation, and names the
 * input that would enable it when the gap is one this caller can close. This list is longer
 * than the Python one: this port runs no anchor cryptography at all, so the checks Python
 * performs partially are declared here as never performed.
 */

/** One declared gap between what this verifier checks and what a receipt claims. */
export interface NotCheckedEntry {
  /** The check not performed (snake_case, stable across languages). */
  check: string;
  /** The requirement family the check belongs to. */
  requirement: string;
  /** Why it is not performed. */
  reason: string;
  /** The caller input that would enable it, or null when never performed. */
  condition: string | null;
}

export const NOT_CHECKED: readonly NotCheckedEntry[] = [
  {
    check: "tsa_signature_verification",
    requirement: "anchor trust",
    reason:
      "the RFC 3161 CMS signature over the receipt digest is never verified; an anchor " +
      "entry is shape-checked and reported unverifiable",
    condition: null,
  },
  {
    check: "tsa_certificate_path",
    requirement: "anchor trust",
    reason:
      "no X.509 chain walk from the RFC 3161 signing certificate to a public root, and " +
      "no pinned TSA key input exists to trust one through",
    condition: null,
  },
  {
    check: "tsa_certificate_revocation",
    requirement: "anchor trust",
    reason: "no CRL or OCSP lookup, so a revoked TSA certificate is not detected",
    condition: null,
  },
  {
    check: "policy_digest_resolution",
    requirement: "verifier check on policy_digest",
    reason:
      "the referenced policy artefact is not fetched or rehashed; the digest is checked " +
      "for shape only, and resolving it needs the Audit Pack manifest",
    condition: null,
  },
  {
    check: "aggregate_anchor_inclusion",
    requirement: "aggregate anchoring",
    reason:
      "where an anchor covers a batch rather than this receipt, the inclusion proof " +
      "linking this receipt to that aggregate is not walked",
    condition: null,
  },
  {
    check: "framework_mapping_claims",
    requirement: "taxonomy extension fields",
    reason:
      "the caller-supplied taxonomy lists are carried under the signature but their " +
      "content is not evaluated against any framework; a receipt can name a control it " +
      "never satisfied and this tool will not say so",
    condition: null,
  },
  {
    check: "receipt_set_completeness",
    requirement: "selective omission",
    reason:
      "this tool verifies the receipt it is handed; it cannot tell that a receipt was " +
      "withheld from the set it belongs to",
    condition: "the seq axis detects a gap only when the predecessor is supplied",
  },
  {
    check: "key_revocation_freshness",
    requirement: "key resolution",
    reason:
      "revocation state is read from the key directory as supplied; the tool does not " +
      "re-fetch it, so a key revoked after that snapshot reads current",
    condition: "the key directory passed by the caller is trusted as given",
  },
  {
    check: "opentimestamps_merkle_path",
    requirement: "anchor cryptographic re-verification",
    reason:
      "the proof's op chain is not evaluated to a merkle root over the receipt digest; " +
      "anchor entries are shape-checked only",
    condition: null,
  },
  {
    check: "opentimestamps_block_placement",
    requirement: "anchor cryptographic re-verification",
    reason:
      "no merkle path is computed, so there is nothing to land in a bitcoin block; the " +
      "tool carries no block-header input",
    condition: null,
  },
  {
    check: "counterparty_receipt_resolution",
    requirement: "counterparty binding",
    reason:
      "the bound peer envelope is not resolved from any store and this API accepts no " +
      "envelope to check against; the axis reports the claim unresolved rather than " +
      "checking the digest",
    condition: null,
  },
  {
    check: "chain_predecessor_retrieval",
    requirement: "hash-chain linkage",
    reason: "the predecessor receipt is not fetched; the link is checked only against one supplied",
    condition: "the predecessor payload passed by the caller",
  },
];

/**
 * Return the non-coverage declaration as a fresh list of fresh entries.
 *
 * Copied on every call so a caller mutating one result cannot narrow what later
 * results declare.
 */
export function notCheckedDeclaration(): NotCheckedEntry[] {
  return NOT_CHECKED.map((entry) => ({ ...entry }));
}
