// Non-coverage declaration each verify() result carries (mirrors Python), so
// silence is not read as coverage; `condition` null = never run, else the input

/** One declared gap between what this verifier checks and a receipt claims */
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

// Longer than Python's list: this port runs no anchor cryptography, so checks
// Python performs partially are declared here as never performed
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
    reason:
      "no live CRL or OCSP lookup offline; revocation status is asserted " +
      "at trust-bundle issue time per root, so a later revocation is missed " +
      "until the bundle is refreshed (see the revocation policy in " +
      "docs/offline-verification.md)",
    condition:
      "a newer trust bundle (bundle_version/issued_at) or freshly re-pinned " +
      "--tsa-key material narrows the assertion lag",
  },
  {
    check: "tsa_trusted_list_status",
    requirement: "anchor trust",
    reason:
      "no EU Trusted List status lookup as of the token's genTime; " +
      "a trust bundle records status asserted at bundle-issue time only",
    condition: null,
  },
  {
    check: "policy_digest_resolution",
    requirement: "verifier check on policy_digest",
    reason:
      "the referenced policy artefact is not fetched, rehashed, or otherwise " +
      "evaluated; resolving it needs the Audit Pack manifest",
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

// Copied per call so a caller mutating one result cannot narrow later ones
/** Non-coverage declaration as a fresh list of fresh entries */
export function notCheckedDeclaration(): NotCheckedEntry[] {
  return NOT_CHECKED.map((entry) => ({ ...entry }));
}

// The prefix, then the adapter extras in order; coverage measures "not reached"
// against this sequence
/** Axis order verify() reports on the normal path for the native format */
export const AXIS_ORDER = [
  "structure",
  "signature",
  "chain",
  "seq",
  "expiry",
  "nonce",
  "key_binding",
  "counterparty",
  "payload_digest",
  "skew",
  "outer_copies",
  "key_status",
  "issuer_bind",
] as const;

/** One entry of the coverage block: a check that did not run, with why. */
export interface CoverageEntry {
  /** The check (a NOT_CHECKED row's `check`, or an axis name). */
  id: string;
  /** not_implemented: never evaluated; not_reached: evaluation stopped first */
  reason: "not_implemented" | "not_reached";
  /** Whether the tool implements the check at all. */
  status: "not_implemented" | "implemented";
  /** The requirement family, on not_implemented entries. */
  requirement?: string;
  /** Caller input that would enable the check, on not_implemented entries */
  condition?: string | null;
}

// A wire block mirrored from the reviewer's tool, so keys stay snake_case
/** Coverage block every result carries beside `notChecked` */
export interface Coverage {
  /** null when the full axis sequence ran, else the axis it stopped at */
  stopped_at: string | null;
  /** not_implemented first (table order), then not_reached in axis order */
  checks_not_evaluated: CoverageEntry[];
}

// Early returns in verify() stop at the structure gate, so stopped_at reads
// "structure" there; axes ending at the last AXIS_ORDER entry ran to the end
/** Build the coverage declaration for one result's axes */
export function coverageDeclaration(axes: ReadonlyArray<{ axis: string }>): Coverage {
  const entries: CoverageEntry[] = NOT_CHECKED.map((row) => ({
    id: row.check,
    reason: "not_implemented",
    status: "not_implemented",
    requirement: row.requirement,
    condition: row.condition,
  }));
  const names = axes.map((a) => a.axis);
  if (names.length > 0 && names[names.length - 1] === AXIS_ORDER[AXIS_ORDER.length - 1]) {
    return { stopped_at: null, checks_not_evaluated: entries };
  }
  const stoppedAt = "structure";
  const after = AXIS_ORDER.slice(AXIS_ORDER.indexOf(stoppedAt) + 1);
  for (const name of after) {
    if (!names.includes(name)) {
      entries.push({ id: name, reason: "not_reached", status: "implemented" });
    }
  }
  return { stopped_at: stoppedAt, checks_not_evaluated: entries };
}
