# Asqav Risk-Acceptance Receipt Action

Mint a signed Asqav risk-acceptance receipt from CI when an exception pull
request is approved, and audit your scanner suppressions against the receipts
that back them. Signing is ML-DSA-65 (FIPS 204) server-side. Receipts support
recordkeeping obligations around security exceptions, with a built-in expiry
so an acceptance can never quietly become permanent.

## What it proves and what it does not

Mint mode signs a small, honest set of facts:

- a scanner finding existed, captured as the SARIF file's SHA-256 digest and
  the finding reference,
- a named approver accepted the risk for a named initiator, read from the PR
  approval event,
- the acceptance was recorded and key-signed at a point in time with a
  declared expiry.

It does not verify the finding, does not re-run the scanner, and does not
attest the GitHub identities. The approver login (the approving reviewer, or
the merging user on a merged pull request) and the initiator login (the PR
author) are producer-asserted and recorded, never verified by Asqav. The
SARIF file is digested on the runner and never uploaded.

## The expiry loop

Every acceptance carries an expiry. When it lapses, the audit mode fails the
CI run, the team removes the suppression, and the scanner flags the finding
again. The risk gets re-decided instead of forgotten:

1. Scanner flags a finding. The team decides to accept the risk for 90 days.
2. An exception PR adds the suppression. On approval, mint mode signs the
   acceptance with `expires: 90d` and the receipt id goes into the
   suppression manifest.
3. A scheduled workflow runs audit mode against the manifest. While the
   receipt is valid, the audit passes.
4. After 90 days the receipt expires, the audit exits nonzero, the
   suppression is removed, and the scanner re-flags the finding. Accepting it
   again means minting a new receipt, with `supersedes` pointing at the old
   one.

## Usage: mint on exception-PR approval

```yaml
name: risk-acceptance
on:
  pull_request_review:
    types: [submitted]

permissions:
  contents: read

jobs:
  mint:
    if: github.event.review.state == 'approved'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run security scanner
        run: my-scanner --output results.sarif

      - name: Mint Asqav risk-acceptance receipt
        id: asqav
        uses: jagmarques/asqav-sdk/github-action-risk-acceptance@main
        with:
          asqav-api-key: ${{ secrets.ASQAV_API_KEY }}
          asqav-agent-id: ${{ secrets.ASQAV_AGENT_ID }}
          sarif-file: results.sarif
          finding-id: "py/sql-injection:4f2a91c0"
          reason: "Input reaches a parameterized query, compensating control reviewed"
          expires: 90d

      - name: Show receipt
        run: |
          echo "signature: ${{ steps.asqav.outputs.signature-id }}"
          echo "verify:    ${{ steps.asqav.outputs.verification-url }}"
```

The approver is read from `github.event.review.user.login` and the initiator
from the PR author. On a merged `pull_request` event the merging user is
recorded as the approver instead.

## Usage: audit the suppression manifest

```yaml
name: suppression-audit
on:
  schedule:
    - cron: "0 6 * * *"
  pull_request:

permissions:
  contents: read

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Audit risk-acceptance receipts
        uses: jagmarques/asqav-sdk/github-action-risk-acceptance@main
        with:
          mode: audit
          asqav-api-key: ${{ secrets.ASQAV_API_KEY }}
          suppression-manifest: .security/suppressions.txt
```

The manifest lists the receipt ids your suppressions rely on, one per line
(`#` comments allowed), or as a JSON array, or as `{"receipts": [...]}`.

The audit fetches each receipt through the public verify endpoint and fails
when any referenced receipt is missing, fails verification, is expired
(evaluated server-side against the receipt's `valid_until`), carries a
receipt type other than `protectmcp:lifecycle:risk_acceptance`, or is named
in the `supersedes` field of another receipt in the manifest. Supersession is
evaluated client-side across the fetched set, so keep both the old and the
new receipt id in the manifest during a handover if you want the audit to
flag the old one.

## Inputs

| Input                  | Required          | Default | Description |
| ---------------------- | ----------------- | ------- | ----------- |
| `mode`                 | no                | `mint`  | `mint` or `audit`. |
| `asqav-api-key`        | yes               |         | Asqav API key in the form `sk_...`. Pass via a repository secret. |
| `asqav-agent-id`       | mint              |         | Asqav agent id whose key signs the receipt. |
| `sarif-file`           | mint              |         | Path to the SARIF file containing the flagged finding. Digested, never uploaded. |
| `finding-id`           | mint              |         | The SARIF ruleId plus fingerprint the exception covers. Recorded as `finding_ref`. |
| `reason`               | mint              |         | Why the risk is accepted. Recorded as `acceptance_reason`. |
| `expires`              | mint              |         | Duration (`90d`, `12h`, `30m`, `3600s`, `2w`) or ISO date. Converted to `valid_seconds`. |
| `supersedes`           | no                |         | Signature id of the acceptance this one replaces. |
| `risk-snapshot-json`   | no                |         | JSON object of producer-asserted third-party risk signals, passed through unmodified. |
| `suppression-manifest` | audit             |         | Path to the manifest of receipt ids. |

## Outputs

| Output             | Description |
| ------------------ | ----------- |
| `signature-id`     | Asqav signature id of the signed receipt (mint mode). |
| `verification-url` | Public URL to verify the signed receipt (mint mode). |

## How the receipt is derived

Mint mode calls `agent.sign(...)` with
`receipt_type="protectmcp:lifecycle:risk_acceptance"`, `compliance_mode=True`,
and `policy_decision="none"`, since a risk-acceptance receipt records a human
decision, not a policy evaluation. The field map:

- `approver_id` from the approving reviewer's login,
- `initiator_id` from the PR author's login,
- `acceptance_reason` from `reason`,
- `accepted_at` from the review's `submitted_at`,
- `finding_ref` from `finding-id`,
- `sarif_digest` = `sha256:` + SHA-256 of the SARIF file bytes,
- `supersedes` and `risk_snapshot` passed through when provided,
- `valid_seconds` from `expires`.
