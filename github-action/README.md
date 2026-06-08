# Asqav Code-Authorship Receipt Action

Sign an Asqav code-authorship receipt from a CI run. On every pull request this
action records who authored a code change and signs that record with ML-DSA-65
(FIPS 204) server-side, then writes it out as an in-toto Statement.

## What it proves and what it does not

This action signs a small, honest set of facts:

- a change existed, namely the diff between a base commit and the head commit,
  captured as a SHA-256 digest,
- that change was key-authored, since the Asqav agent key signed the record,
- the record was chained at time T, as the receipt is hash-chained and
  timestamped by Asqav.

It does not verify the code, does not re-run the diff, does not resolve the
refs, and does not attest the model. The author identity, the tool, and any
model fields are producer-asserted. Model fields are always recorded with
`attestation_source=github-actions-input` so they can never be read as
Asqav-verified. In short: the receipt proves a change with a given digest was
recorded and key-signed at a point in time, nothing more.

## Usage

```yaml
name: code-authorship
on:
  pull_request:

permissions:
  contents: read

jobs:
  sign:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # full history so the base..head diff is available

      - name: Sign Asqav code-authorship receipt
        id: asqav
        uses: jagmarques/asqav-sdk/github-action@main
        with:
          asqav-api-key: ${{ secrets.ASQAV_API_KEY }}
          asqav-agent-id: ${{ secrets.ASQAV_AGENT_ID }}
          change-class: write
          model-id: claude-opus-4-8
          model-version: "2026-05"

      - name: Show receipt
        run: |
          echo "signature: ${{ steps.asqav.outputs.signature-id }}"
          echo "verify:    ${{ steps.asqav.outputs.verification-url }}"
          echo "in-toto:   ${{ steps.asqav.outputs.intoto-statement-path }}"
```

`fetch-depth: 0` matters: the action digests `git diff <base>..<head>`, so the
base commit must be present in the checkout.

## Inputs

| Input            | Required | Default          | Description |
| ---------------- | -------- | ---------------- | ----------- |
| `asqav-api-key`  | yes      |                  | Asqav API key in the form `sk_...`. Pass via a repository secret. |
| `asqav-agent-id` | yes      |                  | Asqav agent id whose key signs the receipt. |
| `change-class`   | no       | `write`          | One of `read`, `write`, `delete`, `execute`, `deploy`. |
| `model-id`       | no       | empty            | Producer-asserted model id. Recorded, never verified. |
| `model-version`  | no       | empty            | Producer-asserted model version. Recorded, never verified. |
| `tool`           | no       | `github-actions` | Producer-asserted authoring tool label. |

## Outputs

| Output                  | Description |
| ----------------------- | ----------- |
| `signature-id`          | Asqav signature id of the signed receipt. |
| `verification-url`      | Public URL to verify the signed receipt. |
| `intoto-statement-path` | Path to the emitted in-toto Statement v1 file. |

## How the receipt is derived

The action reads the git context from the GitHub Actions environment:

- `repo_ref` from `GITHUB_REPOSITORY`,
- `commit_sha` from `GITHUB_SHA`,
- `base_sha` from the pull request base in the event payload,
- `change_ref` from the pull request URL,
- `change_digest` = `sha256:` + SHA-256 of `git diff <base>..<head>`.

It then calls `agent.sign(...)` with
`receipt_type="protectmcp:lifecycle:code_authorship"`, `compliance_mode=True`,
and `policy_decision="none"`, since a code-authorship receipt records no policy
evaluation and signs under the no-policy opt-out.

## in-toto interop

The signed receipt is also wrapped as an
[in-toto Statement v1](https://github.com/in-toto/attestation) and written to a
file. The shape is:

```json
{
  "_type": "https://in-toto.io/Statement/v1",
  "subject": [
    {
      "name": "<owner/repo>@<commit_sha>",
      "digest": { "sha256": "<commit_sha>" }
    }
  ],
  "predicateType": "https://asqav.com/CodeAuthorship/v1",
  "predicate": { "...the signed receipt payload, signature alg ML-DSA-65..." }
}
```

Because the receipt is emitted as a standard in-toto Statement, it slots into
the same attestation workflows that already consume in-toto, including Sigstore
Rekor transparency-log entries and GitHub Artifact Attestations. Upload the file
as a build artifact, push it to a transparency log, or attach it to a release
the same way you would any other in-toto predicate. The Asqav predicate carries
the ML-DSA-65 signature envelope, so a consumer can re-verify the Asqav receipt
independently of the surrounding attestation envelope.
