# Asqav Code-Authorship Receipt Action

Record an Asqav code-authorship receipt from a CI run. On every pull request the
action computes an advisory digest of the change and asks the Asqav server to
sign an authoritative record of who authored it. The server signs with ML-DSA-65
(FIPS 204) and writes the record out as an in-toto Statement.

## What it proves and what it does not

The action does two honest things: it computes an advisory change digest (the
SHA-256 of `git diff <base>..<head>`), and it calls the Asqav server. The
server then does the authoritative work:

- it re-fetches the commit by sha from the repository,
- it recomputes the canonical diff itself,
- it signs an in-toto Statement whose `subject[0].digest.sha256` is the
  SERVER-recomputed diff hash, not the action's advisory value.

The action's digest is advisory. The server reports `digest_match` to say
whether the advisory value agreed with its own recomputation, and that flag is
informational: the binding subject is always the server digest. The author
identity is producer-asserted and recorded, never verified by Asqav.

So the receipt proves a server-verified change to a given commit was recorded
and key-signed at a point in time. The server independently re-derived the diff,
and the binding digest rests on that recomputation rather than on the action.

## Usage

```yaml
name: code-authorship
on:
  pull_request:

permissions:
  contents: read

jobs:
  record:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # full history so the base..head diff is available

      - name: Record Asqav code-authorship receipt
        id: asqav
        uses: jagmarques/asqav-sdk/github-action@main
        with:
          asqav-api-key: ${{ secrets.ASQAV_API_KEY }}
          change-class: write
          author: model:claude-opus-4-8

      - name: Show receipt
        run: |
          echo "signature:    ${{ steps.asqav.outputs.signature-id }}"
          echo "digest-match: ${{ steps.asqav.outputs.digest-match }}"
          echo "server-digest:${{ steps.asqav.outputs.server-digest }}"
          echo "in-toto:      ${{ steps.asqav.outputs.intoto-statement-path }}"
```

`fetch-depth: 0` matters: the action digests `git diff <base>..<head>` for its
advisory value, so the base commit must be present in the checkout.

## Inputs

| Input            | Required | Default          | Description |
| ---------------- | -------- | ---------------- | ----------- |
| `asqav-api-key`  | yes      |                  | Asqav API key (`sk_...`) with the `code_authorship:write` scope. Pass via a repository secret. |
| `change-class`   | no       | `write`          | One of `read`, `write`, `delete`, `execute`, `deploy`. |
| `author`         | no       | empty            | Producer-asserted author identity (for example `human:alice@example.com` or `model:claude-opus-4-8`). Recorded, never verified. |
| `anchor`         | no       | PR URL           | Anchor reference. Defaults to the pull request URL (or the run URL). |

## Outputs

| Output                  | Description |
| ----------------------- | ----------- |
| `signature-id`          | Asqav signature id of the recorded receipt. |
| `digest-match`          | `true` when the advisory digest agreed with the server digest, else `false`. |
| `server-digest`         | The server-recomputed diff digest bound into the Statement subject. |
| `capture-layer`         | The authoritative capture layer stamped by the server (`github_sha_pull`). |
| `intoto-statement-path` | Path to the emitted in-toto Statement v1 file. |

## How the receipt is derived

The action reads the git context from the GitHub Actions environment:

- `repo` from `GITHUB_REPOSITORY`,
- `commit_sha` from `GITHUB_SHA`,
- `base_sha` from the pull request base in the event payload,
- `change_digest` (advisory) = `sha256:` + SHA-256 of `git diff <base>..<head>`.

It then calls `POST /v1/code-authorship` with
`{repo, commit_sha, base_sha, change_digest}` plus the optional `change_class`,
`author`, and `anchor`. The server re-fetches the commit, recomputes the
canonical diff, and returns the signed in-toto Statement, the receipt, the
signing key id, the JWKS url, the `server_digest`, and `digest_match`. The
action writes the server's Statement to disk verbatim.

## How a third party verifies it

The receipt is independently checkable without trusting the action or the
producer:

1. re-fetch the `commit_sha` from the repository,
2. recompute the canonical diff: the sorted changed-files list, each entry
   `{path, status, additions, deletions, patch}`, canonicalized with JCS
   (the JSON Canonicalization Scheme),
3. SHA-256 those canonical bytes and confirm the result equals
   `subject[0].digest.sha256` in the Statement,
4. verify the ML-DSA-65 signature over the Statement using the key published at
   the `jwks_url`, resolved by the `kid`.

A match on step 3 proves the server bound the receipt to a diff anyone can
re-derive from the public commit. The `capture_layer` is `github_sha_pull`,
which is the authoritative layer: the server pulled the commit by sha and
recomputed the diff itself. A receipt carrying `in_process_sdk` or
`passive_telemetry` is a client self-report and is observation only, never an
authoritative decision receipt.

## in-toto interop

The server's signed receipt is an
[in-toto Statement v1](https://github.com/in-toto/attestation) written to a
file. The shape is:

```json
{
  "_type": "https://in-toto.io/Statement/v1",
  "subject": [
    {
      "name": "<owner/repo>@<commit_sha>",
      "digest": { "sha256": "<server-recomputed diff hash>" }
    }
  ],
  "predicateType": "https://asqav.com/attestation/code-authorship/v1",
  "predicate": {
    "capture_layer": "github_sha_pull",
    "asset_class": "code",
    "advisory_client_digest": "sha256:<the action's advisory digest>",
    "digest_match": true
  }
}
```

Because the receipt is a standard in-toto Statement, it slots into the same
attestation workflows that already consume in-toto, including Sigstore Rekor
transparency-log entries and GitHub Artifact Attestations. Upload the file as a
build artifact, push it to a transparency log, or attach it to a release the
same way you would any other in-toto predicate.
