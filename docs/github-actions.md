# Asqav governance checks in GitHub Actions

Run governance validation on every pull request. The SDK ships a `doctor`
subcommand that validates configuration and connectivity, and a programmatic
`export_bundle` for packaging signed receipts into a self-contained,
Merkle-rooted compliance bundle. Both work in CI without changes.

This page shows a minimal workflow you can drop into `.github/workflows/`.

## Prerequisites

- An Asqav API key. Get one at <https://cloud.asqav.com>.
- Add the key as `ASQAV_API_KEY` under repository **Settings → Secrets and
  variables → Actions**.

Without the secret the `doctor` step reports `FAIL  ASQAV_API_KEY not set` and
fails the job, which is what you want.

## Minimal workflow

```yaml
# .github/workflows/governance.yml
name: governance

on:
  pull_request:
  push:
    branches: [main]

jobs:
  governance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install asqav
        run: pip install asqav

      - name: Validate setup
        env:
          ASQAV_API_KEY: ${{ secrets.ASQAV_API_KEY }}
        run: asqav doctor
```

That is the whole CI gate. `asqav doctor` checks the API key is present, the
backend is reachable, and prints the SDK version, then returns non-zero if any
check fails.

## Workflow with compliance bundle export

If your test job signs actions, capture each `SignedActionResponse` to a JSONL
file during the run, then export a bundle from that file as a workflow
artifact. The bundle is what the auditor downloads.

A minimal pytest fixture that appends every receipt:

```python
# conftest.py
import json
import pytest
from pathlib import Path

RECEIPTS_PATH = Path("ci-receipts.jsonl")

@pytest.fixture(autouse=True)
def capture_receipts(monkeypatch):
    """Append every agent.sign(...) response to ci-receipts.jsonl."""
    import asqav

    real_sign = asqav.Agent.sign

    def wrapped(self, action_type, *args, **kwargs):
        sig = real_sign(self, action_type, *args, **kwargs)
        with RECEIPTS_PATH.open("a") as f:
            f.write(json.dumps({
                "signature_id": sig.signature_id,
                "action_id": sig.action_id,
                "agent_id": self.agent_id,
                "action_type": action_type,
                "algorithm": sig.algorithm,
                "timestamp": sig.timestamp,
                "verification_url": sig.verification_url,
                "chain_hash": sig.chain_hash,
            }) + "\n")
        return sig

    monkeypatch.setattr(asqav.Agent, "sign", wrapped)
    yield
```

The matching workflow:

```yaml
name: governance

on:
  pull_request:
  push:
    branches: [main]

jobs:
  governance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install asqav and project deps
        run: |
          pip install asqav
          pip install -r requirements.txt

      - name: Validate setup
        env:
          ASQAV_API_KEY: ${{ secrets.ASQAV_API_KEY }}
        run: asqav doctor

      - name: Run tests (these sign actions via agent.sign)
        env:
          ASQAV_API_KEY: ${{ secrets.ASQAV_API_KEY }}
        run: pytest

      - name: Export compliance bundle
        run: |
          python <<'PY'
          import json
          from pathlib import Path
          from asqav.compliance import export_bundle

          path = Path("ci-receipts.jsonl")
          if not path.exists():
              print("no receipts captured this run, skipping bundle")
              raise SystemExit(0)

          receipts = [json.loads(line) for line in path.read_text().splitlines() if line]
          bundle = export_bundle(receipts, framework="eu_ai_act_art12")
          bundle.to_file("compliance-bundle.json")
          print(f"wrote {bundle.receipt_count} receipts, merkle_root={bundle.merkle_root}")
          PY

      - uses: actions/upload-artifact@v4
        if: hashFiles('compliance-bundle.json') != ''
        with:
          name: compliance-bundle
          path: compliance-bundle.json
          retention-days: 90
```

`export_bundle` accepts plain dicts, `SignatureResponse`, or
`SignedActionResponse`, so the JSONL-of-dicts pattern works. The bundle carries
framework metadata, every receipt, and a Merkle root over them, so an auditor
can verify it offline without calling back to Asqav.

Frameworks shipped today: `eu_ai_act_art12`, `eu_ai_act_art14`, `dora_ict`,
`soc2`. See `python/src/asqav/compliance.py` for the full list.

## Tips

- Pin `asqav` to a specific version once your governance setup is stable
  (`pip install asqav==0.4.4`). Lockstep upgrades catch breaking changes early.
- The `asqav doctor` step reads only environment variables and the public API,
  so it is safe to run on forks. The bundle export touches your API key and
  produces an artifact, so guard it with
  `if: github.event.pull_request.head.repo.fork == false` if you accept
  third-party PRs.
- Pair this with the
  [asqav-compliance GitHub Action](https://github.com/jagmarques/asqav-compliance)
  for static AI-framework detection. The two complement each other: that one
  covers static framework usage, this one covers runtime governance.

## Related

- [SDK Guide](https://asqav.com/docs/sdk) - full SDK reference
- [Compliance docs](https://asqav.com/compliance) - framework definitions and
  audit workflow
- [`docs/user-intent.md`](user-intent.md) - signing user authorization alongside
  the agent action
