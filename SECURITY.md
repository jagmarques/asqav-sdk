# Security Policy

## Reporting Vulnerabilities

Email info@asqav.com with details. We will respond within 48 hours.

Do not open public issues for security vulnerabilities.

## Supported Versions

| Version | Supported |
|---------|----------|
| 0.2.x   | Yes      |
| < 0.2   | No       |

## Cryptographic Details

- Signing: ML-DSA-65 (FIPS 204)
- Key storage: Server-side, never exposed to SDK
- Timestamps: RFC 3161 compliant
- All cryptography runs server-side via liboqs

## Known dependency advisories

These advisories are surfaced by Dependabot against `python/uv.lock` but are not exploitable in the default `pip install asqav` install path. The vulnerable packages live in `python/pyproject.toml` `optional-dependencies` extras and only enter the dependency graph when a caller installs that extra explicitly.

- `GHSA-jxgv-6j54-wwc7` (smolagents, low). Gated by the `[smolagents]` extra. Not installed by default.
- `GHSA-54fq-v6x8-244g` (smolagents, low). Gated by the `[smolagents]` extra. Not installed by default.
- `GHSA-w8v5-vhqr-4h9v` (diskcache, medium). Transitive of `dspy`; gated by the `[dspy]` extra. Not installed by default.
- `GHSA-vvw2-h478-xwr3` (dspy, medium). Gated by the `[dspy]` extra. Not installed by default.

Trivy and pip-audit against `python/uv.lock` both report 0 vulnerabilities in the default install path because their uv scanners treat marker-gated extras as not-installed-by-default. Dependabot remains stricter and flags them regardless.

If you install any of the affected extras (`pip install "asqav[smolagents]"`, `pip install "asqav[dspy]"`, or `pip install "asqav[all]"`) you opt in to the upstream advisory surface for those packages. Review the GHSAs above before pinning those extras in a production deployment.
