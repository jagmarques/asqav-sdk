# Security Policy

## Reporting Vulnerabilities

Email security@asqav.com with details. We will respond within 48 hours.

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
