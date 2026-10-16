# Signed-conflict fixtures (488 C21, C7 core)

Two did:key agentreceipts geneses over one subject (same principal
`did:user:alice`, same read action) with conflicting outcomes (`success`
vs `failure/policy_denied`). Both carry valid Ed25519 signatures from one
key, so the oracle verifies both: the conflict is in the signed terms,
not in signature validity.

Regenerate byte-for-byte with `python ../verifier/gen_signed_conflict_vectors.py`
from `python/` (needs `cryptography`, already a `verify` extra). TEST-ONLY
key, seed `c7` * 32, published here and in the generator:

`did:key:z6MktqPnbGGsq5TR4h9dYRWtkoSQtiZt9CXfUFNcfxA7Fi5x`

This key signs fixtures only and protects nothing real.
