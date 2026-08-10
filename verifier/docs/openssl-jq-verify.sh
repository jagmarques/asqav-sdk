#!/usr/bin/env bash
# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
#
# Executable half of docs/openssl-jq-walkthrough.md (criterion 421)
# Verifies an asqav receipt with jq, sha256sum, and openssl ONLY - no asqav
# component installed. Every recomputation step asserts a pinned value; the
# script exits nonzero on any mismatch and never reports a step it did not run
#
# The published receipt is ML-DSA-65. Native ML-DSA needs OpenSSL >= 3.5; on a
# weaker openssl the signature step is DOCUMENTED, not run, and only when the
# caller opts in with --mldsa-document-only (CI does, on purpose). Without the
# flag the script fails closed there. The Ed25519 part runs on any OpenSSL 3.x
#
# Usage: bash verifier/docs/openssl-jq-verify.sh [--mldsa-document-only]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX="$HERE/fixtures"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

MLDSA_DOC_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --mldsa-document-only) MLDSA_DOC_ONLY=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

for tool in jq openssl base64; do
  command -v "$tool" >/dev/null || { echo "FATAL: $tool not installed" >&2; exit 1; }
done
if command -v sha256sum >/dev/null 2>&1; then
  SHA256_TOOL=sha256sum
  sha256_stdin() { sha256sum | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
  SHA256_TOOL="shasum -a 256"
  sha256_stdin() { shasum -a 256 | cut -d' ' -f1; }
else
  echo "FATAL: neither sha256sum nor shasum found" >&2
  exit 1
fi

sha256_file() { sha256_stdin < "$1" | tr '[:upper:]' '[:lower:]'; }
byte_size() { wc -c < "$1" | tr -d '[:space:]'; }

fail() { echo "FAIL: $*" >&2; exit 1; }

pin_check() {
  local label="$1" file="$2" want_sha="$3" want_bytes="$4"
  [[ -f "$file" ]] || fail "$label: missing file $file"
  local got_sha
  got_sha="$(sha256_file "$file")"
  [[ "$got_sha" == "$want_sha" ]] || fail "$label: sha256 $got_sha != pinned $want_sha"
  local got_bytes
  got_bytes="$(byte_size "$file")"
  [[ "$got_bytes" == "$want_bytes" ]] || fail "$label: $got_bytes bytes != pinned $want_bytes"
  echo "ok [$label] $want_bytes bytes, sha256:$want_sha"
}

    # base64url (RFC 4648 section 5) to raw bytes, padding completed
b64url_decode() {
  local s
  s="$(cat)"
  s="${s//-/+}"
  s="${s//_//}"
  while [[ $(( ${#s} % 4 )) -ne 0 ]]; do s="${s}="; done
  printf '%s' "$s" | base64 -d
}

    # Wrap raw key bytes in SPKI DER + PEM: printf of the fixed prefix, then the key
write_pem() {
  local raw="$1" prefix="$2" pem="$3"
  { printf "$prefix"; cat "$raw"; } > "$WORK/spki.der"
  {
    echo "-----BEGIN PUBLIC KEY-----"
    openssl base64 -in "$WORK/spki.der"
    echo "-----END PUBLIC KEY-----"
  } > "$pem"
}

echo "asqav receipt walkthrough - jq + $SHA256_TOOL + openssl ($(openssl version))"
echo

# --- Step 0: the fixtures are byte-pinned copies of the locked corpus files ---
echo "step 0: fixture byte pins"
pin_check "published-receipt.json" "$FIX/published-receipt.json" \
  e5c81f01d570a9a1cf0aca8f6c773feba8ce01b9580d44ffac3a26dddbf22e63 11222
pin_check "published-jwks.json" "$FIX/published-jwks.json" \
  2ca81e3233f23ebdfa9f230c6914025f0dfd30a90c9cf322f70198c61d098bc4 2851
pin_check "ed25519-receipt.json" "$FIX/ed25519-receipt.json" \
  c02fd4fc8cc26d4784b99515f84d2c61faec8f8d18c1ce6ae72f7406ac3cf084 832
pin_check "ed25519-jwks.json" "$FIX/ed25519-jwks.json" \
  233a208e9564acea9f24faa2ffaa8bebce04d752bee685e581eef16c2b9c51a4 217
echo

# --- Part A: the published ML-DSA-65 prod receipt ---
echo "part A: published receipt (ML-DSA-65, real api.asqav.com payload-mode)"

# A.1: rebuild the canonical payload bytes with jq and compare the SHA-256
jq -cjS '.payload' "$FIX/published-receipt.json" > "$WORK/a_canonical.bin"
pin_check "A.1 canonical payload (jq -cjS .payload)" "$WORK/a_canonical.bin" \
  cc147376ec356902755eae69eb2134f5b8be9d570cd96ef685c6f708db1d2241 1097

# A.2: the anchors bind the envelope minus anchors; rebuild and pin it too
jq -cjS 'del(.anchors)' "$FIX/published-receipt.json" > "$WORK/a_env.bin"
pin_check "A.2 envelope minus anchors (jq -cjS 'del(.anchors)')" "$WORK/a_env.bin" \
  1252c422e91062dae94cf17bd8dc553132d8864ad671cf635476931c3469867e 5607

# A.3: extract the RFC 3161 anchor token and pin its bytes
anchor_type="$(jq -r '.anchors[0].type' "$FIX/published-receipt.json")"
[[ "$anchor_type" == "rfc3161" ]] || fail "A.3: anchor type $anchor_type != rfc3161"
jq -r '.anchors[0].value' "$FIX/published-receipt.json" | b64url_decode > "$WORK/a_anchor.der"
pin_check "A.3 rfc3161 anchor token" "$WORK/a_anchor.der" \
  80e3f6e83c416e33cd1f2f67593a80b294c42fe9bde99211e8975e6f275b07f2 3559

# A.4: the TSTInfo messageImprint inside the token is the A.2 digest, binding
# the timestamp to these exact envelope bytes. openssl asn1parse prints it
imprint="04201252C422E91062DAE94CF17BD8DC553132D8864AD671CF635476931C3469867E"
openssl asn1parse -inform DER -in "$WORK/a_anchor.der" > "$WORK/a_anchor_asn1.txt"
grep -q "$imprint" "$WORK/a_anchor_asn1.txt" \
  || fail "A.4: TSTInfo messageImprint $imprint not found in the anchor token"
echo "ok [A.4 anchor messageImprint binds the A.2 digest] $imprint"

# A.5: the ML-DSA-65 signature over the canonical payload bytes
jq -r '.signature.sig' "$FIX/published-receipt.json" | b64url_decode > "$WORK/a_sig.bin"
pin_check "A.5a signature bytes" "$WORK/a_sig.bin" \
  d573c39f39dff27a3001edc5d9bcc45b3b2da23774a48de9bd7e18a65a8f066e 3309
jq -r '.keys[0].public_key' "$FIX/published-jwks.json" | base64 -d > "$WORK/a_pk.raw"
[[ "$(byte_size "$WORK/a_pk.raw")" == "1952" ]] \
  || fail "A.5: ML-DSA-65 public key must be 1952 bytes"
# SPKI prefix: SEQUENCE{AlgorithmIdentifier(id-ml-dsa-65), BIT STRING header}
MLDSA_SPKI_PREFIX='\x30\x82\x07\xb2\x30\x0b\x06\x09\x60\x86\x48\x01\x65\x03\x04\x03\x12'
MLDSA_SPKI_PREFIX="${MLDSA_SPKI_PREFIX}\x03\x82\x07\xa1\x00"
write_pem "$WORK/a_pk.raw" "$MLDSA_SPKI_PREFIX" "$WORK/a_pk.pem"

MLDSA_CMD=(openssl pkeyutl -verify -pubin -inkey "$WORK/a_pk.pem" -rawin
  -in "$WORK/a_canonical.bin" -sigfile "$WORK/a_sig.bin")
if openssl list -signature-algorithms 2>/dev/null | grep -q 'ML-DSA-65'; then
  out="$("${MLDSA_CMD[@]}" 2>&1)" && rc=0 || rc=$?
  [[ $rc -eq 0 ]] || fail "A.5b: ML-DSA-65 verification failed (exit $rc): $out"
  echo "$out" | grep -q "Signature Verified Successfully" \
    || fail "A.5b: unexpected openssl output: $out"
  echo "ok [A.5b ML-DSA-65 signature verified with $(openssl version)] $out"
elif [[ "$MLDSA_DOC_ONLY" -eq 1 ]]; then
  echo "-- [A.5b ML-DSA-65 signature step: NOT EXECUTED here] -------------------"
  echo "   $(openssl version) has no ML-DSA support; OpenSSL >= 3.5 is required."
  echo "   The exact command and its pinned output on OpenSSL >= 3.5:"
  echo "     openssl pkeyutl -verify -pubin -inkey a_pk.pem -rawin \\"
  echo "       -in a_canonical.bin -sigfile a_sig.bin"
  echo "     => Signature Verified Successfully   (exit 0)"
  echo "   Verified against OpenSSL 3.6.2 when this pin was written. This line is a"
  echo "   documentation of that run, not a claim that it ran on this machine."
  echo "--------------------------------------------------------------------------"
else
  fail "A.5b: $(openssl version) cannot verify ML-DSA-65 (needs OpenSSL >= 3.5);" \
    "re-run with --mldsa-document-only to document the step instead"
fi

# A.6: full RFC 3161 chain walk (documented form). The token is signed with
# ML-DSA-44, so it needs an OpenSSL that parses ML-DSA in CMS AND the TSA trust
# anchor; neither is assumed here, so this step is never claimed as executed
echo "-- [A.6 full RFC 3161 chain walk: documented form, NOT EXECUTED] ---------"
echo "   openssl ts -verify -token_in -in a_anchor.der -data a_env.bin \\"
echo "     -CAfile <asqav-tsa-chain.pem>"
echo "   Requires the TSA trust chain published by asqav for this receipt's"
echo "   anchor; the binding this script verifies offline is the A.4 imprint."
echo "--------------------------------------------------------------------------"
echo

# --- Part B: the Ed25519 corpus receipt (runs on any OpenSSL 3.x) ---
echo "part B: Ed25519 receipt (asqav-native conformance vector)"

jq -cjS '.payload' "$FIX/ed25519-receipt.json" > "$WORK/b_canonical.bin"
pin_check "B.1 canonical payload (jq -cjS .payload)" "$WORK/b_canonical.bin" \
  88051bbc8ba5f41fd1626ade433b923d12ba70a1767201ee78c8b407fc56b580 533

jq -r '.signature.sig' "$FIX/ed25519-receipt.json" | base64 -d > "$WORK/b_sig.bin"
[[ "$(byte_size "$WORK/b_sig.bin")" == "64" ]] || fail "B.2: Ed25519 signature must be 64 bytes"
jq -r '.keys[0].public_key' "$FIX/ed25519-jwks.json" | base64 -d > "$WORK/b_pk.raw"
[[ "$(byte_size "$WORK/b_pk.raw")" == "32" ]] || fail "B.2: Ed25519 public key must be 32 bytes"
# SPKI prefix: SEQUENCE{AlgorithmIdentifier(id-Ed25519), BIT STRING header}
write_pem "$WORK/b_pk.raw" '\x30\x2a\x30\x05\x06\x03\x2b\x65\x70\x03\x21\x00' "$WORK/b_pk.pem"

out="$(openssl pkeyutl -verify -pubin -inkey "$WORK/b_pk.pem" -rawin \
  -in "$WORK/b_canonical.bin" -sigfile "$WORK/b_sig.bin" 2>&1)" && rc=0 || rc=$?
[[ $rc -eq 0 ]] || fail "B.2: Ed25519 verification failed (exit $rc): $out"
echo "$out" | grep -q "Signature Verified Successfully" \
  || fail "B.2: unexpected openssl output: $out"
echo "ok [B.2 Ed25519 signature verified] $out"
echo
echo "walkthrough complete: every executed step matched its pinned value"
