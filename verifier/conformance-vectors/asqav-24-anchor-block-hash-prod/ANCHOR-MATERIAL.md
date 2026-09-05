# Anchor material for asqav-24-anchor-block-hash-prod

Two pieces of PUBLIC material let the offline verifier complete this vector's
anchors axis. Neither is secret: the certificates are the ones embedded in the
receipt's own timestamp-authority token, and the block header is Bitcoin
public data.

- `tsa_trust.pem` — the two certificates embedded in the token (leaf
  `CN=tsa.izenpe.com`, issuer `CN=SUBCA QC IZENPE - TSA`), PEM. Byte equality
  with the token's embedded certificates was verified with the module's own
  parser (`_parse_time_stamp_resp(token)["certs"]` per block, both directions).
- `bitcoin_headers.json` — block 965451's header fields the OpenTimestamps
  check reads: `hash`, `merkle_root` (display hex; the checker reverses it to
  internal order), and `time` (ISO 8601 UTC). The proof's op chain evaluates to
  this merkle root, so the attestation lands in this block.

## Block 965451, fetched from two independent public sources

Fetch time for both: 2026-09-04T22:23:29Z (round 1 also fetched both on
2026-09-04 ~20:40Z; the two fetches agree).

`GET https://blockstream.info/api/block-height/965451`:

```
00000000000000000000e9f3195446ee74b371312941d73e8fdddab86499b499
```

`GET https://blockstream.info/api/block/00000000000000000000e9f3195446ee74b371312941d73e8fdddab86499b499`:

```json
{
    "id": "00000000000000000000e9f3195446ee74b371312941d73e8fdddab86499b499",
    "height": 965451,
    "version": 548012032,
    "timestamp": 1788510662,
    "tx_count": 4399,
    "size": 1512326,
    "weight": 3993560,
    "merkle_root": "76e472dea0ba9cb2adafe0d47ef54b5928c8a9443201300c848c77894e54c57c",
    "previousblockhash": "000000000000000000019162beabbdb079b50a0785c60571a1e523315c6f8926",
    "mediantime": 1788507899,
    "nonce": 3987889047,
    "bits": 386022593,
    "difficulty": 125807076547197.55
}
```

`GET https://mempool.space/api/block/00000000000000000000e9f3195446ee74b371312941d73e8fdddab86499b499`:

```json
{
    "id": "00000000000000000000e9f3195446ee74b371312941d73e8fdddab86499b499",
    "height": 965451,
    "version": 548012032,
    "timestamp": 1788510662,
    "tx_count": 4399,
    "size": 1512326,
    "weight": 3993560,
    "merkle_root": "76e472dea0ba9cb2adafe0d47ef54b5928c8a9443201300c848c77894e54c57c",
    "previousblockhash": "000000000000000000019162beabbdb079b50a0785c60571a1e523315c6f8926",
    "mediantime": 1788507899,
    "nonce": 3987889047,
    "bits": 386022593,
    "difficulty": 125807076547197.55
}
```

The two sources agree on `id`, `height`, `merkle_root` and `timestamp`
(1788510662 = 2026-09-04T08:31:02Z).
