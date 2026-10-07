# R45: T2 Final Silesia Evaluation (Post All Tiers)

**Date**: 2026-10-07
**Status**: Complete
**Purpose**: Final Tier 2 cross-domain evaluation with structured telemetry after all roadmap tiers (S/A/B/C) exhausted.

## Config

```
RWKV-7 0.1B Q8 (129.6 MB) + 14 CM models (97.6 MB) + MatchModel (32.5 MB)
LSTM mixer: H=128, BPTT=8, coupled gates (i=1-f), LayerNorm, Adam(beta1=0.02, beta2=0.9999)
Embedding surgery: center0.3
Total params: ~100M neural + 51,330 mixer + 14 CM hash tables
```

## Results (10KB per file)

| File | Type | BPB | B/s | Tokens | bytes/tok |
|---|---|---|---|---|---|
| xml | Structured markup | 0.5212 | 58 | 4506 | 2.22 |
| nci | Chemical data | 0.5360 | 48 | 5838 | 1.71 |
| samba | Source code (C) | 1.1445 | 105 | 2851 | 3.51 |
| reymont | Polish text | 1.4778 | 53 | 6201 | 1.61 |
| dickens | English text | 1.5465 | 133 | 2460 | 4.07 |
| webster | English dict | 1.5664 | 122 | 2519 | 3.97 |
| mozilla | Executable | 1.6404 | 46 | 8065 | 1.24 |
| mr | Medical image | 1.9375 | 36 | 9341 | 1.07 |
| ooffice | Office binary | 2.5691 | 29 | 9680 | 1.03 |
| osdb | MySQL database | 4.2812 | 36 | 7721 | 1.30 |
| x-ray | Medical X-ray | 4.0903 | 30 | 9966 | 1.00 |
| sao | Astronomy SAO | 6.0483 | 29 | 9525 | 1.05 |

### Composite

| Metric | Value |
|---|---|
| **Mean (all 12)** | **2.2799** |
| **σ (all 12)** | **1.6843** |
| **Worst** | **6.0483** (sao) |
| Text-like mean (6) | 1.1321 |
| Binary mean (6) | 3.4278 |
| Mean throughput | ~62 B/s |

## Telemetry

Structured logs saved to `docs/results/t2-final/<file>.jsonl`.

Each log entry (per ~100 bytes) contains:
- `ts`: Barcelona timestamp (UTC+2)
- `bpb`: cumulative BPB at this point
- `bpb_w`: windowed BPB (last 100 bytes)
- `bps`: throughput (bytes/sec)
- `bit_costs`: array of 8 floats — per-bit cost within each byte [bit0..bit7]
- `match_hits`: cumulative match model hit count
- `match_avg_len`: average match length

### Throughput vs Token Density

Strong inverse correlation between throughput and token count:
- Low token density (text-like, ~2500 tokens) → fast: 105-133 B/s
- High token density (binary, ~9500 tokens) → slow: 29-36 B/s

Root cause: RWKV forward pass is per-token, so more tokens = more compute.
Binary data tokenizes poorly (1 byte ≈ 1 token), text tokenizes well (3-4 bytes/token).

### Domain Clustering

Three natural clusters emerge:

1. **Structured/repetitive** (BPB < 1.0): xml, nci — highly compressible
2. **Text-like** (BPB 1.0-2.0): samba, reymont, dickens, webster, mozilla, mr
3. **High-entropy binary** (BPB > 2.5): ooffice, osdb, x-ray, sao

The RWKV pretrained on text dominates clusters 1-2. Cluster 3 relies
primarily on CM models for any prediction advantage over raw entropy.

## Comparison with Post-Tier A

All 12 BPB values are **identical** to the post-Tier A evaluation.
This confirms that Tiers B (B1-B5) and C (C1-C3) had exactly zero
impact on cross-domain performance — all items were neutral or killed.

The only improvements between pre-S1 and final architecture came from:
- S2 (LayerNorm): -0.0057 on enwik8 100KB
- S3 (BPTT=8): -0.0055 on enwik8 100KB

These improvements affected enwik8 100KB but not the 10KB Silesia evals,
because S2/S3 primarily improve LSTM mixer convergence which needs
more data than 10KB to manifest.

## References

- R25: original Silesia evaluation (pre-surgery)
- R30: post-surgery Silesia baseline
- R28: composite BPB metric definition
- R34-R44: all Tier S/A/B/C experiments
