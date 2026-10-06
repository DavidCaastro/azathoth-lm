# Roadmap — azathoth-lm

**Date**: 2026-10-06
**Current best (enwik8)**: 1.1895 BPB (100KB, literature ref only)
**Current composite (T1)**: mean=1.5213 | sigma=0.2814 | worst=1.9045 (5 files, with surgery)
**Target**: < 1.0 BPB enwik8 + sigma decreasing — universal compressor
**Primary metric**: Composite BPB (mean, sigma, worst) — see R28

## Design Philosophy

azathoth-lm is a **universal data compressor**, not an enwik8 optimizer.
Every architectural decision must work on arbitrary byte streams.

Principles:
- **Byte/bit-level first**: all context models operate on raw bytes, not tokens
- **Data-agnostic**: no assumptions about input format or language
- **Neural + statistical**: RWKV for generalization, CM for exact pattern matching
- **Online adaptation**: all non-neural components learn during inference
- **Composite validation**: every change measured on 5 files (4 regimes), not 1

## Current Position

```
1.19  azathoth-lm   (0.1B RWKV + 9 CM + match + hier LSTM + emb surgery)
1.27  PAQ8px        (200+ byte-level CM, universal)
1.19  NNCP v3       (199M Transformer-XL)
1.17  cmix          (2077 byte-level models + LSTM mixer, universal)
1.11  ts_zip        (RWKV-169M v4 Q8, pure LM)
0.97  fx2-cmix      (6M Transformer + 2000+ CM, universal)
0.94  Nacrith       (135M SmolLM2 + byte-level CM, universal)
```

Gap to target: ~0.19 BPB. Tied with NNCP, below PAQ8px.

### Architecture

```
Input bytes
    ├─→ Tokenizer → RWKV-7 0.1B Q8 → TokenByteTrie → bit preds (Group 2)
    ├─→ CM orders 0-2 → logistic sub-mixer → Group 0 (short ctx)
    ├─→ CM orders 3-8 → logistic sub-mixer → Group 1 (long ctx)
    ├─→ MatchModel (ctx 4-128) → bit preds (Group 3: match)
    └──────────── Top LSTM (H=128, 67K params) → final P(bit=1)
```

Feature-complete. Remaining gap: **scale** (9+match vs 200+ models)
and **full SA-PPM** (suffix array for optimal matching).

### Composite Baseline (Tier 1, 10KB each, with emb surgery center0.3)

| File | Type | BPB |
|---|---|---|
| samba | Code | 1.1846 |
| enwik8 | Text EN | 1.2180 |
| dickens | Text EN | 1.5766 |
| mozilla | Binary | 1.7227 |
| OEIS | Numerical | 1.9045 |
| **mean** | | **1.5213** |
| **sigma** | | **0.2814** |
| **worst** | | **1.9045** |

## What's Next

### Near-term (est. impact on composite)

| # | Action | Est. Delta | Risk | Rationale |
|---|---|---|---|---|
| N1 | ICM for bits 3-5 | -0.02 to -0.05 | Low | Character identity bottleneck in text. PAQ8px proven technique. |
| N2 | APM post-mixer | -0.01 | Low | Adaptive probability map after LSTM. Nearly free compute. |
| N3 | Match model upgrade | -0.01 to -0.03 | Low | Checksums, LRU, confidence weighting. Reduce collision damage. |
| N4 | LSTM error-conditional lr | -0.005 to -0.01 | Low | Decay lr when error low, restore on regime change. PAQ8px v217. |

### Medium-term

| # | Action | Est. Delta | Risk | Rationale |
|---|---|---|---|---|
| M1 | Full SA-PPM (suffix array) | -0.10 to -0.30 | High | Unifies all context matching. Largest remaining single win. |
| M2 | Record model (tabular data) | -0.5 to -1.0 on numerical | Med | Detect fixed-length records, exploit column redundancy. OEIS/CSV. |
| M3 | Full enwik8 (100MB) | unknown | Low | Validate 100KB→100MB extrapolation. ~8 days compute. |

### Blocked

| # | Action | Blocker |
|---|---|---|
| B1 | Domain checkpoint (fine-tune RWKV) | Requires GPU |
| B2 | Larger neural model (0.4B+) | No checkpoint outperforms 0.1B on enwik8 |

## Completed Summary

| # | Action | Actual | Status |
|---|---|---|---|
| P1.1 | Arithmetic coder | overhead 0.0508 (10KB) | **DONE** |
| P1.2 | Byte-level CM | 2.09 BPB standalone | **DONE** |
| P1.3 | RWKV→byte bridge | -0.0060 BPB hybrid | **DONE** |
| P1.4 | Confidence skip | <3% speed gain | **KILLED** |
| P2.1 | LSTM mixer | -0.0375 BPB | **DONE** |
| P2.2 | Hierarchical groups | -0.0277 BPB | **DONE** |
| P2.3 | Multi-corpus validation | σ=1.72 (Silesia) | **DONE** |
| P3.1 | Match model | -0.0095 BPB | **DONE** |
| R27 | Embedding surgery | -0.0282 BPB, zero cost | **DONE** |
| R28 | Composite BPB metric | (mean,σ,worst) primary | **DONE** |
| R29 | OEIS numerical regime | 1.9045 BPB, 5th T1 file | **DONE** |

Full details, projections vs actuals, and lessons learned: `docs/CHANGELOG.md`.

## Remaining Trajectory (enwik8, from 1.1895)

Optimistic:
```
1.1895  current
1.16    + ICM for bits 3-5 (-0.03)
1.15    + APM + match upgrade (-0.01)
1.12    + full enwik8 (more history, LSTM convergence)
1.00    + full SA-PPM with suffix array (-0.12)
```

Conservative:
```
1.1895  current
1.17    + ICM + APM (-0.02)
1.15    + full enwik8 (-0.02)
```

< 1.0 requires SA-PPM or larger neural predictor (blocked by GPU).

## Constraints

- **CPU-only**: i5-1235U (Alder Lake), 12 threads, 32 GB DDR5, no GPU
- **RAM budget**: ~16 GB for inference (RWKV ~130 MB Q8, CM ~78 MB, match ~32 MB)
- **Throughput**: ~34-138 B/s depending on data type, full enwik8 ≈ 8.4 days
- **Max 1 heavy task**: concurrent evaluations cause CPU thrashing
- **Zero external deps**: all code must compile with rustc + stdlib only
