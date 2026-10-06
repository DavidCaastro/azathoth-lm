# Roadmap — azathoth-lm

**Date**: 2026-10-06
**Current best (enwik8)**: 1.1843 BPB (100KB, literature ref only)
**Current composite (T1)**: mean=1.5213 | sigma=0.2814 | worst=1.9045 (5 files, with surgery)
**Current composite (T2)**: mean=2.2799 | sigma=1.6843 | worst=6.0483 (12 Silesia files, post-Tier A + surgery)
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
1.18  azathoth-lm   (0.1B RWKV + 14 CM + match + hier LSTM BPTT=8 + emb surgery)
1.27  PAQ8px v217   (200+ byte-level CM, 3-layer mixer, universal)
1.19  NNCP v3       (199M Transformer-XL)
1.17  cmix v21      (2077 CM + 2x200 LSTM BPTT=100, universal)
1.11  ts_zip        (RWKV-169M v4 Q8, pure LM)
0.97  fx2-cmix-T    (6M Transformer Q4 + 2000+ CM, Hutter Prize Jul 2026)
0.94  Nacrith       (135M SmolLM2 + CM + Hedge mixer, universal)
```

Gap to target: ~0.18 BPB. Below NNCP (1.19), below PAQ8px.

### Architecture

```
Input bytes
    ├─→ Tokenizer → RWKV-7 0.1B Q8 → TokenByteTrie → bit preds (Group 2)
    ├─→ CM orders 0-2 → logistic sub-mixer → Group 0 (short ctx)
    ├─→ CM orders 3-8 + sparse + ICM → logistic sub-mixer → Group 1 (long ctx)
    ├─→ MatchModel (ctx 4-128) → bit preds (Group 3: match)
    └──────────── Top LSTM (H=128, 67K params, BPTT=1) → final P(bit=1)
```

### Key Gap vs Competition (R34 finding)

The #1 architectural deficit is NOT model count — it is **LSTM mixer depth**.
cmix uses BPTT=100 + Adam + LayerNorm + coupled gates. Our BPTT=1 makes the
LSTM essentially a feedforward net with persistent state. Closing this gap
is the highest-impact single action.

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

### Tier S — Critical path (implement first)

The LSTM mixer stack is the single biggest lever. Each item unlocks the next.

| # | Action | Est. Delta | Effort | Rationale |
|---|---|---|---|---|
| ~~S1~~ | ~~LSTM: coupled gates (i=1-f)~~ | **+0.0033 (neutral)** | ~~Low~~ | **DONE (R35).** -25% params (67K→50K), +15% speed (137→158 B/s). Prerequisite confirmed. |
| ~~S2~~ | ~~LSTM: LayerNorm~~ | **-0.0057 (100KB)** | ~~Med~~ | **DONE (R36).** Per-gate LN, +768 params. -0.0510 on 10KB (early boost). Prerequisite confirmed. |
| ~~S3~~ | ~~LSTM: BPTT=8 (1 full byte)~~ | **-0.0055 (100KB)** | ~~High~~ | **DONE (R37).** Adam(beta1=0.02, beta2=0.9999) + grad clip. -0.0017 on 10KB. 148 B/s (-4%). First temporal learning. |
| ~~S4~~ | ~~WordModel (case-folded + word-pair)~~ | **+0.0009 (neutral)** | ~~Med~~ | **DONE (R38).** Redundant with RWKV word-level understanding. +12 MB, -1% speed. Kept for diversity. |

### Tier A — High impact

| # | Action | Est. Delta | Effort | Rationale |
|---|---|---|---|---|
| ~~A1~~ | ~~APM/SSE 1-2 stages (distinct ctx)~~ | **KILLED (+0.10)** | ~~Low~~ | **R39.** +0.10 to +0.19 regression. LSTM well-calibrated, APM bins too sparse at 100KB. Only viable at full enwik8 (100MB). |
| ~~A2~~ | ~~Match model multi-input~~ | **KILLED (+0.001-0.013)** | ~~Low~~ | **R40.** Both multi-external and all-match variants regressed. Best-only match is optimal. Shorter matches add noise. |
| ~~A3~~ | ~~Tweedie denoising (Midicoth)~~ | **Demoted to C** | ~~Med~~ | Only validated on PPM pipelines (not after LSTM). APM/SSE validated in cmix. Interference risk. |

### Tier B — Medium impact

| # | Action | Est. Delta | Effort | Rationale |
|---|---|---|---|---|
| ~~B1~~ | ~~LSTM: 2 layers × 128~~ | **KILLED (+0.0003 at 100KB)** | ~~Med~~ | **R43.** 2×64 gives -0.0216 at 10KB but neutral at 100KB. Only 2 inputs to top LSTM → insufficient information for 2nd layer. Infra retained (--lstm-layers N). |
| ~~B2~~ | ~~BPTT scaling to 16-32~~ | **KILLED (+0.0000 at 100KB)** | ~~Med~~ | **R41.** BPTT=16 exactly neutral, BPTT=32 slight regression. BPTT=8 (1 byte) is optimal for bit-level LSTM. Cross-byte bit patterns are noise. |
| ~~B3~~ | ~~ISSE chains (3-5 stages)~~ | **KILLED (by analogy with A1)** | ~~Med~~ | Same family as APM/SSE (A1). Post-mixer correction overcorrects at 100KB scale. LSTM already well-calibrated. |
| ~~B4~~ | ~~Higher-order CM (orders 12, 16)~~ | **KILLED (+0.0004 at 100KB)** | ~~Low~~ | **R42.** Redundant with RWKV long-context. Confirms R33: CM count not bottleneck in hybrid. |
| ~~B5~~ | ~~WRT preprocessing (256→~205 symbols)~~ | **Deferred** | ~~Med~~ | Redundant with RWKV 65K tokenizer (like WordModel S4). Only viable for CM-only mode. |

### Tier C — Lower priority / speculative

| # | Action | Est. Delta | Effort | Rationale |
|---|---|---|---|---|
| C1 | Online LSTM expert (RATA-CMIX style) | -0.01 to -0.03 | High | 2×200 LSTM as predictor (not mixer). Generates own probabilities alongside RWKV. |
| C2 | Information inheritance between CM orders | -0.005 to -0.01 | Med | Lower-order estimates feed higher-order models (Chained Neural Predictors, April 2026). |
| C3 | Modality-routing (OmniZip-inspired) | -0.05 to -0.15 | High | Learned routing for binary data. Our binary BPB (3.52 mean) is 3x worse than text. |

### Blocked

| # | Action | Blocker |
|---|---|---|
| E1 | Domain checkpoint (fine-tune RWKV) | Requires GPU |
| E2 | Larger neural model (0.4B+) | No checkpoint outperforms 0.1B on enwik8 |
| E3 | Domain-trained small TF (fx2-cmix style) | Requires GPU for pre-training |

### Killed / Deprioritized (with justification)

| # | Action | Original Est. | Why killed | Source |
|---|---|---|---|---|
| ~~A1 old~~ | CM scaling 9→25+ models | -0.05 to -0.10 | Model count not bottleneck. R33 showed +0.0027 (neutral). Quality/diversity > quantity. WordModel is the ONE missing model. | R34 |
| ~~B2 old~~ | Hedge mixer experiment | -0.005 to -0.02 | Nacrith ablation: Hedge converges to w_llm≈1.0 (pass-through). Not useful when models have comparable strength. | R34 |
| ~~D1~~ | Full SA-PPM (suffix array) | -0.10 to -0.30 | Revised to -0.01 to -0.05. No top compressor uses suffix arrays. ppmonstr order-64 ≈ PPMd order-16. Match improvements capture 80% at 5% effort. | R34 |

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
| R30 | Frontier research + roadmap reform | T2 surgery + data-driven priorities | **DONE** |
| R31 | Pretrained symbiosis research | No 2nd neural; CM scaling is the path | **DONE** |
| R32 | G1d 0.1B checkpoint eval | +0.0693 enwik8, +0.3324 samba | **KILLED** |
| R33 | CM scaling A1 Phase 1 | +0.0027 BPB 100KB (neutral on text) | **DONE** |
| R34 | Roadmap audit + 5-thread research | LSTM depth is #1 gap, not CM count | **DONE** |
| S1 | LSTM: coupled gates (i=1-f) | +0.0033 neutral, -25% params, +15% speed | **DONE** |
| S2 | LSTM: LayerNorm per-gate | -0.0057 (100KB), -0.0510 (10KB early boost) | **DONE** |
| S3 | LSTM: BPTT=8 + Adam(beta1≈0) | -0.0055 (100KB), -0.0017 (10KB). 148 B/s. | **DONE** |
| S4 | WordModel (case-folded unigram + bigram) | +0.0009 (neutral). Redundant with RWKV. | **DONE** |
| A1 | APM/SSE post-LSTM | +0.10 to +0.19 regression | **KILLED** |
| A2 | Match multi-input | +0.001 to +0.013 regression | **KILLED** |
| B1 | 2-layer LSTM (2×128, 2×64) | +0.0003 at 100KB (neutral). -0.0216 at 10KB. | **KILLED** |
| B2 | BPTT scaling (16, 32) | +0.0000 at 100KB (neutral). Bit-level ceiling. | **KILLED** |
| B3 | ISSE chains | Killed by analogy with A1 | **KILLED** |
| B4 | Higher-order CM (12, 16) | +0.0004 at 100KB (neutral). Redundant with RWKV. | **KILLED** |

Full details, projections vs actuals, and lessons learned: `docs/CHANGELOG.md`.

## Remaining Trajectory (enwik8, from 1.1852)

Post Tier S + A + B analysis. Every incremental approach has been tested
and found neutral at 100KB. The architecture is at a local minimum.

Optimistic:
```
1.1852  current (post Tier S+A+B: only S2+S3 provided real gains)
1.17    + full enwik8 convergence (100MB, est. 8-13 days)
1.15    + online LSTM expert C1 (-0.01 to -0.03)
1.13    ceiling without GPU (optimistic)
```

Conservative:
```
1.1852  current
1.17    + full enwik8 convergence (-0.015)
1.16    ceiling without GPU (conservative)
```

Sub-1.0 requires domain-tuned neural model (GPU) or breakthrough in
online adaptation. All Tier A/B incremental improvements tested and
found neutral. Next gains likely require fundamentally different
approaches (C-tier) or scaling to full enwik8.

## Constraints

- **CPU-only**: i5-1235U (Alder Lake), 12 threads, 32 GB DDR5, no GPU
- **RAM budget**: ~16 GB for inference (RWKV ~130 MB Q8, CM ~86 MB, match ~32 MB)
- **Throughput**: ~90-138 B/s depending on data type, full enwik8 ≈ 8-13 days
- **Max 1 heavy task**: concurrent evaluations cause CPU thrashing
- **Zero external deps**: all code must compile with rustc + stdlib only
