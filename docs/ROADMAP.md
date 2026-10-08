# Roadmap — azathoth-lm

**Date**: 2026-10-08
**Current best (enwik8)**: 1.1810 BPB (100KB, Phase 1 R53)
**Current composite (T1, 10KB)**: mean=1.4412 | sigma=0.2574 | worst=1.8184 (Phase 4 E3, R56)
**Current composite (T2b, 100KB)**: mean=1.8814 | sigma=1.4825 | worst=5.2470 (12 Silesia, pre-Phase 1)
**Current composite (T3, 100KB)**: mean=3.3788 | sigma=2.5716 | worst=7.9891 (11 modern, pre-Phase 1)
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
1.18  azathoth-lm   (0.1B RWKV + 14 CM + match + hier LSTM BPTT=64+ctx + emb surgery)
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
    ├─→ CM orders 0-2 + word → logistic sub-mixer → Group 0 (short ctx)
    ├─→ CM orders 3-8 + sparse + ICM + word → logistic sub-mixer → Group 1 (long ctx)
    ├─→ MatchModel (ctx 4-128) → bit preds (Group 3: match)
    ├─→ Byte context: bit_pos(8) + last_4_bytes(32) = 40 features
    └──────────── Top LSTM (H=128, 67K params, BPTT=64, coupled, LN, Adam, +40 ctx) → final P(bit=1)
```

### Key Gap vs Competition (R34 + Tier A/B + R51/R53 findings)

The LSTM mixer gap (R34) has been **substantially closed**: coupled gates, LayerNorm,
BPTT=64, Adam optimizer, byte context enrichment (40 features) all implemented
(S1-S3 + R53). Remaining gap vs cmix: **data scale** (100KB vs 100MB) and
**BPTT length** (cmix: 100 bytes = 800 bits, ours: 64 bits = 8 bytes).
Phase 1 (R53) confirmed: byte context + BPTT=64 passes T1 composite gate.

### Composite (Tier 1, 10KB each, Phase 1 R53, 2026-10-08)

| File | Type | BPB (Phase 1) | BPB (S3 baseline) | Delta | B/s |
|---|---|---|---|---|---|
| samba | Code | 1.1481 | 1.1445 | +0.0036 | ~90 |
| enwik8 | Text EN | **1.1666** | 1.1680 | **-0.0014** | 92 |
| dickens | Text EN | **1.5346** | 1.5465 | **-0.0119** | ~65 |
| mozilla | Binary | 1.6599 | 1.6404 | +0.0195 | 40 |
| OEIS | Numerical | **1.8200** | 1.8378 | **-0.0178** | 44 |
| **mean** | | **1.4658** | 1.4674 | **-0.0016** | |
| **sigma** | | **0.2677** | 0.3030 | **-0.0353** | |
| **worst** | | **1.8200** | 1.8378 | **-0.0178** | |

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
| ~~C1~~ | ~~Online LSTM expert (RATA-CMIX style)~~ | **KILLED (+0.0127)** | ~~High~~ | **R44.** Expert as external creates new mixer group. Group overhead > prediction value at 100KB. Code retained (--expert). |
| ~~C2~~ | ~~Information inheritance between CM orders~~ | **KILLED (by analysis)** | ~~Med~~ | Redundant with hierarchical mixer (already combines all orders). Same pattern as R33/B4: CM enhancement neutral with RWKV. |
| ~~C3~~ | ~~Modality-routing (OmniZip-inspired)~~ | **KILLED (by analysis)** | ~~High~~ | Violates no-domain-detection principle. Adding routing creates new mixer complexity → same regression pattern as C1/A2. |

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
| R45 | T2 final Silesia eval (12+2 files) | T1: mean=1.4674. T2: mean=2.2799. Telemetry. | **DONE** |
| R46 | Domain analysis + MoE feasibility | 4 clusters, 3 failure factors, MoE levels defined | **DONE** |
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
| C1 | Online LSTM expert | +0.0127 at 100KB. New group overhead > prediction value. | **KILLED** |
| C2 | Information inheritance | Redundant with mixer. Killed by analysis. | **KILLED** |
| C3 | Modality-routing | Violates no-domain-detection. Killed by analysis. | **KILLED** |

| R52 | Adaptive preprocessing (Phase 0) | KILLED: +4.13 mozilla. Incompatible with RWKV. | **KILLED** |
| R53 | Byte-context LSTM (Phase 1) | -0.0033 enwik8 100KB. T1 composite PASS (all 3 ↓). | **DONE** |
| R54 | Tweedie post-correction (Phase 2) | KILLED: +0.0293 (shrinkage), +0.0789 (calibration). Interferes with mixer. | **KILLED** |
| R55 | Neural blend (Phase 3) | T1 mean -0.0221, binary -0.0096. Expert pre-blend with RWKV, no new group. | **DONE** |
| R56 | Order-chain (Phase 4 E3) | T1 mean -0.0247, all 5 domains ↓. Chain order-N → N+1 hash. | **DONE** |
| R56 | Rank encoding (Phase 4 E4) | Mean +0.010, sigma +0.008. MTF context harms exact matching. | **KILLED** |

Full details, projections vs actuals, and lessons learned: `docs/CHANGELOG.md`.

## Remaining Trajectory (enwik8, from 1.1852)

Post Tier S + A + B + C. **All roadmap items exhausted at 100KB.**

### Fundamental finding: 100KB ceiling

Every approach tested — 17 experiments across 4 tiers — converges to
the same result: the architecture is at a **hard local minimum** at
100KB scale. The root cause is the hierarchical mixer:

- Adding new externals creates new groups → parameter overhead > information gain
- Adding model complexity (BPTT, layers, orders) → neutral due to limited data
- Post-mixer correction (APM/SSE/ISSE) → overcorrects with sparse bins

The ONLY gains that worked (S2 LayerNorm -0.0057, S3 BPTT=8 -0.0055) improved
existing components rather than adding new ones.

### Cross-domain analysis (R46)

Four domain clusters identified. Three orthogonal failure factors quantified:
tokenization quality (40% of BPB variance), intrinsic entropy (35%),
RWKV pretraining alignment (25%). Key insight: **context mixing IS soft MoE** —
the gap vs cmix is scale + preprocessing, not routing mechanism.

Dynamic CM instantiation analyzed and found risky at current scale:
- CMs with few observations produce confident but unreliable predictions
- LSTM mixer cannot distinguish real vs spurious confidence (no observation count)
- Adding/removing groups destabilizes mixer convergence (same pattern as R44)
- Only viable with self-gating + minimum observation threshold + >1MB data

### R51 — Organic Architecture Reform (replaces Tier N)

R50 (25-file, 100KB baselines) + ecosystem research (2024-2026) revealed that
our gap vs SOTA is **integration, not models**:
- BPTT=8 bits vs cmix's 128 (16x gap)
- Mixer blind to actual bytes (only sees 4 group logits)
- No post-correction (Tweedie proposed but KILLED R54 — see below)

**The old Tier N items (N4, N5, N6) are subsumed by R51's phased organic reform.**
See `docs/research/r51-organic-architecture-reform.md` for full analysis,
mathematical validation (3x verified per layer), and 11 research sources.

| Phase | Action | Est. Delta | Risk | Rationale |
|---|---|---|---|---|
| ~~**0**~~ | ~~Adaptive preprocessing (delta + byte-plane split)~~ | **KILLED (R52)** | — | Transforms destroy RWKV predictions. Incompatible with pre-trained models. |
| **1** | **Byte-context LSTM** (44 floats, BPTT=64) | **CONFIRMED (R53)** | Low | T1 composite gate PASS: mean -0.0016, sigma -0.0353, worst -0.0178. All 3 ↓. |
| ~~**2**~~ | ~~Tweedie post-correction~~ (2048 buckets, 24 KB) | **KILLED (R54)** | — | Both shrinkage and calibration regress. Second adaptation loop interferes with mixer's online learning. Same root cause as SSE. |
| **3** | **Neural blend** (44K expert pre-blend with RWKV) | **CONFIRMED (R55)** | Low | T1 composite PASS: mean -0.0221, sigma -0.0005, worst +0.0015. Binary -0.0096. |
| **4** | **CM order-chain** (E3) | **CONFIRMED (R56)** | Low | T1 composite PASS: mean -0.0247, sigma -0.0103, worst -0.0016. All 5 domains ↓. |
| ~~**4**~~ | ~~Rank-based encoding (E4)~~ | **KILLED (R56)** | — | Mean +0.010, sigma +0.008, worst +0.015. MTF context destroys exact matching at 10KB. |

**Unexplored edges** (E1-E6): byte-plane split, WHT feature expansion,
CM information inheritance, rank-based encoding, RWKV→uSSM self-distillation,
prediction horizon adaptation. Details in R51.

**Validation gate per phase** (R28): mean DOWN + sigma SAME/DOWN + worst not UP >0.05.
Phases are independent — failure of one does not block others.

### Projected path forward

```
1.1843  S3 baseline (post all optimization series)
1.1810  ✓ Phase 1 CONFIRMED (byte-context LSTM, BPTT=64) — R53
  ----  ✗ Phase 0 KILLED (preprocessing incompatible with RWKV) — R52
  ----  ✗ Phase 2 KILLED (post-correction fights adaptive mixer) — R54
1.1666  ✓ Phase 3 CONFIRMED (neural blend, -0.0096 binary) — R55
1.1633  ✓ Phase 4 E3 CONFIRMED (order-chain, all 5 domains ↓) — R56
  ----  ✗ Phase 4 E4 KILLED (rank encoding, MTF context harms matching) — R56
~1.15   CPU ceiling (enwik8), T2b ~1.50
<1.0    requires domain-tuned neural model (GPU)
```

### Legacy: Tier N items (superseded by R51)

| Old # | Status | Disposition |
|---|---|---|
| N1 | DONE (R47) | RWKV helps all 14 files. Diagnostic complete. |
| N2 | DEMOTED | Mixer input overhead pattern. Subsumed by R51 Phase 1 (byte context). |
| N3 | DONE (R49) | E8/E9 implemented. ooffice -0.18. |
| N4 | → R51 Phase 0 | Delta coding subsumed by adaptive preprocessing. |
| N5 | → R51 Phase 3/4 | Specialized CM subsumed by uSSM + CM order-chain. |
| N6 | Deferred | Full enwik8 100MB eval (8-29 days). Scale validation after R51 phases. |

## Constraints

- **CPU-only**: i5-1235U (Alder Lake), 12 threads, 32 GB DDR5, no GPU
- **RAM budget**: ~16 GB for inference (RWKV ~130 MB Q8, CM ~86 MB, match ~32 MB)
- **Throughput**: ~90-138 B/s depending on data type, full enwik8 ≈ 8-13 days
- **Max 1 heavy task**: concurrent evaluations cause CPU thrashing
- **Zero external deps**: all code must compile with rustc + stdlib only
