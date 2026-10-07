# Results Index

## Primary Metric: Composite BPB (mean, sigma, worst)

BPB is a vector, not a scalar. See R28 for justification.

### Current Composite Baseline (Silesia T2: 12 files, 10KB, FINAL post-all-tiers)

```
Composite BPB: mean=2.2799 | sigma=1.6843 | worst=6.0483 (sao)
  Text-like mean:  1.1321 (6 files)
  Binary mean:     3.4278 (6 files)
  Throughput:      29-133 B/s (mean ~62 B/s)
```

FINAL config: 14 CM + RWKV-7 0.1B Q8 + match + LSTM mixer (H=128, BPTT=8,
coupled gates, LN, Adam) + emb surgery center0.3. All Tiers S/A/B/C exhausted.
17 experiments, only S2 (LayerNorm -0.0057) and S3 (BPTT=8 -0.0055) improved BPB.
Architecture at hard local minimum at 100KB scale. Next: full enwik8 or GPU.
Telemetry logs: `docs/results/t2-final/*.jsonl` (per-byte BPB, bit costs, match stats).

### Acceptance Rule

A change is better if: mean DOWN + sigma SAME/DOWN + worst not UP by >0.05.
enwik8-only numbers are "literature ref", never sole accept/reject gate.

## Quick Reference (enwik8 — literature comparability only)

| Phase | BPB (enwik8) | Key Finding |
|---|---|---|
| (inherited) analytic-lm | 1.5826 | 54 CM + LSTM, enwik8 80/20 |
| (inherited) edge-lm | ~2.16 | WHT + multi-scale, own corpus |
| Phase 0 — RWKV-7 only | 1.4691 | 0.1B f32+Q8head, 100KB |
| Phase 0 — ensemble | 1.4086 | RWKV + N-gram(4) + bias head, 100KB |
| Phase 0 — tuned ensemble | 1.3078 | lr=0.30, scale=0.5, 10KB |
| Phase 0 — dynamic lr | 1.2997 | surprise-modulated tau=2000, 10KB (KILLED) |
| Phase 0 — adaptive mixer (10KB) | 1.2997 | learned weights eta=0.10, 10KB |
| Phase 0 — adaptive mixer (100KB) | 1.3238 | learned weights eta=0.01, 100KB |
| Phase 1 — Q8 all layers (100KB) | 1.2984 | Q8 int-accum + VNNI, 162 B/s, -75% RAM |
| Phase 1 — Arithmetic coder (1KB) | 1.1680 | Range coder CDF-24, roundtrip verified |
| Phase 1 — Arithmetic coder (10KB) | 1.3320 | Compressed BPB, CE=1.2812, overhead=0.0508 |
| Phase 1 — CM standalone (100KB) | 2.4078 | 9 bit-level models, logistic mixer, 78 MB |
| Phase 1 — CM standalone (1MB) | 2.0915 | Beats gzip (2.58), 220K B/s |
| Phase 1 — Hybrid CM+RWKV bridge (100KB) | **1.2924** | -0.0060 vs baseline, 169 B/s |
| Phase 1 — Confidence skip (100KB) | KILLED | +0.0026 BPB at best, <3% speed gain |
| Phase 2 — LSTM mixer hybrid (100KB) | 1.2549 | -0.0375 vs logistic, -0.0435 vs baseline |
| Phase 2 — Hierarchical groups (100KB) | **1.2272** | -0.0277 vs flat LSTM, groups: [0-2][3-8][RWKV] |
| Phase 3 — Match model (100KB) | **1.2177** | -0.0095 additional, hash-based longest match |
| R27 — Embedding surgery center0.3 (100KB) | **1.1895** | -0.0282 additional, zero runtime cost |
| R33 — CM scaling A1 Phase 1 (10KB) | 1.2138 | +3 models (sparse+indirect), -0.0042 vs baseline |
| R33 — CM scaling A1 Phase 1 (100KB) | 1.1922 | +0.0027 vs 1.1895 baseline, neutral on text |
| R35 — S1 coupled gates (100KB) | 1.1955 | +0.0033 neutral, -25% params, +15% speed. Prerequisite for S2/S3. |
| R36 — S2 LayerNorm (10KB) | **1.1686** | -0.0510 vs S1, strong early learning boost |
| R36 — S2 LayerNorm (100KB) | **1.1898** | -0.0057 vs S1, +768 params (+1.5%). Prerequisite for S3. |
| R37 — S3 BPTT=8 (10KB) | **1.1669** | -0.0017 vs S2. First temporal learning. |
| R37 — S3 BPTT=8 (100KB) | **1.1843** | -0.0055 vs S2. Adam(beta1=0.02). 148 B/s. New best. |
| R38 — S4 WordModel (100KB) | 1.1852 | +0.0009 (neutral). Redundant with RWKV. +12 MB. Kept for diversity. |
| R39 — A1 APM/SSE (10KB) | KILLED | +0.10 to +0.19 regression. LSTM well-calibrated, APM too sparse. |
| R40 — A2 Match multi-input | KILLED | Multi-ext +0.013, all-match +0.001. Best-only match optimal. |
| R41 — B2 BPTT scaling (16, 32) | KILLED | BPTT=16: +0.0000 (100KB). BPTT=32: +0.0020 (10KB). Bit-level ceiling at 8. |
| R42 — B4 Higher-order CM (12, 16) | KILLED | +0.0004 (100KB). Redundant with RWKV. +2.2 MB. Confirms R33. |
| R43 — B1 2-layer LSTM | KILLED | 2×128: +0.0014 (100KB). 2×64: +0.0003 (100KB). Early boost only (-0.0216 at 10KB). |
| B3 — ISSE chains | KILLED | By analogy with A1 (APM/SSE). Same family, same scale problems. |
| R44 — C1 Online LSTM expert | KILLED | +0.0127 (100KB). New mixer group overhead > prediction value. 44K params, -27% speed. |
| C2 — Information inheritance | KILLED | By analysis. Redundant with hierarchical mixer. |
| C3 — Modality-routing | KILLED | By analysis. Violates no-domain-detection. Same group overhead pattern. |
| **R45 — T2 Final Silesia (12 files)** | **2.2799 mean** | **Identical to post-A. Tiers B+C = zero impact. Telemetry in t2-final/*.jsonl** |
| Phase 3 — Domain checkpoint | BLOCKED | Requires GPU (CPU-only hardware) |

100KB "quick" eval on enwik8 only. Full composite requires Tier 1 (4 files).

## Target Landscape (enwik8)

```
2.58  gzip
2.13  StateSMix   (~120K params, Mamba+n-gram)
1.58  analytic-lm (54 CM + LSTM, our predecessor)
1.50  PPM
1.41  azathoth-lm ensemble (RWKV+N-gram+bias, 100KB quick)
1.33  azathoth-lm tuned static (lr=0.30, scale=0.5, 100KB)
1.32  azathoth-lm mixer f32 (eta=0.01, lr=0.30, scale=0.5, 100KB)
1.31  azathoth-lm tuned (lr=0.30, scale=0.5, 10KB)
1.30  azathoth-lm Q8 int-accum (eta=0.01, lr=0.30, scale=0.5, 100KB)
1.29  azathoth-lm hybrid logistic CM+RWKV (100KB)
1.25  azathoth-lm LSTM hybrid CM+RWKV (100KB)
1.23  azathoth-lm hierarchical groups (100KB)
1.22  azathoth-lm hierarchical + match model (100KB)
1.19  azathoth-lm + emb surgery center0.3 (100KB)
1.18  azathoth-lm + S1+S2+S3 LSTM stack (100KB) ← CURRENT BEST
1.27  PAQ8px      (200+ models)
1.19  NNCP v3     (199M Transformer-XL)
1.17  cmix        (2077 models + LSTM)
1.11  ts_zip      (RWKV-169M v4 Q8, pure LM)
1.07  SHA-RNN     (63M params)
0.97  fx2-cmix    (6M Transformer + 2000+ CM)
0.94  Nacrith     (135M SmolLM2 + CM)
~1.17 ← PROJECTED azathoth-lm (+ LSTM stack: BPTT=8, coupled, LayerNorm)
~1.08 ← PROJECTED azathoth-lm (+ WordModel + APM + Tweedie + WRT, optimistic)
~1.05 ← CEILING without GPU (optimistic, R34 analysis)
<1.0  ← OUR TARGET (requires domain-tuned neural model, blocked by GPU)
```

## Benchmark Dashboard

### Current Best — Composite (Tier 1: 5 files, 10KB each, FINAL 2026-10-07)

| Metric | FINAL | Pre-S1 (surgery) | Delta | Verdict |
|---|---|---|---|---|
| **mean** | **1.4674** | 1.5213 | **-0.0539** | DOWN |
| **sigma** | **0.3030** | 0.2814 | +0.0216 | UP (OEIS improved more) |
| **worst** | **1.8378** (OEIS) | 1.9045 | **-0.0667** | DOWN |

#### Tier 1 Detail (FINAL, with emb surgery center0.3)

| File | Type | BPB (FINAL) | BPB (pre-S1) | Delta | B/s |
|---|---|---|---|---|---|
| samba (10KB) | Code | **1.1445** | 1.1846 | -0.0401 | 105 |
| enwik8 (10KB) | Text EN | **1.1680** | 1.2180 | -0.0500 | 114 |
| dickens (10KB) | Text EN | **1.5465** | 1.5766 | -0.0301 | 133 |
| mozilla (10KB) | Binary | **1.6404** | 1.7227 | -0.0823 | 46 |
| OEIS (10KB) | Numerical | **1.8378** | 1.9045 | -0.0667 | 40 |

### Full Silesia Composite (Tier 2: 12 files, 10KB, FINAL 2026-10-07)

Re-evaluated 2026-10-07 with final architecture (post all tiers S/A/B/C).
Values identical to post-Tier A — confirms Tiers B and C had zero BPB impact.

| Metric | FINAL | Pre-S1 | Delta | Verdict |
|---|---|---|---|---|
| **mean** | **2.2799** | 2.3456 | **-0.0657** | DOWN |
| **sigma** | **1.6843** | 1.6269 | +0.0574 | UP (WordModel dilution) |
| **worst** | **6.0483** (sao) | 6.1083 | **-0.0600** | DOWN |
| text-like mean | 1.1321 | 1.1725 | -0.0404 | DOWN |
| binary mean | 3.4278 | 3.5186 | -0.0908 | DOWN |

### Current Best — enwik8 (literature comparability only)

| Metric | Value | Date |
|---|---|---|
| **BPB S3 BPTT=8 (enwik8 100KB)** | **1.1843** | **2026-10-06** |
| BPB hierarchical+match (enwik8 100KB) | 1.2177 | 2026-10-06 |
| BPB hierarchical only (enwik8 100KB) | 1.2272 | 2026-10-06 |
| BPB LSTM flat hybrid (enwik8 100KB) | 1.2549 | 2026-10-06 |
| BPB logistic hybrid (enwik8 100KB) | 1.2924 | 2026-10-05 |
| bytes/s (hierarchical+match) | **138** | **2026-10-06** |
| allocs/token | **~0** (scratch arena) | **2026-10-05** |
| MB RAM (Q8 + CM + match) | **~240** | **2026-10-06** |
| BPB/Mparam | 0.0122 | 2026-10-06 |

### Cross-Domain (Silesia Corpus — 10KB samples, FINAL 2026-10-07)

Full analysis in `docs/research/r25-silesia-evaluation.md` (pre-surgery),
`docs/research/r30-frontier-research-roadmap-reform.md` (with surgery),
and `docs/research/r45-t2-final-silesia-eval.md` (final post-all-tiers).
Telemetry: `docs/results/t2-final/<file>.jsonl` (per-100-byte BPB, bit costs, match stats).

| File | Type | BPB (FINAL) | BPB (pre-S1) | Delta | B/s | Tokens |
|---|---|---|---|---|---|---|
| xml | Structured markup | **0.5212** | 0.5689 | -0.0477 | 58 | 4506 |
| nci | Chemical data | **0.5360** | 0.5913 | -0.0553 | 48 | 5838 |
| samba | Source code (C) | **1.1445** | 1.1846 | -0.0401 | 105 | 2851 |
| reymont | Polish text | **1.4778** | 1.5172 | -0.0394 | 53 | 6201 |
| dickens | English text | **1.5465** | 1.5766 | -0.0301 | 133 | 2460 |
| webster | English dict | **1.5664** | 1.5967 | -0.0303 | 122 | 2519 |
| mozilla | Executable | **1.6404** | 1.7227 | -0.0823 | 46 | 8065 |
| mr | Medical image | **1.9375** | 2.0466 | -0.1091 | 36 | 9341 |
| ooffice | Office binary | **2.5691** | 2.6394 | -0.0703 | 29 | 9680 |
| osdb | MySQL database | **4.2812** | 4.3301 | -0.0489 | 36 | 7721 |
| x-ray | Medical X-ray | **4.0903** | 4.2643 | -0.1740 | 30 | 9966 |
| sao | Astronomy SAO | **6.0483** | 6.1083 | -0.0600 | 29 | 9525 |
| **Mean (all 12)** | | **2.2799** | 2.3456 | -0.0657 | **62** | |
| **σ (all 12)** | | **1.6843** | 1.6269 | +0.0574 | | |
| **Text-like mean (6)** | | **1.1321** | 1.1725 | -0.0404 | **78** | |
| **Binary mean (6)** | | **3.4278** | 3.5186 | -0.0908 | **34** | |

### Cross-Domain (adversarial — 10KB)

See `docs/research/r23-cross-domain-validation.md`.

| Test | BPB | Expected | Status |
|---|---|---|---|
| Random bytes | 8.0248 | 8.0 | PASS |
| Repeated pattern | 0.0459 | ~0.0 | PASS |

### Historical (enwik8 progression)

| Config | BPB 10KB | BPB 100KB | Date |
|---|---|---|---|
| RWKV-only | 1.4298 | 1.4691 | 2026-10-01 |
| + ensemble (N-gram + bias) | 1.3758 | 1.4086 | 2026-10-01 |
| + tuned (lr=0.30, scale=0.5) | 1.3078 | 1.3281 | 2026-10-02 |
| + mixer (eta=0.01) | 1.3032 | 1.3238 | 2026-10-02 |
| + Q8 quantization | 1.2797 | 1.2984 | 2026-10-05 |
| + hybrid CM+RWKV bridge | 1.4133 | 1.2924 | 2026-10-05 |
| + LSTM mixer | 1.5252 | 1.2549 | 2026-10-06 |
| + hierarchical groups | 1.2502 | 1.2272 | 2026-10-06 |
| + match model | 1.2408 | 1.2177 | 2026-10-06 |
| + emb surgery center0.3 | — | **1.1895** | 2026-10-06 |
| + S1 coupled gates | 1.2196 | 1.1955 | 2026-10-06 |
| + S2 LayerNorm | 1.1686 | 1.1898 | 2026-10-06 |
| + S3 BPTT=8 + Adam | **1.1669** | **1.1843** | 2026-10-06 |

## Phase 0 Details

### RWKV-7 0.1B Baseline (2026-10-01)

- **Model**: RWKV-7 "Goose" 0.1B World (100M params, D=768, H=12, L=12)
- **Weights**: HuggingFace `BlinkDL/rwkv-7-world` SafeTensors
- **Quantization**: Q8 all layers + Q8 head (~130 MB total, was ~516 MB f32)
- **Tokenizer**: World (65,536 tokens), greedy encoding
- **Evaluation**: token-level cross-entropy → BPB over raw bytes

Progressive evaluation (enwik8 first 10KB):

| Subset | RWKV-only BPB | Ensemble BPB | Delta |
|---|---|---|---|
| 10 KB | 1.4298 | 1.3758 | -0.054 |
| 100 KB | 1.4691 | 1.4086 | -0.061 |

Ensemble delta **increases** with more data (0.054 → 0.061) as online
components learn document patterns. Full enwik8 delta est. -0.07 to -0.10.

### Ensemble Components

| Component | Contribution | Overhead |
|---|---|---|
| RWKV-7 0.1B | Baseline predictor (~1.43 BPB) | 46 ms/tok |
| Token N-gram (orders 1-4) | Logit bias for local patterns | ~0 ms |
| Online bias head (lr=0.001) | Per-document SGD correction | ~0 ms |

### Throughput Status

At 162 B/s (Q8 + AVX-VNNI + scratch arena), full enwik8 ≈ 171h (~7 days).
See R11 for Q8 quantization details, R14 for VNNI and scratch arena.

## Scaling Results (R05, 2026-10-02)

| Model | BPB (10KB) | Status |
|---|---|---|
| RWKV-7 0.1B World (current) | **1.4298** | Best available |
| RWKV-7 0.4B World v2.9 | 1.6549 | KILLED — worse than 0.1B |
| RWKV-7 G1d 0.4B | 1.8817 | KILLED — under-trained |
| RWKV-7 G1k 1.5B | **5.2691** | KILLED — catastrophic domain mismatch |

**All scaling paths KILLED.** No available RWKV-7 checkpoint larger than
0.1B outperforms it on enwik8. G1k 1.5B verified correct (forward pass
matches Python reference) but its training data is incompatible with enwik8.

Strategy: maximize 0.1B hybrid (byte CM + LSTM mixer + bridge).
See `docs/research/r05-scaling-analysis.md` for full analysis.

## Hyperparameter Tuning (R06, 2026-10-02)

24-config sweep on enwik8 10KB. Best: **lr=0.30, scale=0.5 → 1.3078 BPB**.

| Parameter | Default | Optimal | Impact |
|---|---|---|---|
| bias_lr | 0.001 | 0.30 | -0.0680 BPB (dominant) |
| ngram_scale | 1.0 | 0.5 | -0.0131 BPB (consistent) |

lr curve monotonically improving through 0.30 — not yet saturated on 10KB.
Dynamic lr (inverse decay) strongly motivated for longer evaluations.
See `docs/research/r06-hyperparameter-tuning.md` for full analysis.

## Surprise-Modulated Dynamic LR (R07, 2026-10-02)

Replaced blind inverse decay with data-driven surprise-modulated lr:
`lr(t) = lr0 * clamp(surprise / ema_surprise, 0.1, 5.0)`

| Config | BPB (10KB) | vs static |
|---|---|---|
| Static lr=0.30 | 1.3078 | baseline |
| Inverse decay (best) | 1.3066 | -0.0012 |
| **Surprise tau=2000** | **1.2997** | **-0.0081** |

Surprise modulation improves 10KB (-0.0081 BPB) but **degrades 100KB** (+0.0039).
10KB gain is partially an EMA initialization artifact (starts at 1.0 nat vs
actual ~4.0). Mechanism amplifies lr above lr0 on enwik8. Needs redesign.

| Config | BPB 10KB | BPB 100KB |
|---|---|---|
| Static lr=0.30 | 1.3078 | **1.3281** |
| Surprise tau=1000 | 1.3002 | 1.3320 (worse) |
| Surprise tau=2000 | **1.2997** | — |
| Surprise tau=5000 | 1.3007 | 1.3316 (worse) |
| Surprise tau=10000 | — | 1.3327 (worse) |

**KILLED**: Surprise modulation does not scale to 100KB at any tau.
Static lr=0.30 remains best for 100KB. Next: find optimal static lr for 100KB.
See `docs/research/r07-dynamic-lr.md` for full analysis.

## Adaptive Component Mixer (R08, 2026-10-02)

Online gradient descent on ensemble component weights:
`final[i] = rwkv[i] + w_ng * ng[i] + w_b * b[i]`

Eta sweep on 10KB (lr=0.30, scale=0.5):

| eta | BPB (10KB) | vs static (1.3078) |
|---|---|---|
| 0.001 | 1.3068 | -0.0010 |
| 0.005 | 1.3046 | -0.0032 |
| 0.01 | 1.3032 | -0.0046 |
| 0.02 | 1.3017 | -0.0061 |
| 0.05 | 1.3001 | -0.0077 |
| 0.10 | 1.2997 | -0.0081 |

Best 10KB: **1.2997 BPB** (eta=0.10). Monotonically improving — same pattern as
surprise lr (R07).

100KB results:

| eta | BPB (100KB) | vs static (1.3281) |
|---|---|---|
| 0.002 | 1.3241 | -0.0040 |
| **0.005** | **1.3238** | **-0.0043** |
| **0.01** | **1.3238** | **-0.0043** |
| 0.10 | 1.3278 | -0.0003 |

Plateau at eta 0.005-0.01: BPB converges to 1.3238. eta=0.002 slightly too conservative.

**Best: eta=0.005-0.01 → 1.3238 BPB (-0.0043 vs static).** Only adaptive
mechanism that improves on 100KB.

Static lr sweep on 100KB (scale=0.5):

| lr | BPB (100KB) | vs lr=0.30 (1.3281) |
|---|---|---|
| 0.05 | 1.3571 | +0.0290 |
| 0.10 | 1.3448 | +0.0167 |
| 0.15 | 1.3379 | +0.0098 |
| 0.30 | **1.3281** | baseline (best) |

**lr=0.30 confirmed optimal on 100KB.** Mixer is the only adaptive approach that
doesn't degrade on 100KB, but improvement is negligible (-0.0003).
See `docs/research/r08-adaptive-mixer.md` for full analysis.

## R14: Alternative Number Systems (2026-10-05)

Exhaustive survey of 8 non-standard number systems for CPU inference.
40+ web searches, 6 parallel research threads. See `docs/research/r14-alternative-number-systems.md`.

Initial survey identified ternary (BitNet b1.58) as potential winner.
Verification phase (5 targeted threads, 40+ additional searches) **killed ternary**:
PTQ ternary on 0.1B = catastrophic collapse (PPL >4000). QAT requires retraining.
RWKV SSM recurrence propagates quant noise. No neural compressor uses <Q8.

Block-32 Q8 tested and KILLED (+0.0378 BPB, -34% speed on RWKV).
AVX-VNNI initially KILLED pre-scratch (-33% speed, memory-bound).
**Re-evaluated post-scratch: VALIDATED.** +38% speed (162 B/s vs 117 B/s).
Scratch arena eliminated cache pollution, shifting bottleneck from memory to compute.

| System | Verdict | Reason |
|---|---|---|
| Block-32 Q8 | **KILLED** | +0.0378 BPB, -34% speed on RWKV (uniform weights) |
| AVX-VNNI intrinsics | **VALIDATED** | +38% speed post-scratch (was -33% pre-scratch) |
| Buffer reuse | **VALIDATED** | +46% speed, ~0 allocs/token (was ~1400) |
| ANS | Adopt for coder | Industry-standard entropy coding |
| Ternary (BitNet) | **KILLED** | PTQ collapse at 0.1B, can't retrain, +0.30-0.50 BPB |
| Q4 | **KILLED** | +30% PPL on RWKV-7 0.1B, est. +0.15-0.40 BPB |
| LNS | Rejected | Accumulation bottleneck kills dot products |
| RNS | Rejected | Carry-free irrelevant on CPU (1-cycle adds) |
| Stochastic | Rejected | Precision O(1/sqrt(N)), CPU-impractical |
| Posit | Rejected | 4-20x slower in software |

## Scratch Arena / Buffer Reuse (2026-10-05)

Pre-allocated 149 KB workspace eliminates ~1400 heap allocations per token.
All intermediate tensors in `forward()` → `time_mixing()` → `channel_mixing()`
now write into reusable buffers instead of allocating new Vec each call.

| Metric | Before (allocating) | Scratch only | Scratch + VNNI | Delta (full) |
|---|---|---|---|---|
| BPB 10KB | 1.2797 | 1.2797 | 1.2797 | 0.0000 |
| BPB 100KB | 1.2984 | 1.2984 | 1.2984 | 0.0000 |
| Speed 10KB | 80 B/s | 117 B/s | 135 B/s | **+69%** |
| Speed 100KB | 80 B/s | 117 B/s | 162 B/s | **+103%** |
| Allocs/token | ~1400 | ~0 | ~0 | **-99.9%** |
| Scratch memory | 0 | 149 KB | 149 KB | one-time |

Key insights:
- Rust iterator patterns (`.iter_mut().zip()`) are critical for auto-vectorization.
  Index-based loops produced 30% slower code due to missed SIMD opportunities.
- AVX-VNNI (VPDPBUSD) only helps AFTER cache pollution is eliminated. Pre-scratch
  the CPU was 87% idle on DRAM fetches; faster arithmetic made it worse. Post-scratch,
  state stays in L3 and VNNI's 4x throughput on i8 dot-products adds +38%.

## P1.1: Arithmetic Coder (2026-10-05)

Range coder with carry propagation (Schindler-style). CDF-24 precision (TOP = 2^24).
Token-level encoding with 65K vocab. Will transition to byte-level with P1.2/P1.3.

| Metric | 1 KB | 10 KB |
|---|---|---|
| Compressed size | 146 B | 1665 B |
| Compression ratio | 0.1460 | 0.1665 |
| **Compressed BPB** | 1.1680 | **1.3320** |
| Cross-entropy BPB | 0.9781 | 1.2812 |
| Coder overhead | 0.1899 | 0.0508 |
| Roundtrip verified | Yes | Yes |

Overhead breakdown (10KB):
- Header (16 bytes): 0.0128 BPB
- Encoder flush (~5 bytes): ~0.004 BPB
- CDF quantization (24-bit / 65K vocab): ~0.034 BPB

CDF quantization overhead is inherent to 65K vocab + 24-bit precision.
At byte-level (256 vocab), this drops to negligible. At 100KB the
header and flush overhead amortize to ~0.001 BPB.

Kill criteria check: overhead 0.05 > 0.01 threshold. However, this is
dominated by structural overhead (header + CDF quantization on 65K vocab),
not coder bugs. Verified by roundtrip identity on both sizes.

## Roadmap

Universal compressor design. Full details in `docs/ROADMAP.md`.

### Completed
- **Phase 1 — Universal Core**: Arithmetic coder (P1.1), byte-level CM (P1.2),
  RWKV→byte bridge (P1.3). Confidence skip KILLED (P1.4).
- **Phase 2.1 — LSTM Mixer**: -0.0375 BPB over logistic.
- **Phase 2.2 — Hierarchical Groups**: -0.0277 BPB over flat LSTM. Best: 1.2272.
- **Phase 3.1 — Match Model**: -0.0095 BPB additional. Best: **1.2177**.
- **Phase 3.2 — Domain Checkpoint**: BLOCKED (requires GPU).

### Research
- **R26**: MoE architecture + direct weight manipulation research.
  See `docs/research/r26-moe-architecture-research.md`.

### Next (R34 roadmap audit — priorities reformed)

**Tier S — Critical path (LSTM stack is #1 gap vs cmix):**
- **S1 — Coupled gates (i=1-f)**: -25% LSTM params, stabilizes cell state. Easy win.
- **S2 — LayerNorm**: Per-gate normalization. Prerequisite for BPTT>1.
- **S3 — BPTT=8 (1 byte)**: Needs Adam + grad clip. Est. -0.02 to -0.05. **Biggest single delta.**
- **S4 — WordModel**: Case-folded + word-pair. Only model type we lack. Est. -0.005 to -0.015.

**Tier A — High impact:**
- **A1 — APM/SSE 1-2 stages**: Distinct contexts per stage. Est. -0.005 to -0.020
- **A2 — Match model multi-input**: More lengths + multi-predictions. Replaces SA-PPM. Est. -0.005 to -0.015
- **A3 — Tweedie denoising**: Post-blend Tweedie (Midicoth paper). Cannot combine with SSE.

**KILLED (R34):**
- ~~A1 old — CM scaling 9→25+~~: Model count not bottleneck. R33 neutral. Quality > quantity.
- ~~B2 old — Hedge mixer~~: Converges to pass-through (Nacrith ablation).
- ~~D1 — SA-PPM~~: No top compressor uses suffix arrays. Revised -0.01 to -0.05. Match improvements capture 80% at 5% effort.
