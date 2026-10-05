# Results Index

## Quick Reference

| Phase | BPB | Key Finding |
|---|---|---|
| (inherited) analytic-lm | 1.5826 | 54 CM + LSTM, enwik8 80/20 |
| (inherited) edge-lm | ~2.16 | WHT + multi-scale, own corpus |
| Phase 0 — RWKV-7 only | 1.4691 | 0.1B f32+Q8head, 100KB |
| Phase 0 — ensemble | 1.4086 | RWKV + N-gram(4) + bias head, 100KB |
| Phase 0 — tuned ensemble | 1.3078 | lr=0.30, scale=0.5, 10KB |
| Phase 0 — dynamic lr | 1.2997 | surprise-modulated tau=2000, 10KB (KILLED) |
| Phase 0 — adaptive mixer (10KB) | 1.2997 | learned weights eta=0.10, 10KB |
| Phase 0 — adaptive mixer (100KB) | 1.3238 | learned weights eta=0.01, 100KB |
| Phase 1 — Q8 all layers (100KB) | **1.2984** | Q8 int-accum, 117 B/s, -75% RAM (BEST) |

100KB "quick" eval. Full enwik8 now feasible: 117 B/s → ~10 days (was ~25 days at 46 B/s).

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
1.30  azathoth-lm Q8 int-accum (eta=0.01, lr=0.30, scale=0.5, 100KB) ← CURRENT
1.30  azathoth-lm dynamic lr (surprise-mod tau=2000, 10KB)
1.27  PAQ8px      (200+ models)
1.19  NNCP v3     (199M Transformer-XL)
1.17  cmix        (2077 models + LSTM)
1.11  ts_zip      (RWKV-169M v4 Q8, pure LM)
1.07  SHA-RNN     (63M params)
0.97  fx2-cmix    (6M Transformer + 2000+ CM)
0.94  Nacrith     (135M SmolLM2 + CM)
~1.25 ← PROJECTED azathoth-lm (0.1B + tuned ensemble + skip)
~0.95 ← KILLED: G1k 1.5B = 5.27 BPB (domain mismatch)
<1.0  ← OUR TARGET (requires domain-matched checkpoint)
```

## Benchmark Dashboard

| Metric | Value | Date |
|---|---|---|
| BPB ensemble (enwik8 100KB) | 1.4086 | 2026-10-01 |
| BPB ensemble (enwik8 10KB) | 1.3758 | 2026-10-01 |
| BPB tuned ensemble (enwik8 10KB) | 1.3078 | 2026-10-02 |
| BPB dynamic lr (enwik8 10KB) | 1.2997 | 2026-10-02 |
| BPB tuned static (enwik8 100KB) | 1.3281 | 2026-10-02 |
| BPB surprise tau=1000 (enwik8 100KB) | 1.3320 | 2026-10-02 (KILLED) |
| BPB mixer eta=0.10 (enwik8 10KB) | 1.2997 | 2026-10-02 |
| BPB mixer eta=0.10 (enwik8 100KB) | 1.3278 | 2026-10-02 |
| BPB mixer eta=0.01 (enwik8 100KB) | 1.3238 | 2026-10-02 |
| **BPB Q8 int-accum (enwik8 100KB)** | **1.2984** | **2026-10-05** |
| BPB Q8 int-accum (enwik8 10KB) | 1.2797 | 2026-10-05 |
| BPB RWKV-only (enwik8 10KB) | 1.4298 | 2026-10-01 |
| bytes/s | **117** | **2026-10-05** |
| allocs/token | **~0** (scratch arena) | **2026-10-05** |
| MB RAM (Q8 all layers) | **~130** | **2026-10-05** |
| ms/tok | ~34 | 2026-10-05 |
| BPB/Mparam | 0.0130 | 2026-10-05 |
| ARC-C | — | — |
| HellaSwag | — | — |
| MMLU | — | — |
| Winogrande | — | — |

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

At 117 B/s (Q8 all layers, int-accum kernel), full enwik8 ≈ 237h (~10 days).
See R11 for Q8 quantization details, R03/R04 for further optimization.

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

Strategy: maximize 0.1B ensemble (confidence skip, tuned N-gram, CDF-24).
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
AVX-VNNI implemented and KILLED (-33% speed, pipeline is memory-bound not compute-bound).
**Real path: buffer reuse / arena allocator + per-row Q8** (address actual bottleneck).

| System | Verdict | Reason |
|---|---|---|
| Block-32 Q8 | **KILLED** | +0.0378 BPB, -34% speed on RWKV (uniform weights) |
| AVX-VNNI intrinsics | **KILLED** | -33% speed; memory-bound, not compute-bound |
| Buffer reuse | **High priority** | 370 allocs/token pollute cache, 13% bandwidth util |
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

| Metric | Before (allocating) | After (scratch) | Delta |
|---|---|---|---|
| BPB 10KB | 1.2797 | 1.2797 | 0.0000 (identical) |
| BPB 100KB | 1.2984 | 1.2984 | 0.0000 (identical) |
| Speed 10KB (same-session) | 80 B/s | 117 B/s | **+46%** |
| Allocs/token | ~1400 | ~0 | **-99.9%** |
| Scratch memory | 0 | 149 KB | one-time |

Key insight: Rust iterator patterns (`.iter_mut().zip()`) are critical for
auto-vectorization. Index-based loops (`for i in 0..n { out[i] = ... }`) produced
30% slower code due to missed SIMD opportunities.

## Roadmap

Prioritized next actions in `docs/ROADMAP.md`. Key phases:
1. **Phase 1**: CDF-24, N-gram 5-6, confidence skip, full enwik8
2. **Phase 2**: Context mixing models, LSTM mixer, byte-level path
3. **Phase 3**: Domain-matched checkpoint, SA-PPM, hierarchical mixer
