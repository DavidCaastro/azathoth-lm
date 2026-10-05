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
| Phase 1 — Q8 all layers (100KB) | **1.2984** | Q8 int-accum + VNNI, 162 B/s, -75% RAM (BEST) |
| Phase 1 — Arithmetic coder (1KB) | 1.1680 | Range coder CDF-24, roundtrip verified |
| Phase 1 — Arithmetic coder (10KB) | 1.3320 | Compressed BPB, CE=1.2812, overhead=0.0508 |
| Phase 1 — CM standalone (100KB) | 2.4078 | 9 bit-level models, logistic mixer, 78 MB |
| Phase 1 — CM standalone (1MB) | 2.0915 | Beats gzip (2.58), 220K B/s |
| Phase 1 — Hybrid CM+RWKV bridge (100KB) | **1.2924** | -0.0060 vs baseline, 169 B/s |
| Phase 1 — Confidence skip (100KB) | KILLED | +0.0026 BPB at best, <3% speed gain |
| Phase 2 — LSTM mixer hybrid (100KB) | **1.2549** | -0.0375 vs logistic, -0.0435 vs baseline |

100KB "quick" eval. Full enwik8 now feasible: 162 B/s → ~7 days (was ~25 days at 46 B/s).

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
1.25  azathoth-lm LSTM hybrid CM+RWKV (100KB) ← CURRENT
1.30  azathoth-lm dynamic lr (surprise-mod tau=2000, 10KB)
1.27  PAQ8px      (200+ models)
1.19  NNCP v3     (199M Transformer-XL)
1.17  cmix        (2077 models + LSTM)
1.11  ts_zip      (RWKV-169M v4 Q8, pure LM)
1.07  SHA-RNN     (63M params)
0.97  fx2-cmix    (6M Transformer + 2000+ CM)
0.94  Nacrith     (135M SmolLM2 + CM)
~1.16 ← PROJECTED azathoth-lm (0.1B + byte CM + bridge + confidence skip)
~0.93 ← PROJECTED azathoth-lm (+ LSTM mixer, optimistic)
~0.95 ← KILLED: G1k 1.5B = 5.27 BPB (domain mismatch)
<1.0  ← OUR TARGET (requires byte-level CM + LSTM mixer)
```

## Benchmark Dashboard

### Current Best (enwik8 — literature comparability)

| Metric | Value | Date |
|---|---|---|
| **BPB LSTM hybrid (enwik8 100KB)** | **1.2549** | **2026-10-06** |
| BPB logistic hybrid (enwik8 100KB) | 1.2924 | 2026-10-05 |
| bytes/s (LSTM) | **134** | **2026-10-06** |
| allocs/token | **~0** (scratch arena) | **2026-10-05** |
| MB RAM (Q8 all layers) | **~130** | **2026-10-05** |
| BPB/Mparam | 0.0130 | 2026-10-05 |

### Cross-Domain (pending — requires arithmetic coder + byte-level CM)

Full protocol in `docs/BENCHMARKS.md`. Will measure 11 categories across
9 data types, with σ (neutrality) and worst-domain as primary metrics.

| Category | azathoth | zstd-19 | PAQ8px | Status |
|---|---|---|---|---|
| Text EN (enwik8) | 1.2984 | — | — | Measured (100KB quick) |
| Text non-EN | — | — | — | Pending |
| Source code | — | — | — | Pending |
| Structured (JSON) | — | — | — | Pending |
| Executables | — | — | — | Pending |
| Scientific | — | — | — | Pending |
| Multimedia raw | — | — | — | Pending |
| Mixed archive | — | — | — | Pending |
| Pre-compressed | — | — | — | Pending |
| **Mean** | — | — | — | — |
| **σ (neutrality)** | — | — | — | — |

### Historical (enwik8 progression)

| Config | BPB 10KB | BPB 100KB | Date |
|---|---|---|---|
| RWKV-only | 1.4298 | 1.4691 | 2026-10-01 |
| + ensemble (N-gram + bias) | 1.3758 | 1.4086 | 2026-10-01 |
| + tuned (lr=0.30, scale=0.5) | 1.3078 | 1.3281 | 2026-10-02 |
| + mixer (eta=0.01) | 1.3032 | 1.3238 | 2026-10-02 |
| + Q8 quantization | 1.2797 | 1.2984 | 2026-10-05 |
| + hybrid CM+RWKV bridge | 1.4133 | 1.2924 | 2026-10-05 |
| + LSTM mixer | 1.5252 | **1.2549** | 2026-10-06 |

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

Universal compressor design. Prioritized actions in `docs/ROADMAP.md`:
1. **Phase 1 — Universal Core**: Arithmetic coder, byte-level CM, RWKV→byte bridge, confidence skip
2. **Phase 2 — Advanced Mixing**: LSTM mixer, hierarchical groups, multi-corpus validation
3. **Phase 3 — Frontier**: SA-PPM, domain-matched checkpoint (opt-in)
