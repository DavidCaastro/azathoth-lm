# Roadmap — azathoth-lm

**Date**: 2026-10-06
**Current best (enwik8)**: 1.1895 BPB (100KB enwik8, literature ref only)
**Current composite (T1)**: mean=1.5213 | sigma=0.2814 | worst=1.9045 (with surgery, 5 files)
**Target**: < 1.0 BPB enwik8 + sigma decreasing — universal compressor
**Primary metric**: Composite BPB (mean, sigma, worst) — see R28

## Design Philosophy

azathoth-lm is a **universal data compressor**, not an enwik8 optimizer.
enwik8 is the primary benchmark because the literature uses it, but every
architectural decision must work on arbitrary byte streams: text, binaries,
images, audio, genomic data, mixed formats.

The systems that achieve sub-1.0 BPB (cmix, PAQ8px, fx2-cmix, Nacrith) are
all universal compressors. They operate at byte/bit level. Their enwik8
numbers are a consequence of being good at compressing *anything*.

Principles:
- **Byte/bit-level first**: all context models operate on raw bytes, not tokens
- **Data-agnostic**: no assumptions about input format or language
- **Neural + statistical**: RWKV for generalization, CM for exact pattern matching
- **Online adaptation**: all non-neural components learn during inference
- **Multi-corpus validation**: measure on enwik8 + at least one non-text corpus

## Current Position

```
1.19  azathoth-lm   (0.1B RWKV + 9 CM + match + hier LSTM + emb surgery, ~115 B/s)
1.27  PAQ8px        (200+ byte-level CM, universal)
1.19  NNCP v3       (199M Transformer-XL)
1.17  cmix          (2077 byte-level models + LSTM mixer, universal)
1.11  ts_zip        (RWKV-169M v4 Q8, pure LM)
0.97  fx2-cmix      (6M Transformer + 2000+ CM, universal)
0.94  Nacrith       (135M SmolLM2 + byte-level CM, universal)
```

Gap to target: ~0.22 BPB. **Below PAQ8px** at 100KB (-0.05 BPB).

### What we have vs what we need

| Component | azathoth-lm (current) | PAQ8px/cmix/Nacrith |
|---|---|---|
| Neural predictor | RWKV-7 0.1B Q8 (token→byte bridge) | Transformer/LM (byte-level bridge) |
| Context models | 9 byte-level CM + longest-match (4-128 bytes) | 50-2000+ (byte/bit-level) |
| Mixer | Hierarchical: logistic sub-mixers + LSTM top (67K params) | LSTM / logistic multi-layer |
| Entropy coder | Range coder CDF-24 (roundtrip verified) | Full arithmetic coder |
| Operating level | Byte/bit (MSB-first decomposition) | Byte/bit |
| Adaptation | Online SGD on all components | Online SGD on all components |
| Match predictor | Hash-based longest match (6 context lengths, 32 MB) | Suffix array / LZ |

The architecture is now **feature-complete**: neural + statistical CM +
match model + hierarchical LSTM mixer + arithmetic coder. Remaining gap is
**scale** (9+match vs 200+ models) and **full SA-PPM** (suffix array for
optimal variable-length matching).

## Phase 1 — Universal Compressor Core

Priority: build the missing infrastructure that makes azathoth-lm a real
compressor, not just a cross-entropy measurer.

### P1.1: Arithmetic Coder (CDF-24)

- **Impact**: est. -0.05 to -0.10 BPB (closes probability→bits gap)
- **Effort**: Medium
- **Risk**: Low — well-understood algorithm, standard in all top compressors
- **Rationale**: Without an arithmetic coder, we don't compress — we measure.
  The gap between cross-entropy BPB and actual compressed BPB is 0.05-0.10
  depending on tail handling. CDF-24 (24-bit precision) is the industry standard.
  This is domain-agnostic: works on any byte stream.
- **Heritage**: No prior attempt. Clean implementation.
- **Kill criteria**: If compressed BPB > cross-entropy BPB + 0.01, bug in coder.

### P1.2: Byte-level Context Mixing Infrastructure

- **Impact**: est. -0.05 to -0.15 BPB (the missing architectural half)
- **Effort**: High
- **Risk**: Medium — hash table memory, collision management
- **Rationale**: This is the single biggest architectural gap. Every sub-1.0
  system has byte-level context models. Our current token-level N-gram only
  works for text with the World tokenizer. Byte-level CM works on anything.

  Core components to port from analytic-lm heritage:
  1. **Byte-level match models** (orders 1-8): hash tables keyed by recent
     byte context, predict next byte distribution. Work on any data.
  2. **Bit-level decomposition**: predict 8 bits per byte (MSB first).
     heritage.md confirms bit-level > byte-level (1.58 vs 1.645 BPB).
  3. **Recency-weighted hash tables**: 4-way associative, decay=0.90.
  4. **Logistic mixing**: stretch predictions to logit space, mix additively.

  These operate in parallel with RWKV. RWKV predicts at token granularity;
  CM predicts at byte/bit granularity. A bridge layer converts RWKV token
  logits to byte-level probabilities for mixing.
- **Heritage**: analytic-lm achieved 1.58 BPB with 54 CM + LSTM on enwik8.
  Top ~20 models by contribution are the porting target. Diminishing returns
  after ~50 models.
- **Kill criteria**: If 10 byte-level CM models add < 0.01 BPB on 100KB, stop.

### P1.3: RWKV-to-Byte Bridge

- **Impact**: Enables mixing neural (token) + statistical (byte) predictions
- **Effort**: Medium
- **Risk**: Low-Medium — token/byte alignment is the main complexity
- **Rationale**: RWKV outputs token-level logits (65K vocab). CM operates at
  byte/bit level. To mix them, we need a bridge that converts RWKV's token
  probability distribution into a byte-level prediction.

  Approach: for each byte position, marginalize over all tokens that could
  emit that byte at that position. Cache the token→byte mapping at load time.
  This is data-agnostic (World tokenizer covers all byte values).
- **Heritage**: Nacrith does this (SmolLM2 token logits → byte mixer).
- **Kill criteria**: If bridge latency > 5ms/byte, too slow.

### P1.4: Confidence Skip

- **Impact**: 2-5x throughput, est. ~0 BPB loss
- **Effort**: Medium
- **Risk**: Medium — aggressive thresholds degrade BPB
- **Rationale**: When RWKV confidence is very high (top-1 prob > threshold),
  CM corrections are negligible. Skip ensemble computation for those tokens.
  Universal optimization — works regardless of data type.
- **Heritage**: Listed in heritage.md Tier 2 as untried. No known failure.
- **Kill criteria**: If BPB increases > 0.005 at any threshold, too aggressive.

## Phase 2 — Advanced Mixing

Priority: replace linear mixing with temporal mixing that captures
cross-model dependencies.

### P2.1: LSTM Mixer

- **Impact**: est. -0.05 to -0.22 BPB (analytic-lm's biggest single win)
- **Effort**: High
- **Risk**: Medium — BPTT>1 during eval overfits (heritage.md)
- **Rationale**: heritage.md documents LSTM mixing as +0.22 BPB over linear.
  This is the largest known architectural win. An LSTM can capture temporal
  patterns in how models' relative accuracy shifts over the byte stream.
  Benefits most with multiple CM models (P1.2) to mix.
- **Heritage**: BPTT>1 during eval = FAIL. Use BPTT=1. HID=128+. SGD > Adam.
- **Kill criteria**: If LSTM mixer BPB > current mixer BPB on 100KB, stop.

### P2.2: Hierarchical Model Groups

- **Impact**: est. -0.02 to -0.05 BPB
- **Effort**: Medium-High
- **Risk**: Medium — SSE overcorrection documented in heritage.md
- **Rationale**: Group models by type (exact match, neural, statistical),
  mix within groups, then mix group outputs. Reduces interaction noise.
  Standard in cmix (3-layer mixer hierarchy).
- **Heritage**: Cascaded SSE always overcorrects. Hierarchical by type untried.

### P2.3: Cross-Domain Validation Suite

- **Impact**: Keeps system honest — prevents domain overfitting
- **Effort**: Low-Medium (data sourcing + automation)
- **Risk**: None
- **Rationale**: Full benchmark protocol defined in `docs/BENCHMARKS.md`.
  11 categories across 9 data types: text (EN + non-EN), source code,
  structured data, executables, scientific, multimedia raw, mixed archives,
  pre-compressed. σ (cross-domain variance) and worst-domain BPB are
  first-class metrics alongside mean BPB.
  Compare against gzip, zstd-19, brotli-11, PAQ8px, lzma2.
  Includes adversarial tests: random, repeated, domain-switch, pre-compressed.
- **Anti-gaming**: No domain detection. No corpus-specific hyperparameters.
  All adaptation must be data-driven and online.
- **Kill criteria**: If σ increases while mean BPB improves, we're overfitting.

## Phase 3 — Frontier Techniques

Priority: high-complexity techniques for pushing toward <1.0 BPB.

### P3.1: SA-PPM / Suffix Array Predictor

- **Impact**: est. -0.10 to -0.30 BPB
- **Effort**: Very High
- **Risk**: High — O(n) construction, 400-800 MB for 100MB input
- **Rationale**: Suffix array unifies all context matching into one optimal
  structure. PPM alone achieves ~1.50 BPB. With neural predictions, could be
  the single largest remaining win. Universal — works on any byte stream.
- **Heritage**: Tier 1 in heritage.md. Never attempted.
- **Dependencies**: RAM budget (~400-800 MB for enwik8).

### P3.2: Domain-Matched Neural Checkpoint

- **Impact**: est. -0.10 to -0.20 BPB
- **Effort**: Very High (needs GPU)
- **Risk**: High — GPU access, domain-specific = less universal
- **Rationale**: Fine-tuning RWKV on target domain improves base predictions.
  ts_zip achieves 1.11 BPB with RWKV-169M. But this is domain-specific,
  contradicting universality. Useful as an optional mode, not core design.
- **Kill criteria**: Must be opt-in. Default mode must work without fine-tuning.

## Priority Matrix

| # | Action | Est. Delta BPB | Actual | Status |
|---|---|---|---|---|
| P1.1 | Arithmetic coder | -0.05 to -0.10 | overhead 0.0508 (10KB) | **DONE** |
| P1.2 | Byte-level CM | -0.05 to -0.15 | 2.09 BPB standalone | **DONE** |
| P1.3 | RWKV→byte bridge | enables mixing | -0.0060 BPB hybrid | **DONE** |
| P1.4 | Confidence skip | 2-5x speed | <3% speed gain | **KILLED** |
| P2.1 | LSTM mixer | -0.05 to -0.22 | **-0.0375 BPB** | **DONE** |
| P2.2 | Hierarchical groups | -0.02 to -0.05 | **-0.0277 BPB** | **DONE** |
| P2.3 | Multi-corpus validation | honesty check | σ=1.72 (Silesia 12 files) | **DONE** |
| R28 | Composite BPB as primary metric | structural de-bias | (mean,σ,worst) replaces enwik8-only | **DONE** |
| R29 | Numerical regime (OEIS) | adds 5th Tier 1 file | OEIS 1.9045 BPB, distinct regime confirmed | **DONE** |
| P3.1 | Match model (simplified SA-PPM) | -0.10 to -0.30 | **-0.0095 BPB** | **DONE** |
| P3.2 | Domain checkpoint | -0.10 to -0.20 | — | **BLOCKED** (GPU) |

## Projected Trajectory

### Projections vs Actuals (Phase 1-2)

```
Projected                           Actual
─────────                           ──────
1.2984  baseline                    1.2984  baseline
1.24    + arith coder (-0.05)       n/a     (coder adds overhead, not BPB gain)
1.12    + byte CM (-0.12)           2.09    standalone (not additive — CM feeds mixer)
1.10    + bridge (-0.02)            1.2924  hybrid logistic (-0.0060)
 n/a    + confidence skip           KILLED  (<3% speed, RWKV=97% compute)
0.93    + LSTM mixer (-0.17)        1.2549  LSTM hybrid (-0.0375 vs logistic)
```

### Key Lessons

- **Cumulative gains don't add linearly.** Each component's contribution
  depends on what's already in the stack. CM standalone = 2.09, but its
  value is in providing diverse inputs for the LSTM mixer.
- **LSTM gain (-0.0375) less than heritage predicted (-0.22)** because we
  have 10 models (9 CM + 1 RWKV) vs analytic-lm's 54. Still improving at 100KB.
- **Confidence skip was architectural dead end** — RWKV sequential state
  prevents skipping the dominant compute cost.

### Phase 2-3 Actuals

```
1.2549  LSTM flat hybrid
1.2272  + hierarchical grouping (-0.0277)
1.2177  + match model (-0.0095)
1.1895  + embedding surgery center0.3 (-0.0282)
```

### Remaining Trajectory (from 1.1895)

Optimistic:
```
1.1895  current (100KB enwik8, hierarchical + match + emb surgery)
1.16    + ICM models for bits 3-5 (-0.03)
1.15    + APM post-mixer + match upgrade (-0.01)
1.12    + full enwik8 (more match history, LSTM convergence)
1.00    + full SA-PPM with suffix array (-0.12)
```

Conservative:
```
1.1895  current
1.17    + ICM + APM (-0.02)
1.15    + full enwik8 (-0.02)
```

The < 1.0 target now within ~0.19 BPB. Full SA-PPM remains the
highest-impact remaining technique.
P3.2 (domain checkpoint) is blocked by GPU hardware.

## Completed

| Phase | Result | Date |
|---|---|---|
| RWKV-7 0.1B integration | 1.4691 BPB baseline | 2026-10-01 |
| Token ensemble (N-gram + bias) | 1.4086 BPB (-0.0605) | 2026-10-01 |
| Hyperparameter tuning | 1.3078 BPB (10KB) | 2026-10-02 |
| AdaptiveMixer | 1.3238 BPB (100KB, -0.0043) | 2026-10-02 |
| Q8 quantization (all layers) | 1.2984 BPB (-0.0254), -75% RAM | 2026-10-05 |
| Scratch arena (buffer reuse) | +46% speed (80→117 B/s), ~0 allocs | 2026-10-05 |
| AVX-VNNI post-scratch | +38% speed (117→162 B/s) | 2026-10-05 |
| P1.1: Arithmetic coder (CDF-24) | 1.3320 compressed BPB (10KB), roundtrip OK | 2026-10-05 |
| P1.2: Byte-level CM (standalone) | 2.09 BPB (1MB), 9 models, 78 MB, 220K B/s | 2026-10-05 |
| P1.3: RWKV→byte bridge + hybrid | **1.2924 BPB** (100KB), -0.0060 vs baseline, 169 B/s | 2026-10-05 |
| P1.4: Confidence skip | KILLED — <3% speed gain, RWKV dominates compute | 2026-10-05 |
| P2.1: LSTM mixer | **1.2549 BPB** (100KB), -0.0375 vs logistic, 134 B/s | 2026-10-06 |
| P2.2: Hierarchical model groups | **1.2272 BPB** (100KB), -0.0277 vs flat LSTM, 138 B/s | 2026-10-06 |
| P3.1: Match model (simplified SA-PPM) | **1.2177 BPB** (100KB), -0.0095 additional, 138 B/s | 2026-10-06 |
| P2.3: Cross-domain (initial) | 5 domains tested, σ=0.964, adversarial PASS | 2026-10-06 |
| P2.3: Silesia Corpus eval | 12 files, mean 2.29 BPB, σ=1.72, Weissman=4.9 | 2026-10-06 |
| R27: Embedding surgery center0.3 | **1.1895 BPB** (100KB), -0.0282 additional, zero cost | 2026-10-06 |
| P3.2: Domain checkpoint | BLOCKED — requires GPU (i5-1235U CPU only) | 2026-10-06 |
| R28: Composite BPB metric | Primary metric = (mean, σ, worst), not enwik8-only | 2026-10-06 |
| R29: OEIS numerical regime | Tier 1 = 5 files (4 regimes), OEIS 1.9045 BPB | 2026-10-06 |

## Constraints

- **CPU-only**: i5-1235U (Alder Lake), 12 threads, 32 GB DDR5, no GPU
- **RAM budget**: ~16 GB for inference (RWKV ~130 MB Q8, CM hash tables est. ~6-8 GB)
- **Throughput**: 138 B/s hierarchical+match (was 134 B/s flat LSTM), full enwik8 ≈ 201h (~8.4 days)
- **Max 1 heavy task**: concurrent evaluations cause CPU thrashing
- **Zero external deps**: all code must compile with rustc + stdlib only
