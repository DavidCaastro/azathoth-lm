# Changelog — azathoth-lm

Chronological record of completed phases, projections vs actuals,
and architectural lessons learned. Extracted from ROADMAP.md to keep
the roadmap focused on present state and future direction.

For current state and next steps, see `ROADMAP.md`.
For metrics dashboard, see `docs/results/INDEX.md`.
For individual experiments, see `docs/research/rNN-*.md`.

---

## Phase 0 — Foundation (2026-10-01 to 2026-10-02)

### RWKV-7 0.1B Integration (2026-10-01)

- **Result**: 1.4691 BPB baseline (100KB enwik8)
- **Model**: RWKV-7 "Goose" 0.1B World (100M params, D=768, H=12, L=12)
- **Weights**: HuggingFace `BlinkDL/rwkv-7-world` SafeTensors
- **Tokenizer**: World (65,536 tokens), greedy encoding

### Token Ensemble (2026-10-01)

- **Result**: 1.4086 BPB (-0.0605)
- **Components**: RWKV + N-gram(orders 1-4) + online bias head
- Delta increases with data (0.054→0.061) as online components learn.

### Hyperparameter Tuning (2026-10-02)

- **Result**: 1.3078 BPB (10KB)
- 24-config sweep. Best: lr=0.30, ngram_scale=0.5.
- lr curve monotonically improving through 0.30 — not saturated.
- See `docs/research/r06-hyperparameter-tuning.md`.

### Surprise-Modulated LR (2026-10-02) — KILLED

- **Result**: -0.0081 on 10KB but +0.0039 on 100KB at all tau values.
- 10KB gain is EMA initialization artifact (starts 1.0 vs actual ~4.0).
- See `docs/research/r07-dynamic-lr.md`.

### AdaptiveMixer (2026-10-02)

- **Result**: 1.3238 BPB (100KB, -0.0043)
- Online gradient descent on component weights. Optimal eta: 0.005-0.01.
- Weights converge to w_ng≈1.8, w_b≈1.1.
- See `docs/research/r08-adaptive-mixer.md`.

### Scaling Analysis (2026-10-02) — ALL KILLED

- RWKV-7 0.4B World v2.9: 1.6549 (worse than 0.1B)
- RWKV-7 G1d 0.4B: 1.8817 (under-trained)
- RWKV-7 G1k 1.5B: 5.2691 (catastrophic domain mismatch)
- **No RWKV-7 checkpoint >0.1B outperforms 0.1B on enwik8.**
- See `docs/research/r05-scaling-analysis.md`.

---

## Phase 1 — Universal Compressor Core (2026-10-05)

### P1.1: Arithmetic Coder (CDF-24)

- **Planned impact**: -0.05 to -0.10 BPB
- **Actual**: Coder adds overhead (0.0508 at 10KB), not BPB gain.
  Overhead dominated by CDF quantization on 65K vocab.
- **Rationale**: Without arithmetic coder, we don't compress — we measure.
  CDF-24 (24-bit precision) is industry standard.
- **Result**: 1.3320 compressed BPB (10KB), roundtrip verified.
- Range coder with Schindler carry propagation, 65K vocab.

### P1.2: Byte-level Context Mixing

- **Planned impact**: -0.05 to -0.15 BPB
- **Actual**: 2.09 BPB standalone (1MB). Not additive — CM feeds mixer.
- **Rationale**: Single biggest architectural gap. Every sub-1.0 system has
  byte-level context models. Ported from analytic-lm heritage.
- **Components**: 9 orders (bit-level, MSB first), 4-way associative hash
  tables, recency decay=0.90, logistic mixing.
- **Heritage**: analytic-lm achieved 1.58 BPB with 54 CM. Top ~20 models
  ported. Diminishing returns after ~50.
- 78 MB hash tables, 220K B/s standalone throughput.

### P1.3: RWKV-to-Byte Bridge

- **Planned impact**: Enables mixing neural + statistical predictions
- **Actual**: -0.0060 BPB hybrid (1.2924 at 100KB), 169 B/s
- **Approach**: TokenByteTrie marginalizes RWKV token logits to byte-level
  probabilities. 116K trie nodes from 65K vocab entries.
- **Heritage**: Nacrith does this (SmolLM2 → byte mixer).

### P1.4: Confidence Skip — KILLED

- **Planned impact**: 2-5x throughput
- **Actual**: <3% speed gain, +0.0026 BPB at best.
- **Root cause**: RWKV sequential state prevents skipping the dominant
  compute cost (97% of time). Cannot skip RWKV forward pass without
  corrupting state for subsequent tokens.

### Q8 Quantization (all layers)

- **Result**: 1.2984 BPB (-0.0254), -75% RAM, +80% speed
- All 6 large matrices per layer: Q8 per-row (i8 + f32 scale).
- i8×i8→i32 accumulation auto-vectorizes to AVX2/VNNI.
- Double quantization acts as implicit regularization.

### Scratch Arena (buffer reuse)

- **Result**: +46% speed (80→117 B/s), ~0 allocs/token
- 149 KB pre-allocated workspace. ~1400 allocs/token → ~0.
- Rust iterator patterns critical for SIMD auto-vectorization.

### AVX-VNNI (post-scratch)

- **Result**: +38% speed (117→162 B/s)
- `q8_mat_vec_mul_vnni_into` — zero-alloc VNNI kernel.
- Only works after scratch arena eliminated cache pollution.
- Pre-scratch VNNI was -33% (memory-bound). Post-scratch +38% (compute-bound).
- **Key lesson**: optimize memory access BEFORE compute instructions.

---

## Phase 2 — Advanced Mixing (2026-10-06)

### P2.1: LSTM Mixer

- **Planned impact**: -0.05 to -0.22 BPB
- **Actual**: -0.0375 BPB (1.2549 at 100KB), 134 B/s
- **Config**: H=128, 67K params, BPTT=1, SGD lr=0.002
- **Why less than heritage (-0.22)**: 10 models (9 CM + 1 RWKV) vs
  analytic-lm's 54. Still improving at 100KB.
- **Heritage confirmed**: BPTT=1 only. SGD > Adam. HID=128+.

### P2.2: Hierarchical Model Groups

- **Planned impact**: -0.02 to -0.05 BPB
- **Actual**: -0.0277 BPB (1.2272 at 100KB), 138 B/s
- **Groups**: [CM 0-2] [CM 3-8] [RWKV bridge] [Match model]
- Logistic sub-mixers per group → LSTM top mixer.
- Reduces interaction noise between unrelated model types.

### P2.3: Cross-Domain Validation

- **Initial**: 5 domains tested, σ=0.964, adversarial PASS (2026-10-06)
- **Silesia Corpus**: 12 files, mean 2.29 BPB, σ=1.72, Weissman=4.9
- **Evolution**: Led to R28 (composite metric) after bias analysis.
  Now formalized as Tier 1/2/3 eval suite with composite gate.

---

## Phase 3 — Frontier Techniques (2026-10-06)

### P3.1: Match Model (simplified SA-PPM)

- **Planned impact**: -0.10 to -0.30 BPB
- **Actual**: -0.0095 BPB (1.2177 at 100KB), 138 B/s
- Hash-based longest match, 6 context lengths (4-128 bytes), 32 MB.
- Far from full SA-PPM potential — simplified hash approach.

### P3.2: Domain Checkpoint — BLOCKED

- Requires GPU. i5-1235U CPU only.
- ts_zip achieves 1.11 BPB with RWKV-169M, but domain-specific.

---

## Research Milestones (2026-10-06)

### R27: Embedding Surgery + Telemetry Deep Analysis

- **Result**: 1.1895 BPB (100KB enwik8), -0.0282, zero runtime cost
- center0.3: blend byte embeddings (rows 0-255) 30% toward global centroid.
- Telemetry analysis: 3 bit-cost regimes, U-shape = regime change not overfitting.
- See `docs/research/r27-telemetry-deep-analysis.md`.

### R28: Composite BPB Metric (structural de-biasing)

- Identified 4 types of measurement bias (corpus, sample, technique selection,
  validation asymmetry).
- Redefined primary metric as vector (mean, sigma, worst).
- Accept rule: mean DOWN + sigma SAME/DOWN + worst not UP >0.05.
- Tier 1/2/3 eval suite formalized.
- Surgery validated as universal: all 3 composite metrics improve.
- See `docs/research/r28-composite-metric-debias.md`.

### R29: OEIS Numerical Regime

- LaTeX is NOT distinct from code at byte level.
- Pure numerical data IS genuinely distinct (low entropy, high BPB).
- OEIS integrated as 5th Tier 1 file: 1.9045 BPB with surgery.
- Surgery largest improvement on OEIS (-0.0551) of all domains.
- See `docs/research/r29-numerical-regime-benchmark.md`.

---

## Projections vs Actuals

### Phase 1-2 Projections

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

### Phase 2-3 Actuals

```
1.2549  LSTM flat hybrid
1.2272  + hierarchical grouping (-0.0277)
1.2177  + match model (-0.0095)
1.1895  + embedding surgery center0.3 (-0.0282)
```

### Key Lessons

1. **Cumulative gains don't add linearly.** Each component's contribution
   depends on what's already in the stack. CM standalone = 2.09, but its
   value is in providing diverse inputs for the LSTM mixer.

2. **LSTM gain (-0.0375) less than heritage predicted (-0.22)** because we
   have 10 models (9 CM + 1 RWKV) vs analytic-lm's 54.

3. **Confidence skip was architectural dead end** — RWKV sequential state
   prevents skipping the dominant compute cost.

4. **Surgery helps non-text MORE than text** — largest delta on OEIS (-0.0551),
   smallest on enwik8 (-0.0228). Universal technique.

5. **Single-corpus metrics cause structural bias** — led to R28 composite
   metric to prevent enwik8-specific optimization.

---

## Optimization Series (2026-10-06 to 2026-10-07)

### S1: Coupled Gates (R35)

- **Result**: 1.1955 BPB (+0.0033 neutral), -25% params, +15% speed
- i_gate = 1 - f_gate. Prerequisite for S2/S3.

### S2: LayerNorm (R36)

- **Result**: 1.1898 BPB (-0.0057), +768 params (+1.5%)
- Strong early learning boost. Prerequisite for S3.

### S3: BPTT=8 (R37)

- **Result**: 1.1843 BPB (-0.0055 vs S2). New best.
- Adam(beta1=0.02, beta2=0.9999). First temporal learning. 148 B/s.

### S4: WordModel (R38)

- **Result**: 1.1852 BPB (+0.0009 neutral). Kept for diversity.
- Redundant with RWKV word-level knowledge. +12 MB.

### R27: Embedding Surgery center0.3

- **Result**: 1.1895 BPB (-0.0282), zero runtime cost
- Blend byte embeddings toward global centroid. Universal.

### R49: E8/E9 Transform

- **Result**: ooffice -0.1811, mozilla +0.006, text neutral. T2 mean -0.0151.
- Byte transform for x86 executables. Preprocessing works.

---

## Tier Experiments (A/B/C series, 2026-10-06 to 2026-10-07) — ALL KILLED

| Experiment | Result | Root cause |
|---|---|---|
| A1 APM/SSE (R39) | +0.10 to +0.19 | LSTM well-calibrated, APM too sparse at 100KB |
| A2 Match multi-input (R40) | +0.013 | Mixer group overhead pattern |
| B1 2-layer LSTM (R43) | +0.0014 (2×128) | Early boost only, regresses at 100KB |
| B2 BPTT scaling 16/32 bits (R41) | +0.0000/+0.0020 | Bit-level ceiling — same info processed more |
| B3 ISSE chains | KILLED by analogy | Same family as A1 (SSE), same scale problems |
| B4 Higher-order CM 12/16 (R42) | +0.0004 | Redundant with RWKV |
| C1 Online LSTM expert (R44) | +0.0127 | Mixer group overhead > prediction value |
| C2 Information inheritance | KILLED by analysis | Redundant with hierarchical mixer |
| C3 Modality-routing | KILLED by analysis | Violates no-domain-detection rule |

---

## Analysis & Baselines (2026-10-07 to 2026-10-08)

### R45: T2 Final Silesia (12 files, 10KB)

- **Result**: mean=2.2799, sigma=1.6843, worst=6.0483 (sao)
- Architecture at hard local minimum at 100KB scale.
- 17 experiments, only S2 (-0.0057) and S3 (-0.0055) improved BPB.

### R46: Domain Analysis + MoE Feasibility

- Identified 4 domain clusters (A/B/C/D) and 3 driving factors.
- Tokenization (40%) + entropy (35%) + RWKV alignment (25%).
- Context mixing IS soft MoE — gap vs cmix is scale + preprocessing.

### R47: RWKV Per-Domain Contribution

- RWKV helps all 14 files (5-61%). No bypass viable.
- Contribution by cluster: A=51%, B=51%, C=28%, D=12%.

### R48: Benchmark Corpus Investigation

- Silesia (2003) morphologically outdated for modern data.
- AIT DCC 2026 = best modern alternative. T3 eval suite designed.
- T1b/T2b (100KB) correct header-bias in binary files.

### R50: T1b/T2b/T3 Baselines (25 files, 100KB)

- **T1b**: enwik8=1.1852, OEIS=2.5353
- **T2b**: mean=1.8814, sigma=1.4825, worst=5.2470 (sao)
- **T3**: mean=3.3788, sigma=2.5716, worst=7.9891 (ait-D)
- 12.1h total. State files saved (AZ02 format, ~130 MB each).
- 3D position matrix: X=B/Tok, Y=BPB, Z=bits3-5%. Three zones identified.
- Key: B/Tok~1.0 cliff, T3 is 1.8x harder than T2b.

### R51: Organic Architecture Reform

- **Status**: Proposal approved, pending implementation.
- Central finding: gap vs SOTA is integration, not models.
  BPTT=8 vs 128 (16x gap), mixer blind to bytes, no post-correction.
- 3-layer organic solution + adaptive preprocessing:
  - Phase 0: Delta + byte-plane split (0 risk)
  - Phase 1: Byte-context LSTM, BPTT=64, 44-float input (low risk)
  - Phase 2: Tweedie post-correction, 24KB (low risk, math-guaranteed)
  - Phase 3: uSSM byte-level D=32 L=2, pre-blend with RWKV (medium risk)
  - Phase 4: CM order-chain + rank encoding (experimental)
- 6 unexplored edges (E1-E6) from bias analysis.
- Target: enwik8 ~1.14, T2b mean ~1.50.
- Backed by 11 sources (2024-2026 ecosystem research).

---

## R51 Organic Reform — Implementation (2026-10-08)

### R52: Phase 0 — Adaptive Preprocessing — KILLED

- **Transforms**: delta (stride 1/2/4/8), byte-plane split (stride 2/4/8), auto-detect
- **Auto-detection**: 8KB entropy sample, threshold 0.15 bits/byte, zero false positives on text
- **Result**: CATASTROPHIC on hybrid mode. Pre-trained RWKV sees transformed data
  as out-of-distribution → confident wrong predictions.
  - mozilla: 1.14 → 5.26 BPB (+4.13)
  - mr: 1.42 → 3.81 BPB (+2.39)
  - ooffice: 2.93 → 5.97 BPB (+3.04)
  - All text files: identity (neutral, correct)
- **Root cause**: Preprocessing only viable when ALL predictors are online-adaptive.
  Pre-trained models learn priors from original data distribution; transforms
  create anti-priors. Exception: E8/E9 works because it touches <1% of bytes.
- **Code retained**: E8/E9 still used, `--preprocess` flag available.
- See `docs/research/r52-adaptive-preprocessing-phase0.md`.

### R53: Phase 1 — Byte-Context LSTM — CONFIRMED

- **Changes**: BPTT 8→64 (8 bytes temporal), 40 context features (bit position
  one-hot + last 4 decoded bytes as binary), context separation in LSTM
  (model probs get mixing weights, context enriches hidden state only).
- **Result (enwik8 100KB)**: 1.1810 BPB (-0.0033 vs S3 baseline 1.1843)
- **T1 Composite Gate (10KB, 5 files)**:
  - mean: 1.4658 (-0.0016) ✓
  - sigma: 0.2677 (-0.0353) ✓
  - worst: 1.8200 (-0.0178) ✓
  - **All three criteria improved — PASS.**
- **LR tuning**: 0.002 optimal. Higher (0.004-0.008) all worse.
- **Throughput**: -19% to -25% (40-92 B/s vs 40-133 B/s baseline).
- **Parameters**: 66,690 (vs 51,330 baseline), +15K for context features.
- **Key insight**: BPTT=64 converges slower (8x fewer Adam updates) but to
  a better minimum. Crossover at ~60-70KB on enwik8.
- **Structural difference from B2 (KILLED)**: B2 tested BPTT=16/32 with same
  4-float input → zero benefit. R53 adds 40 NEW features, making longer
  temporal context meaningful.
- See `docs/research/r53-phase1-byte-context-lstm.md`.

### R54: Phase 2 — Tweedie Post-Correction — KILLED

- **Approach 1 (Tweedie shrinkage)**: Track per-context (bit_pos × last_byte = 2048
  buckets) mean/variance of mixer logits. Shrink toward context mean using James-Stein.
  - MIN_COUNT=8: enwik8 1.1959 (+0.0293)
  - MIN_COUNT=64: enwik8 1.1767 (+0.0101)
  - MIN_COUNT=1000: enwik8 1.1676 (+0.0010, near no-op)
- **Approach 2 (bias calibration)**: Track EMA(prediction - actual_bit) per context.
  Subtract bias to correct systematic over/under-prediction.
  - 10KB: 1.2455 (+0.0789) — 1.8154 before gradient isolation fix
  - 100KB with 10KB warmup: 1.2305 (+0.0453 vs 1.1852 baseline)
- **Critical bug found and fixed**: Initially passed corrected prediction to mixer
  update, contaminating LSTM gradients. After fix, regression reduced but still present.
- **Root cause**: The LSTM mixer is online-adaptive — it corrects its own biases via
  gradient descent. Adding ANY post-correction (shrinkage, calibration, SSE, APM)
  creates a second adaptation loop that interferes with the first. James-Stein
  assumptions (independent observations, fixed parameters) are violated. This is the
  generalized form of "cascaded SSE overcorrects" — even single-stage correction
  harms an already-adapting system.
- **Code deleted**: Module removed (zero dead code policy).
- See `docs/research/r54-phase2-tweedie-postcorrection.md`.

### R55: Phase 3 — Neural Blend — CONFIRMED

- **Approach**: Reuse existing LstmExpert (44K params) but BLEND its byte
  predictions with RWKV at probability level before the mixer. Adaptive alpha
  (sigmoid gate) degrades gracefully: alpha→1.0 when expert is bad.
- **Key difference from R44 (KILLED)**: R44 added expert as separate mixer group
  (overhead) or in shared group (contamination). R55 blends at byte level — mixer
  sees identical Group 2 shape, zero new parameters.
- **T1 Composite Gate (10KB, 5 files)**:
  - mean: 1.4437 (-0.0221) — **DOWN**
  - sigma: 0.2672 (-0.0005) — **DOWN**
  - worst: 1.8215 (+0.0015) — **OK** (< 0.05)
  - **All three criteria satisfied — PASS.**
- **Per-domain**: enwik8 unchanged (alpha≈0.99), mozilla -0.0096 (expert helps binary),
  dickens -0.0008, samba -0.0016, OEIS +0.0015 (noise).
- **CLI**: `--neural-blend` flag, `--blend-lr F` (default 0.005).
- **Throughput**: negligible impact (expert ~0.1% of RWKV cost).
- See `docs/research/r55-phase3-neural-blend.md`.

### R56: Phase 4 — CM Order-Chain (E3) — CONFIRMED

- **Approach**: Feed order-N's stretch(p) as quantized hash context to order-N+1.
  4-bin quantization of lower-order logit mixed into FNV hash. Higher-order models
  effectively "specialize" based on lower-order confidence.
- **T1 Composite Gate (10KB, 5 files)**:
  - mean: 1.4412 (-0.0247) — **DOWN**
  - sigma: 0.2574 (-0.0103) — **DOWN**
  - worst: 1.8184 (-0.0016) — **DOWN**
  - **All three criteria satisfied — PASS.**
- **Per-domain**: ALL 5 domains improved. enwik8 -0.0033, dickens -0.0022,
  samba -0.0013, mozilla -0.0150 (strongest), OEIS -0.0016.
- **Zero new parameters, zero new memory**: chain encoded in hash function.
- **CLI**: `--order-chain` flag.
- See `docs/research/r56-phase4-order-chain-rank-encoding.md`.

### R56: Phase 4 — Rank Encoding (E4) — KILLED

- **Approach**: MTF (Move-To-Front) encoding on history bytes. Store frequency
  rank instead of raw byte as CM context.
- **Result**: mean +0.0098, sigma +0.0082, worst +0.0146. All three criteria FAIL.
- **Root cause**: MTF is unstable at 10KB — ranks change continuously, destroying
  exact byte matching. Context-space (ranks) vs prediction-space (raw bytes) mismatch.
- Code removed (zero dead code policy).

### CLI Defaults Change

- `--hierarchical`, `--match`, `--emb-surgery center0.3` now ON by default.
- Use `--no-hierarchical`, `--no-match`, `--no-emb-surgery` to disable.
- **Root cause**: During R53 T1 eval, flags were accidentally omitted,
  producing false catastrophic results (+0.88 on mozilla) that almost killed
  Phase 1. Making full pipeline the default eliminates this error class.
- Minimal valid command: `cargo run --release -- hybrid-eval --weights weights/rwkv7-0.1b --bytes N --input PATH`

---

## Killed Approaches (full list)

| Approach | Result | Root cause |
|---|---|---|
| Surprise-modulated lr | +0.0039 on 100KB | EMA init artifact |
| Inverse decay lr | -0.0012 at best | Blind schedule |
| StateSMix entropy-adaptive N-gram | +0.0016 on 100KB | Noise |
| All RWKV scaling paths (0.4B, 1.5B) | Worse than 0.1B | Domain mismatch / under-trained |
| Ternary (1.58-bit) | PTQ collapse (PPL >4000) | RWKV recurrence propagates quant noise |
| Q4 quantization | +30% PPL | Too aggressive for 0.1B |
| Block-32 Q8 | +0.0378 BPB, -34% speed | RWKV uniform weights don't benefit |
| AVX-VNNI pre-scratch | -33% speed | Memory-bound (cache pollution) |
| All alt number systems | Various | See R14 |
| Confidence skip (P1.4) | <3% speed, +0.0026 BPB | RWKV=97% compute, can't skip |
| A1 APM/SSE (R39) | +0.10 to +0.19 | LSTM well-calibrated, APM sparse at 100KB |
| A2 Match multi-input (R40) | +0.013 | Mixer group overhead pattern |
| B1 2-layer LSTM (R43) | +0.0014 | Early boost only, regresses at 100KB |
| B2 BPTT=16/32 bits (R41) | +0.0000/+0.0020 | Bit-level ceiling, same info |
| B4 Higher-order CM (R42) | +0.0004 | Redundant with RWKV |
| C1 Online LSTM expert (R44) | +0.0127 | Mixer group overhead > prediction value |
| R52 Adaptive preprocessing | +4.13 mozilla | Transforms destroy RWKV pre-trained predictions |
| R54 Tweedie post-correction | +0.0293 to +0.0789 | Second adaptation loop fights mixer's online learning |
| R55 Neural blend (Phase 3) | **-0.0221 mean** | **CONFIRMED: expert pre-blend with RWKV, T1 composite PASS** |
| R56 Order-chain (Phase 4 E3) | **-0.0247 mean** | **CONFIRMED: chain order-N → N+1 hash, all 5 domains ↓** |
| R56 Rank encoding (Phase 4 E4) | +0.0098 mean | MTF context destroys exact matching at 10KB |

---

## Strategic Research & Roadmap Restructure (2026-10-09)

### R57: Strategic Positioning

- **Purpose**: Define azathoth-lm's end goal beyond compression benchmarks.
- **Key insight**: azathoth-lm is a universal data synthesis core, not a compressor.
  Compression is the proving ground, NOT the product.
- **Subproduct**: adaptive inference middleware — sellable NOW.
- See `docs/research/r57-strategic-positioning.md`.

### R58: Multi-Backbone Inventory

- **Purpose**: Exhaustive inventory of pretrained backbones across ALL domains.
- **50+ models cataloged** across 12 domain categories.
- **Key discovery**: MambaByte (353M, byte-level SSM, vocab=256, Apache-2.0).
  Native byte-level model — outputs P(byte) directly, no tokenizer bridge needed.
  O(1) per step (like RWKV). Domain variants: Code, ArXiv, Books, Wiki, PG-19.
- **Architecture gap identified**: code is RWKV-monolith (`bridge.rs` hardcodes
  `WorldTokenizer`, `main.rs` lines 589-714 inline). Needs `ByteBackbone` trait.
- **Upgrade path**: RWKV7-G1k 0.1B (191M, 5T+ tokens) is drop-in upgrade to
  current 0.1B World v2.8 (100M, ~1T tokens).
- See `docs/research/r58-multi-backbone-inventory.md`.

### R59: T2b R56 Deep Analysis

- **Purpose**: Deep quantitative analysis of T2b R56 results (12 Silesia × 100KB).
- **T2b R56 composite**: mean=1.8678 (-0.0136), sigma=1.3857 (-0.0337),
  worst=4.9844 (-0.2626). 10/12 files improve.
- **Convergence analysis**: 4 files strongly converging (samba, webster, osdb, sao),
  2 diverging (mozilla, x-ray), 1 anti-learning (ooffice).
- **Bit-cost profiles**: 3 distinct patterns identified:
  - Profile A (ASCII text): bits 3-5 dominate (56-66%)
  - Profile B (restricted range): bits 5-7 dominate (70-81%)
  - Profile C (high entropy/binary): near-uniform distribution
  - Profile D (non-English UTF-8): bit1 anomalously high (12% vs 6%)
- **ooffice regression root cause (+0.3616)**: ANTI-LEARNER — only file where BPB
  increases over time (2.39→2.93→3.29). OLE2 alternates structured/random sections.
  Order-chain propagates stale context, neural-blend injects OOD noise.
- **reymont regression root cause (+0.0406)**: UTF-8 byte/character mismatch.
  Polish diacritics are 2-byte sequences; order-chain crosses char boundaries.
- **Competitive position**: **We beat PAQ8PX v217 on dickens at only 100KB**
  (1.3387 vs 1.4568). PAQ uses the full 10MB file.
- **Scale projections**: 6/12 files show 30-50% improvement potential at 1MB+.
- See `docs/research/r59-t2b-r56-deep-analysis.md`.

### R60: Roadmap Restructure — Multi-Backbone Architecture

- **Purpose**: Restructure roadmap with security-first, evaluate-before-integrate approach.
- **Sequence**: EVALUATE (security + quality) → MODULARIZE (decouple) → INTEGRATE (adopt).
- **Phase A**: Backbone Security & Quality Gate (zero code changes)
  - Weight format security policy: pickle NEVER safe (arbitrary code execution by design).
    Documented: CVE-2026-4372, CVE-2026-1839, ShadowPickle (2026), JFrog zero-days.
    SafeTensors only audited format (Trail of Bits 2023).
  - Per-candidate metadata cards with security risk, format, SHA-256, conversion needs.
  - 5-phase evaluation protocol with kill criteria at each phase.
  - Already killed (Phase 0): ESM-2, DNABERT-2, ByT5, BLT, MEGABYTE.
- **Phase B**: Architecture Modularization — **COMPLETE** (2026-10-09)
  - B.1: `ByteBackbone` trait (5 methods, minimal interface).
  - B.2: `RwkvBackbone` struct (model + tokenizer + bridge + token tracking).
  - B.3: `BackboneOrchestrator` (primary RWKV + auxiliary Vec, pre-blend).
  - B.4: Bit-level adapter (pre-existing `byte_probs_to_bit_preds`).
  - B.5: Validation gate PASSED — all T1 files within ±0.0006 BPB.
  - Files: `src/domain/backbone.rs` (new), `src/main.rs` (refactored).
- **Phase C**: Backbone Adoption — C.0.x internal optimization complete
  - ~~C.0.1~~: G1k 0.1B upgrade — KILLED (model doesn't exist, G1k starts at 1.5B).
  - ~~C.0.2~~: Confidence-gated order-chain — KILLED (ooffice -0.0128, 3.5% of regression, insufficient).
  - C.0.3: Eval at 1MB scale — PENDING.
  - ~~C.0.4~~: G1k 1.5B Q8 as second backbone — KILLED.
    Standalone: 2.1779 BPB (vs 1.1650 0.1B). Dual pre-blend: 1.3781 (+0.2131).
    G1k 1.5B predicts worse than 0.1B World at 10KB. Uniform blend dilutes quality.
    Commands: `--weights weights/rwkv7-g1k-1.5b --bytes 10000` (standalone),
    `--backbone2 weights/rwkv7-g1k-1.5b --bytes 10000` (dual). Speed: 16-22 B/s.
  - **C.0.5**: Stride-4 sparse model — **DONE** (ait-E -0.5232 BPB, text neutral +0.0008).
  - ~~C.0.6~~: CM confidence-based RWKV skip — KILLED (+0.76 to +1.20, breaks adaptive mixer).
  - ~~C.1~~: MambaByte-Code 353M — KILLED (security: .pth only, pickle = arbitrary code execution).
  - C.2: ProGen2-small — CONFIRMED (BSD-3, SafeTensors native, covers ait-A protein).
    - ~~Evo 2~~: KILLED (nucleotides ≠ amino acids, RAM too large).
    - ~~BioGPT~~: deprioritized (redundant with RWKV for text).
    - ~~WaveNet~~: deprioritized (no pretrained weights found).
  - **C.0.x lesson**: 4 killed, 1 done. Internal optimization exhausted. Only adding
    new information works (C.0.5). Modifying mixer inputs externally always fails.
- **Phase D**: Advanced Features (confidence-skip, per-backbone weighting, scale up,
  streaming mode). D.4 (self-distillation) killed with C.1.
- **Replaces**: old Tiers S/A/B/C + R51 (all complete/archived).

### R61: AIT DCC 2026 Backbone Coverage Analysis

- **Purpose**: Map T3 Modern files to backbone candidates, verify Phase A compliance.
- **Method**: Domain inspection (file/xxd), HuggingFace API queries, web search for
  candidates covering IEEE-754 floats, executables, protein, astronomical data.
- **Result**: 6/11 covered by existing candidates, 1 incompressible (ait-D pseudo-random),
  4 gaps — ALL are IEEE-754 float data.
- **Candidates killed (Phase A)**:
  - BOA Constrictor (Mamba HEP): AGPL-3.0 license (incompatible).
  - AstroPT (galaxy images): patch-level, not byte-level (architecture mismatch).
  - Large Byte Model (binary analysis): weights not published.
  - Evo 2 1B (DNA): wrong domain (nucleotides ≠ amino acids) + RAM too large.
- **Key finding**: No pretrained autoregressive model exists for IEEE-754 float data
  with compatible license. Solution is byte-plane separation (ZipNN technique)
  applied selectively to CM models only (RWKV sees raw bytes per R52 lesson).
- **C.0.5 stride-4 sparse**: ait-E -0.5232 BPB (6.74→6.22), text neutral (+0.0008). DONE.
- **C.0.2 confidence gate**: ooffice -0.0128 (3.5% of +0.36 regression). KILLED — insufficient.
- **C.0.6 CM-based RWKV skip**: enwik8 +0.76 to +1.20, ooffice +0.03. KILLED —
  intermittent input changes break adaptive mixer (same root cause as R54).
- See `docs/research/r61-ait-dcc-backbone-coverage-analysis.md`.
- See `docs/research/r60-roadmap-restructure-multi-backbone.md`.
