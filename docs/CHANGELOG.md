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
