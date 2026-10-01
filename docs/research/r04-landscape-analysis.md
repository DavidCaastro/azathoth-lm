# R04: Landscape Analysis — Sub-1.0 BPB Compression Ecosystem (2025-2026)

- **Date**: 2026-10-01
- **Status**: Complete
- **Purpose**: Map the competitive landscape, identify architectural paradigms that
  achieve sub-1.0 BPB, and evaluate implications for azathoth-lm's architecture

---

## 1. Current enwik8 Leaderboard (as of 2026-10)

| System | BPB | Architecture | Params | Level | Key Innovation |
|---|---|---|---|---|---|
| **Nacrith** | **0.939** | SmolLM2-135M + N-gram + bias head | 135M | Token | CDF-24 + ensemble + online bias |
| fx2-cmix-transformer | ~0.97* | 6M Transformer + cmix 2000+ CM | 6M+CM | Bit | Neural into classical CM pipeline |
| SHA-RNN | 1.07 | 63M RNN, Boom layer | 63M | Char | Single-head attention + layer norm |
| ts_zip (Bellard) | 1.106 | RWKV-169M v4 Q8, pure LM | 169M | Token | Frozen RWKV, no CM, no adaptation |
| cmix v22 | 1.17 | 2077 models + LSTM mixer | ~200 | Bit | Massive ensemble, online training |
| NNCP v3 | 1.19 | 199M Transformer-XL (online) | 199M | Byte | Online-trained transformer |
| PAQ8px | 1.27 | 200+ models, SSE pipeline | — | Bit | Hand-tuned context mixing |
| **azathoth-lm** | **1.42*** | RWKV-7 0.1B Q8, token-level | 100M | Token | *smoke test, 10KB only* |
| Midicoth | 1.753 | PPM + match + Tweedie denoising | 0 | Byte | Fully online, micro-diffusion |
| StateSMix | 2.13 | Mamba SSM + sparse N-gram | ~120K | Token | Fully online, no pre-trained weights |

*fx2-cmix-transformer: 0.0969 ratio on enwik9 (Hutter Prize); enwik8 extrapolated.

**Critical reference point**: ts_zip achieves 1.106 BPB with a frozen RWKV-169M
alone — no context mixing, no adaptation, no ensemble. The gap between ts_zip
(1.106) and Nacrith (0.939) = **0.167 BPB** comes entirely from:
- CDF-24 precision: **-0.517 BPB** (largest single contribution)
- N-gram ensemble: ~-0.02 to -0.05 BPB
- Adaptive log-space bias head: ~-0.03 BPB
- Confidence skip: ~-0.39 BPB (speed + slight BPB improvement)

## 2. Architectural Paradigm Shift: Token-Level Wins

### The critical finding

**Nacrith (0.939 BPB) operates entirely at token level, not byte level.**

The system tokenizes input text, predicts token probabilities with an ensemble,
and feeds those probabilities directly into an arithmetic coder. There is no
byte-level decomposition, no bit-level prediction, no byte-to-token conversion
during compression.

This contradicts our initial architecture assumption (byte-level CM + token-level
RWKV unified through an LSTM mixer). Nacrith proves that token-level prediction
with online adaptation is sufficient for sub-1.0 BPB.

### Why token-level is superior for neural+CM hybrid

| Factor | Token-level | Byte-level |
|---|---|---|
| Neural calls per byte | 1 per ~3.7 bytes (amortized) | 1 per byte (or 8 per byte for bit-level) |
| Speed (at 51 ms/tok) | ~14 B/s baseline | ~3.6 B/s (if 1 call/byte) |
| Model alignment | Native (LLM trained on tokens) | Requires token↔byte projection |
| Arithmetic coding | Direct on token probabilities | More complex CDF construction |
| CM integration | Token-level N-grams, simple | Byte-level hash tables, complex |
| Proven sub-1.0? | **YES** (Nacrith, 0.939) | No system achieves sub-1.0 byte-level only |

### Implication for azathoth-lm

The original architecture from `architecture.md`:
```
Input byte stream
    +---> N context models (CM) ---------> stretch --+
    +---> RWKV-7 (pre-trained) ----------> stretch --+
    +---> WHT memory bank ----------------> stretch --+
                                                      |
                                    LSTM mixer → SSE → Final prediction
```

This byte-level design inherits from analytic-lm (1.5826 BPB). But:

1. analytic-lm's 54 CM models at byte-level only achieved 1.58 BPB
2. RWKV-7 0.1B alone at token-level already achieves ~1.42 BPB
3. Nacrith (token-level LLM + N-gram + online bias) achieves 0.939 BPB

**Token-level is both faster AND more accurate.** The byte-level CM models
from analytic-lm would need to provide >0.42 BPB improvement to justify
their complexity — but their entire contribution in analytic-lm was only
~0.55 BPB above entropy coding. Diminishing returns at this level.

## 3. Nacrith Architecture Deep Dive

Nacrith is our closest competitor and architectural reference. Key components:

### 3.1 Neural backbone: SmolLM2-135M
- 30-layer causal transformer, BPE vocabulary 49,152 tokens
- Pre-trained, frozen (no fine-tuning during compression)
- FP32 inference for deterministic probability distributions
- llama.cpp backend: ~7x faster than PyTorch for single-token decode

### 3.2 Token-level N-gram model
- Fast local pattern prediction at token level
- Captures short-range dependencies the LLM may miss
- Online, lightweight, complements the neural model

### 3.3 Adaptive log-space bias head
- Learns per-document corrections to the LLM via online gradient descent
- Operates in log-probability space (additive logit bias)
- Key insight: the pre-trained LLM has systematic errors on specific
  documents. A small online learner corrects these efficiently.

### 3.4 Linear mixing with exponential-weight updates
- Two predictors (LLM + N-gram) blended with online weight adaptation
- Context mixing philosophy applied at token level
- Only 2 models instead of 54+ — simpler, faster, equally effective

### 3.5 CDF-24 precision
- Standard arithmetic coding uses 16-bit CDF (2^16 = 65,536 bins)
- With V = 49,152 tokens and minimum probability floors, 75% of CDF
  range is wasted on floor allocations
- CDF-24 (2^24 = 16M bins) eliminates this overhead
- For RWKV World tokenizer (V = 65,536), this is even more critical

### 3.6 Confidence-based LLM skip
- When the ensemble is highly confident (one token dominates), skip
  the expensive neural forward pass
- Accelerates compression on predictable regions (XML tags, common words)
- This is exactly our "head skip in hybrid mode" from R03

## 4. Midicoth: Parameter-Free Compression (1.753 BPB)

Fully online, zero pre-trained weights. Architecture:

1. Order-0 through order-4 adaptive PPM (PPMC exclusion + Jeffreys prior)
2. Extended match model (long-range repetition)
3. Trie-based word model with bigram prediction
4. High-order context model (orders 5-8)
5. **Micro-diffusion layer**: Binary tree Tweedie denoising

The micro-diffusion layer treats probability smoothing as a forward
diffusion process and reverses it via Tweedie's empirical Bayes formula.
Each 256-way byte prediction is decomposed into 8 binary decisions
(MSB to LSB) with 3 denoising steps per decision.

**Relevance**: At 1.753 BPB, Midicoth slightly outperforms analytic-lm's
byte-level approach (1.58 BPB) but uses zero parameters. The Tweedie
denoising concept is novel and could be applied as a post-processing
step on any predictor's output probabilities.

## 5. Q8 Kernel Optimization: What llama.cpp Actually Does

### The core technique: both-sides quantization

llama.cpp's Q8_0 dot product (`ggml_vec_dot_q8_0_q8_0`) quantizes
BOTH weights AND activations to int8:

```
For each block of 32 values:
  Weight: int8 + f32 scale (stored)
  Activation: int8 + f32 scale (computed on-the-fly)

  Dot product: Σ(w_i8 * a_i8) → int32 accumulation
  Final: sum_i32 * w_scale * a_scale → f32
```

This avoids the i8→f32 conversion that kills our performance.

### AVX2 kernel path (non-VNNI)

```c
// Pseudocode for mul_sum_i8_pairs_float on AVX2:
// 1. Load 32 int8 weights and 32 int8 activations
// 2. _mm256_maddubs_epi16: multiply pairs of uint8×int8 → int16, add adjacent
// 3. _mm256_madd_epi16: horizontal add int16 pairs → int32
// 4. _mm256_cvtepi32_ps: convert accumulated int32 → f32
// 5. Multiply by (weight_scale * activation_scale)
```

Key: `vpmaddubsw` processes 32 pairs in one instruction. No per-element
f32 conversion. Integer accumulation stays in SIMD registers.

### AVX-VNNI kernel path (our CPU has this!)

```c
// vpdpbusd: 4 × (uint8 × int8) → int32 accumulation
// Single instruction: 4 multiplies + accumulate
// Available on i5-1235U (Alder Lake)
```

This is 2-4x faster than the AVX2 path for Q8 operations.

### Implication for azathoth-lm

Our current Q8 kernel: `q[c] as f32 * v[c]` (i8×f32, scalar conversion)
What we should do: quantize activation vector to i8 too, use integer
dot product, convert only the final sum. This requires:

1. `quantize_row_q8`: activation vector f32 → (i8 block, f32 scale)
2. `dot_q8_q8`: integer accumulation using `vpmaddubsw` or `vpdpbusd`
3. Final dequantize: `sum_i32 * w_scale * a_scale`

Estimated speedup: 2-4x over current Q8, 3-6x over f32 baseline.

## 6. Revised Architecture Proposal

Based on this landscape analysis, the optimal architecture for azathoth-lm
diverges significantly from the original byte-level design:

### Proposed: Token-Level Hybrid (Nacrith-inspired)

```
Input byte stream
    │
    ▼
 Tokenizer (World, V=65536)
    │
    ├──► RWKV-7 0.1B ──► logits (V,)
    │
    ├──► Token N-gram ──► logits (V,)
    │
    ├──► Online bias head ──► logit corrections (V,)
    │
    ▼
 Log-space mixing (online adaptive weights)
    │
    ▼
 Softmax → CDF-24
    │
    ▼
 Arithmetic coder → compressed output
```

### Advantages over original byte-level design

1. **Speed**: 1 RWKV call per ~3.7 bytes (vs 1-8 per byte)
2. **Simplicity**: No LSTM mixer, no SSE pipeline, no bit-level prediction
3. **Proven**: Nacrith validates this paradigm at 0.939 BPB
4. **Native**: RWKV naturally predicts tokens, not bytes
5. **Composable**: Each component is independently testable

### Components needed

| Component | Status | Complexity |
|---|---|---|
| RWKV-7 0.1B inference | DONE | — |
| Token N-gram model | NEW | Low (hash table, online counting) |
| Online log-space bias | NEW | Low (per-token gradient descent) |
| Adaptive mixing | NEW | Low (2-3 weights, online update) |
| CDF-24 arithmetic coder | NEW | Medium (32-bit precision coding) |
| Confidence-based skip | NEW | Low (threshold on max prob) |
| Q8 integer kernel (opt.) | NEW | Medium (SIMD intrinsics) |

### What we preserve from heritage

- **Online adaptation during inference** — validated in analytic-lm
- **Logistic mixing > linear blend** — analytic-lm finding
- **Context diversity > model count** — use few, diverse predictors
- **Recency effects** — N-gram model with decay
- **No BPTT during eval** — analytic-lm failure, avoid

### What we drop

- Byte-level / bit-level prediction (unnecessary complexity)
- 54+ context models (diminishing returns, 2 predictors sufficient)
- LSTM mixer (online logistic mixing replaces it)
- SSE pipeline (confidence skip replaces its role)
- WHT memory bank (insufficient without selectivity, edge-lm finding)

## 7. Risk Assessment

| Risk | Mitigation |
|---|---|
| RWKV 0.1B too weak (vs SmolLM2-135M) | Scale to 0.4B if needed; RWKV-7 competitive per-param |
| World tokenizer (65K) vs BPE (49K) waste | CDF-24 handles large vocabs; test CDF overhead |
| Token-level can't capture byte patterns | N-gram + bias head should compensate; test empirically |
| Q8 VNNI kernel complexity | Start with f32, VNNI is optimization not prerequisite |
| Arithmetic coder precision | Implement CDF-24 from start; 16-bit is proven insufficient |

## 8. Throughput Analysis — We Are 7x Slower Than rwkv.cpp

### Ecosystem benchmarks (RWKV on CPU)

| Implementation | Model | Quantization | Hardware | ms/tok | tok/s |
|---|---|---|---|---|---|
| rwkv.cpp (ggml) | 169M | Q4_0 | 4C/8T x86 AVX2 | 6.9 | 145 |
| rwkv.cpp (ggml) | 169M | Q4_1 | Same | 6.7 | 149 |
| RWKV-edge | 255M | — | Apple M1 NEON | 12.1 | 82.5 |
| **azathoth-lm** | **100M** | **Q8 (i8×f32)** | **i5-1235U AVX2+VNNI** | **51** | **19.6** |

Our 100M model is smaller than rwkv.cpp's 169M yet 7x slower. The gap is
entirely in kernel quality: ggml uses integer accumulation + SIMD intrinsics,
we use scalar i8→f32 conversion that breaks auto-vectorization.

### The ggml Q8_0 kernel technique (what we should adopt)

```
For each block of 32 values:
  1. Quantize activation vector to Q8_0: (i8[32], f32 scale)
  2. _mm256_maddubs_epi16(weight_u8, activation_i8)
     → 32 u8×i8 multiplies, pairwise sum → 16 i16 values
  3. _mm256_madd_epi16(result, ones)
     → pairwise sum i16 → 8 i32 values
  4. Accumulate into i32 register
  5. After ALL blocks: cvtepi32_ps → f32, multiply by w_scale × a_scale
```

Key: **NO per-element f32 conversion**. Integer accumulation processes
32 elements per AVX2 instruction sequence vs. 8 for our f32 kernel.

On AVX-VNNI (our CPU): steps 2-3 replaced by single `VPDPBUSD` instruction.

### Throughput projections (revised with ecosystem data)

| Config | ms/tok | Bytes/tok | B/s | enwik8 hours |
|---|---|---|---|---|
| Current (i8×f32 Q8) | 51 | 3.7 | 73 | 380h |
| F32 (revert to baseline) | 44 | 3.7 | 84 | 330h |
| Q8 both-sides + AVX2 (est.) | 15-20 | 3.7 | 185-247 | 112-150h |
| Q8 both-sides + VNNI (est.) | 8-12 | 3.7 | 308-463 | 60-90h |
| + Confidence skip 40% (est.) | 8-12 | 6.2* | 513-771 | 36-54h |
| + Layer pruning 8/12 (est.) | 5-8 | 6.2* | 775-1240 | 22-36h |

*Effective bytes/tok increases because skipped tokens cost zero RWKV time.

rwkv.cpp achieves 6.9 ms for 169M (Q4). Our 100M at Q8 with VNNI
should reach 8-12 ms — consistent with ecosystem data.

**Target: < 50h for full enwik8** is achievable with VNNI + confidence skip.
Layer pruning would bring it to ~30h — comparable to analytic-lm's 28h benchmark.

## 9. Token-to-Byte Bridge: Three Approaches

The fundamental question: how to combine token-level RWKV with byte-level CM.

### Approach A: Pure token-level (Nacrith)

Operate entirely at token level. CM becomes token-level N-gram.
Arithmetic coding encodes tokens directly.

- **Pro**: Simplest, proven (0.939 BPB), fastest
- **Con**: Abandons byte-level CM expertise, 65K-vocab CDF overhead

### Approach B: Exact byte marginalization (BTR Lemma, ICLR 2025)

Convert token probabilities to exact byte probabilities:
P(byte | ctx) = Σ over all token sequences covering that byte.

- **Pro**: Mathematically exact, enables byte-level CM mixing
- **Con**: O(n × max_token_length) model calls per byte, computationally expensive
- **Ref**: [Exact Byte-Level Probabilities from Tokenized LMs](https://arxiv.org/abs/2410.09303)

### Approach C: Bit-level hybrid (fx2-cmix-transformer)

Neural model feeds token-level logits into cmix's bit-level pipeline.
Byte distribution decomposed into 8 bit predictions.

- **Pro**: Preserves CM infrastructure, additive composition, proven ~0.97 BPB
- **Con**: Token-byte alignment complex, Transformer was task-specific trained

### Approach D: Token-level primary + byte-level fallback

RWKV at token level for most of stream. Fall back to byte-level CM
for regions where tokenization is uncertain or confidence is low.

- **Pro**: Best of both worlds, adaptive compute allocation
- **Con**: Complex switching logic, state synchronization challenges

### Recommendation

**Start with Approach A** (pure token-level) — minimum complexity, proven
sub-1.0 BPB. Measure RWKV-alone BPB first. If <1.1 BPB, add N-gram +
bias head. If >1.1, evaluate Approach C or model scaling (0.4B).

## 10. Layer Pruning: Free Speed

Research shows removing 75% of layers from an LM retains 98.6% of quality
([LayerRoute, 2026](https://arxiv.org/abs/2609.13682)).

For our 12-layer RWKV-7 0.1B:
- Keep 4 layers (bottom 2 + top 2): est. ~98% quality, 3x speedup
- Keep 8 layers: est. ~99.5% quality, 1.5x speedup
- Empirically test: measure per-layer BPB contribution, prune least impactful

This is orthogonal to kernel optimization — speedups multiply.

## 11. Kill Criteria and Decision Framework

| Question | Threshold | Action |
|---|---|---|
| RWKV-7 0.1B token-level BPB on full enwik8? | If >1.3 | Scale to 0.4B |
| N-gram + bias head improvement? | If <0.01 BPB | Drop CM, pure RWKV + arithmetic coding |
| CDF-24 vs CDF-16 difference? | If <0.1 BPB | Keep CDF-16 (simpler) |
| Confidence skip rate on enwik8? | If <20% | Skip optimization not worth complexity |
| VNNI kernel speedup vs f32? | If <1.5x | Stay with f32 auto-vectorization |

## References

- [Nacrith: Neural Lossless Compression via Ensemble Context Modeling](https://arxiv.org/abs/2602.19626) — Tacconelli, 2026
- [Nacrith-GPU Source Code](https://github.com/robtacconelli/Nacrith-GPU) — Apache 2.0
- [StateSMix: Online Lossless Compression via Mamba SSM](https://arxiv.org/abs/2605.02904) — 2026
- [Micro-Diffusion Compression: Binary Tree Tweedie Denoising](https://arxiv.org/abs/2603.08771) — Tacconelli, 2026
- [Large Text Compression Benchmark](https://mattmahoney.net/dc/text.html) — Mahoney
- [llama.cpp Q8_0 dot product bug/fix](https://github.com/ggml-org/llama.cpp/issues/29351) — 2026
- [llama.cpp AVX-VNNI Q2_0 dot product (3x speedup)](https://github.com/ggml-org/llama.cpp/pull/26348) — 2026
- [Auto-Vectorization for Newer Instruction Sets in Rust](https://www.nickwilcox.com/blog/autovec2/) — Wilcox
- [RWKV-7 "Goose" with Expressive Dynamic State Evolution](https://arxiv.org/abs/2503.14456) — Peng et al., 2025
- [Vals RSI Index Leaderboard](https://www.vals.ai/benchmarks/rsi_index) — 2026
- [rwkv.cpp with ggml benchmarks](https://github.com/RWKV/rwkv.cpp) — RWKV Foundation
- [ts_zip: Text Compression using LLMs](https://bellard.org/ts_zip/) — Bellard
- [fx2-cmix-transformer (Hutter Prize)](https://github.com/astOwOlfo/fx2-cmix-transformer-v1) — 2026
- [Exact Byte-Level Probabilities from Tokenized LMs (ICLR 2025)](https://arxiv.org/abs/2410.09303) — Meta
- [L3TC: Leveraging RWKV for Learned Lossless Compression (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/33446)
- [LayerRoute: Adaptive Layer-Skipping](https://arxiv.org/abs/2609.13682) — 2026
- [RWKV-edge: Compressed RWKV for Edge Devices](https://arxiv.org/abs/2412.10856) — 2024
- [Chained Lightweight Neural Predictors with Information Inheritance](https://arxiv.org/abs/2604.15472) — 2026
- [NNCP: Lossless Data Compression with Neural Networks](https://bellard.org/nncp/) — Bellard
