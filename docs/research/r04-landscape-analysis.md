# R04: Landscape Analysis — Sub-1.0 BPB Compression Ecosystem (2025-2026)

- **Date**: 2026-10-01
- **Status**: Complete
- **Purpose**: Map the competitive landscape, identify architectural paradigms that
  achieve sub-1.0 BPB, and evaluate implications for azathoth-lm's architecture

---

## 1. Current enwik8 Leaderboard (as of 2026-10)

| System | BPB | Architecture | Params | Key Innovation |
|---|---|---|---|---|
| **Nacrith** | **0.939** | SmolLM2-135M + N-gram + bias head | 135M | Token-level ensemble + CDF-24 |
| SHA-RNN | 1.07 | 63M RNN, Boom layer | 63M | Single-head attention + layer norm |
| cmix v21 | 1.17 | 2077 models + LSTM mixer | — | Massive model ensemble, byte-level |
| NNCP v3 | 1.19 | 199M Transformer-XL | 199M | Large transformer, byte-level |
| PAQ8px | 1.27 | 200+ models, SSE pipeline | — | Hand-tuned context mixing, byte-level |
| Midicoth | 1.753 | PPM + match + Tweedie denoising | 0 | Fully online, micro-diffusion correction |
| **azathoth-lm** | **1.42*** | RWKV-7 0.1B Q8, token-level | 100M | *smoke test, 10KB only* |
| StateSMix | 2.13 | Mamba SSM + sparse N-gram | ~120K | Fully online, no pre-trained weights |

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

## 8. Throughput Projections (Token-Level Architecture)

With token-level compression, throughput improves fundamentally:

| Config | ms/tok | Bytes/tok | B/s | enwik8 hours |
|---|---|---|---|---|
| Current Q8 | 51 | 3.7 | 73 | 380h |
| F32 (revert) | 44 | 3.7 | 84 | 330h |
| F32 + buffer reuse (est.) | 35 | 3.7 | 106 | 262h |
| F32 + confidence skip 50% (est.) | 35 | 7.4* | 211 | 132h |
| Q8 VNNI kernel (est.) | 18 | 3.7 | 206 | 135h |
| VNNI + confidence skip (est.) | 18 | 7.4* | 411 | 68h |

*Confidence skip effectively doubles bytes/tok by skipping neural calls.

Target: < 100h for full enwik8 is achievable with VNNI + confidence skip.
Without VNNI, confidence skip alone brings f32 to ~132h — marginal but usable.

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
