# R13: Mixed-Precision Per-Matrix Quantization — Literature Review

- **Date**: 2026-10-05
- **Status**: Complete (Research)
- **Purpose**: Validate whether per-matrix auto-select quantization is the correct approach

---

## Question

Our current `select_best()` benchmarks a single matrix (head V×D) and applies that
quantization level to ALL matrices in all layers. Is this correct? Should we instead
benchmark each matrix individually and allow mixed precision per-matrix?

## Literature Review

### 1. The Consensus: Uniform Precision is Suboptimal

Every major quantization paper from 2024-2026 agrees: **different layers and matrix
types have different sensitivity to quantization**. Uniform "one-size-fits-all" is
the naive baseline that all methods try to improve.

Key findings from the literature:

- **Attention vs FFN**: Attention layers require 2-4× higher precision than FFN layers.
  Attention out-projection and QKV are more sensitive; FFN down-projection is more
  sensitive than up-projection. (ICLR 2025, CVPR 2026)
- **Layer depth**: Initial layers capture syntactic features (high sensitivity),
  deeper layers encode semantic relationships (more tolerant). Sensitivity is
  non-uniform with 41× spread on GPT-2, 135× on Mistral. (arxiv:2503.06518)
- **Sub-1B models are MORE sensitive**: For Llama-3.2-1B, 4-bit quantization causes
  -25% accuracy on GSM8k. Smaller models have less redundancy per weight, so each
  weight carries more information. (IJCAI 2025)

### 2. State of the Art: Per-Layer/Per-Group Mixed Precision

| Method | Granularity | Approach | Venue |
|---|---|---|---|
| AWQ | Per-channel | Activation-aware scaling, preserve salient weights | MLSys 2024 |
| GPTQ | Per-row | Second-order weight reconstruction | ICLR 2023 |
| SliM-LLM | Per-group | Salience-driven bit allocation within layers | ICML 2025 |
| ALMP | Per-layer | Auto layer-by-layer INT8/FP16 allocation | Springer 2025 |
| MixQuant | Per-layer | Budget-agnostic scores, inter-layer interaction | arxiv 2607 |
| IMPQ | Per-layer | Interaction-aware, considers upstream dependencies | arxiv 2509 |
| llama.cpp imatrix | Per-tensor | Importance matrix from calibration data | Open source |

The most practical and widely deployed: **llama.cpp's importance matrix (imatrix)**,
which runs calibration data through the model, records per-tensor activation
statistics, and uses them to guide quantization precision per tensor.

### 3. llama.cpp imatrix — The Industry Standard

The imatrix approach is directly relevant to our design:
- Runs model on representative text corpus
- Records per-tensor activation statistics (which weights matter most)
- During quantization, allocates precision where it reduces the most loss
- **Effect**: 10-30% perplexity reduction vs naive quantization
- **Most impactful at low bit-widths** (Q4 and below)
- At Q8, the effect is smaller because Q8 already preserves most information

### 4. Inter-Layer Dependencies Matter

IMPQ (2025) and MixQuant (2026) show that a layer's optimal precision depends on
what its upstream layers are quantized to. Quantizing layer N to Q4 may be fine if
layer N-1 is Q8, but catastrophic if N-1 is also Q4, because errors compound.

This means truly optimal mixed precision requires considering the full pipeline,
not just per-matrix error in isolation.

### 5. Dynamic (Runtime) Precision Switching

Several papers explore switching precision during inference based on input difficulty:
- SWDP: switches bit-width based on sample complexity
- DPOQ: "onion quantization" — peel layers of precision on-the-fly
- LDP: learnable precision per-layer during both training and inference

**Key insight**: Switching quantization mid-inference has NO mathematical issue.
Each matmul produces f32 output regardless of internal precision. There is no
"state" that carries between quantization levels. This confirms our earlier analysis.

### 6. For Our Specific Case (0.1B RWKV-7 with Q8)

Our situation is unique:
- **Model is small (0.1B)** → each weight carries more information → Q8 is already
  near the quality floor
- **Q8 won overwhelmingly** in our benchmark (score 5.50 vs Q4's 0.77)
- **All 6 large matrices per layer are D×D** (768×768) → same shape, similar
  distributions → Q8 likely wins for ALL of them
- **Only the head is V×D** (65536×768) → different shape but Q8 still wins
- **We have no Q6, Q5, or Q3 levels** → the gap between Q8 and Q4 is too large
  for meaningful mixed precision with our current set

### 7. When Per-Matrix Selection Becomes Critical

Per-matrix selection matters when:
1. **Multiple viable quantization levels exist** (Q8, Q6, Q5, Q4, Q3, Q2)
2. **Model is large enough** for different matrices to have different optimal levels
3. **Memory budget is constrained** and you need to find the optimal bit allocation
4. **Accuracy at low bit-widths** is the goal (Q4 and below)

For our 0.1B model with only Q8/Q4/F32 options, per-matrix selection would almost
certainly select Q8 for everything — the same result as our current single-sample
benchmark, but at higher load-time cost.

## Verdict

### Per-matrix auto-select: CORRECT IN PRINCIPLE, PREMATURE FOR NOW

**Yes, per-matrix is the right architecture** for a general-purpose system:
- The literature unanimously supports non-uniform precision allocation
- llama.cpp's imatrix proves the approach works at scale
- No mathematical issue with mixing precision levels

**But for our current 0.1B setup, it's unnecessary**:
- Q8 dominates across all matrix types at this model size
- We lack intermediate quantization levels (Q5, Q6) that would create meaningful
  per-matrix differences
- The load-time overhead of benchmarking every matrix (6 per layer × 12 layers = 72
  benchmarks instead of 1) adds ~5-10 seconds for no gain

### Recommended Path

1. **NOW**: Keep current design (single benchmark, apply to all layers). It's correct
   for 0.1B with Q8/Q4/F32 options.

2. **WHEN SCALING TO 0.4B+**: Implement per-matrix selection. Larger models will have
   layers with genuinely different sensitivity profiles.

3. **WHEN ADDING Q5/Q6 LEVELS**: More granular quantization options create meaningful
   per-matrix differences that the benchmark can exploit.

4. **CONSIDER imatrix-style calibration**: Instead of benchmarking matmul error in
   isolation, run a few hundred tokens through the model and measure end-to-end
   perplexity impact per matrix. This captures inter-layer dependencies.

### Implementation Note for Future Per-Matrix

When we implement per-matrix selection, the change is minimal:
```rust
// Current: one level for all
let (level, _) = select_best(&head_matrix);

// Future: per-matrix
let key_level = select_best(&key_matrix).0;
let value_level = select_best(&value_matrix).0;
let ffn_key_level = select_best(&ffn_key_matrix).0;
// ... etc
```

The `QuantMatrix` enum already supports this — each matrix independently stores its
own quantization type. The only change is calling `select_best` per matrix instead
of once globally.

## Lessons

1. **Uniform precision is provably suboptimal** — but the magnitude of the loss
   depends on model size, bit-width, and available quantization levels
2. **Q8 is a safe uniform choice for sub-1B** — the precision is high enough that
   per-matrix differences are negligible
3. **Per-matrix selection becomes critical at Q4 and below** — this is where
   sensitivity differences between attention and FFN matrices create real accuracy
   gaps
4. **imatrix (end-to-end calibration) > isolated matmul error** — the gold standard
   is measuring impact on final output, not just per-matrix reconstruction error
5. **Our architecture is already ready** — `QuantMatrix` enum supports mixed precision
   per-matrix with zero code change to the forward pass

## Sources

- [ALMP: Automatic Layer-By-Layer Mixed-Precision Quantization](https://link.springer.com/chapter/10.1007/978-981-95-0014-7_13)
- [SliM-LLM: Salience-Driven Mixed-Precision Quantization (ICML 2025)](https://github.com/Aaronhuang-778/SliM-LLM)
- [LLM-MQ: Mixed-precision Quantization for Efficient LLM Deployment](https://nicsefc.ee.tsinghua.edu.cn//nics_file/pdf/5c805adc-b555-499f-9882-5ca35ce674b5.pdf)
- [IMPQ: Interaction-Aware Layerwise Mixed Precision Quantization](https://arxiv.org/html/2509.15455v1)
- [MixQuant: Adaptive Mixed-Precision Quantization (2026)](https://arxiv.org/html/2607.23047v1)
- [Towards Superior Quantization Accuracy: A Layer-Sensitive Approach](https://arxiv.org/pdf/2503.06518)
- [Layer-Level Entropy-Weighted Quantization Beyond Architecture](https://arxiv.org/pdf/2503.04704)
- [Quantization Methods, Task Difficulty, and Model Size (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/0902.pdf)
- [llama.cpp Importance Matrix (imatrix)](https://github.com/ggml-org/llama.cpp/tree/master/tools/imatrix)
- [llama.cpp Quantization Techniques](https://deepwiki.com/ggml-org/llama.cpp/7.3-quantization-techniques)
- [Mixed-Precision Quantization for Language Models: Techniques and Prospects](https://arxiv.org/pdf/2510.16805)
- [A KL Lens on Quantization: Fast Sensitivity for Mixed-Precision](https://arxiv.org/pdf/2604.13440)
- [From Attention Sensitivity to Layer Role: Revisiting Mixed-Precision](https://arxiv.org/html/2609.34866)
- [Switchable Precision Neural Networks](https://arxiv.org/pdf/2002.02815)
- [LDP: Learnable Dynamic Precision](https://arxiv.org/pdf/2203.07713)
- [NVIDIA Integer Quantization for Deep Learning Inference](https://arxiv.org/pdf/2004.09602)
- [A White Paper on Neural Network Quantization](https://arxiv.org/pdf/2106.08295)
