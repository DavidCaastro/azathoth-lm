# R26: Dynamic MoE Architecture & Direct Weight Manipulation

**Date**: 2026-10-06
**Status**: Complete — Research & design (no implementation)
**Purpose**: Investigate a dynamic Mixture-of-Experts architecture for azathoth-lm
that applies components selectively based on data characteristics, and explore
direct weight manipulation techniques to create domain-specialized RWKV variants
without GPU retraining.

## Motivation

Silesia Corpus evaluation (R25) revealed a fundamental divide:

| Regime | Mean BPB | B/s | RWKV contribution |
|---|---|---|---|
| Text-like (6 files) | 1.02 | 82 | Strong — trained on text/code |
| Binary (6 files) | 3.56 | 31 | Negligible — tokenizer mismatch |

The system uses identical components with identical configuration for fundamentally
different data types, paying the computational cost of the worst case always. RWKV-7
World costs 97% of compute but adds nothing on binary data. The match model is 4%
hit rate on x-ray but 98% on xml.

## Part 1: Dynamic MoE Architecture

### State of the Art

#### OmniZip (CVPR 2026) — RWKV-7 + MoE routing

The most directly relevant system. Uses RWKV-7 as backbone with MoE integrated at
two levels:

1. **Time Mixing MoE**: Replaces the V projection layer with 4 experts (top-2
   sparse routing). A learnable router assigns tokens to specialized experts by
   modality, enabling modality-aware context modeling.

2. **Feedforward MoE**: Replaces the standard MLP with MoE-based module. Each
   expert is a small MLP, allowing modality-specific nonlinear representations.

Model sizes: 4.8M (s), 38M (m), 152M (l). Supports 7 modalities (image, text,
speech, tactile, gene, database). ~1 MB/s on MacBook CPU.

**Key insight**: OmniZip proves RWKV-7 + MoE routing works for universality.
But requires training from scratch with MoE integrated — not applicable to our
pre-trained checkpoint without GPU.

Ref: https://arxiv.org/html/2602.22286v1

#### DualComp (2025) — Dual-modality with MoE routing

- Modality-routing MoE for text vs image
- Modality-unified tokenization (reversible transform to tokens)
- ~200 KB/s on CPU, 1.107 bpb text, 2.834 bpb images
- ~90x faster than Llama3-8B for text compression

**Key insight**: Modality-unified tokenization + routing is viable even without
enormous models. The tokenizer matters as much as the model.

Ref: https://arxiv.org/pdf/2505.16256

#### cmix / hifi-rata-cmix — Gold standard of model diversity

- cmix v21: 2,077 independent models of diverse types
- hifi-rata-cmix: 575 mixed models + 6M Q4 transformer + 2x200 online LSTM
- PAQ8px: 58 context models + 8 neural mixers + 3-stage APM cascade

Model types in PAQ8px/cmix include: context (orders 0-14), indirect context,
match, sparse, word, text, executable, image, DMC, PPM, byte, record, distance,
linear prediction, and adaptive prediction models.

**Key insight**: Diversity of model TYPES matters more than quantity of same type.
cmix groups models by function: exact-match, indirect context, word-level, sparse,
format-specific. The mixer handles weighting.

Ref: https://www.byronknoll.com/cmix.html, https://github.com/axfrgo/hifi-rata-cmix

#### MambaByte (353M) — Byte-level without tokenizer

- Mamba SSM trained directly on bytes (vocab 256, no tokenizer)
- 0.93 BPB on PG19 (353M params, 8K context)
- Significantly more robust to text corruption and synthetic noise
- Checkpoints available: HuggingFace `JunxiongWang/MambaByte_*`

**Key insight**: A byte-level model is inherently better for binary data — no
tokenizer mismatch (1 byte = 1 token always). MambaByte is the natural complement
to RWKV-7 World.

Ref: https://arxiv.org/pdf/2401.13660v2

#### Fast Byte Latent Transformer (ICML 2026)

- Byte-level with "latent patches" compressing byte sequences into dense repr
- 50-92% reduction in memory-bandwidth cost vs byte-level pure
- BLT-D (diffusion) generates multiple bytes per step
- Implementation in C and CUDA

Ref: https://arxiv.org/pdf/2605.08044

#### StateSMix — Online pure (no pre-training)

- Mamba SSM (120K params) + 9 n-gram hash tables, trained online from scratch
- 2.12 BPB on enwik8 — worse than us (1.22) but no pre-training
- Pure C + AVX2, ~2000 tokens/s

**Key insight**: Online-only ceiling is ~2.0-2.1 BPB. Pre-training essential for <1.5.

Ref: https://arxiv.org/pdf/2605.02904

### Proposed Architecture

```
Input byte stream
     |
     +---> [Regime Detector] <-- entropy, ASCII ratio, byte histogram
     |         |                 (sliding window 256 bytes, online)
     |         |
     |         +-- TEXT regime (entropy < 5.0, ASCII > 80%)
     |         |     -> RWKV-7 World (ACTIVE)
     |         |     -> CM orders 0-8 (ACTIVE)
     |         |     -> Match model (ACTIVE)
     |         |     -> Word-level CM (NEW, ACTIVE)
     |         |
     |         +-- BINARY regime (entropy > 5.0 or ASCII < 50%)
     |         |     -> RWKV-7 World (BYPASS -- saves 97% compute)
     |         |     -> MambaByte 353M (ACTIVE -- byte-level, robust)
     |         |     -> CM orders 0-8 (ACTIVE)
     |         |     -> Match model (ACTIVE, extended tables)
     |         |     -> Binary CM (NEW: opcodes, headers)
     |         |
     |         +-- STRUCTURED regime (entropy < 3.0, high repetition)
     |               -> RWKV-7 World (ACTIVE if tokens match)
     |               -> CM orders 0-16 (ACTIVE, more orders)
     |               -> Match model (ACTIVE, lengths up to 256)
     |
     +---> [Hierarchical LSTM mixer] <-- receives active expert outputs
               |
               +---> Final P(bit=1) -> arithmetic coder
```

### Design Principles

1. **Regime detection, not domain detection**: We detect statistical properties
   (entropy, ASCII ratio, byte histogram), not file formats. Adaptation is online
   and data-driven, not format-specific. This respects the "no domain detection"
   principle from BENCHMARKS.md.

2. **Gating by cost**: The detector optimizes both quality AND compute. If RWKV-7
   costs 97% of compute and contributes nothing on binary data, bypassing it is a
   ~30x speedup for free.

3. **Complementary experts**: RWKV-7 World (text) and MambaByte (bytes) cover
   different domains. They complement, not compete.

4. **The mixer already adapts**: Our hierarchical LSTM mixer already learns online
   which models to weight down. Explicit routing is a compute optimization, not a
   quality one — the mixer would arrive at the same weighting but paying the cost
   of all forward passes.

### Candidate Neural Pillars

| Model | Params | Type | Vocab | Checkpoints | RAM Q8 | Strength |
|---|---|---|---|---|---|---|
| RWKV-7 0.1B World | 100M | SSM/RNN | 65K | BlinkDL | ~130 MB | Text, code, multilingual |
| MambaByte 353M | 353M | SSM | 256 | JunxiongWang | ~360 MB | Byte-level, noise-robust |
| SmolLM2 135M | 135M | Transformer | 49K | HuggingFace | ~140 MB | General text (Nacrith uses this) |
| Mamba-130M | 130M | SSM | 50K | state-spaces | ~135 MB | General, efficient |

### RAM Budget

```
Component              RAM (Q8)     Active in
---------------------------------------------
RWKV-7 0.1B World       130 MB      TEXT, STRUCTURED
MambaByte 353M Q8       ~360 MB     BINARY
CM 9 orders              78 MB      ALWAYS
Match model              32 MB      ALWAYS
Word CM (new)            ~20 MB     TEXT
Binary CM (new)          ~40 MB     BINARY
LSTM mixer               <1 MB      ALWAYS
---------------------------------------------
TOTAL peak              ~661 MB     (all active)
TOTAL text regime       ~261 MB     (no MambaByte)
TOTAL binary regime     ~511 MB     (no RWKV World)
```

Fits comfortably in 32 GB. Even with both models loaded simultaneously.

### Implementation Phases

| Phase | Action | Impact | Effort |
|---|---|---|---|
| A | Regime detector (entropy + ASCII, 256-byte window) | Enables routing | Low |
| B | RWKV bypass in binary regime (CM-only) | -0.5 to -1.0 BPB binary, +3x speed | Low |
| C | Integrate MambaByte 353M as second pillar | -0.5 to -2.0 BPB binary | High |
| D | More CM models (indirect, sparse, word, binary) | -0.05 to -0.15 global | Medium |
| E | Dynamic gating with LSTM (learned vs heuristic) | Refinement | Medium |

Phase B alone — without adding any new model — could improve binary domains
significantly. If osdb drops from 4.40 to ~2.5 (CM standalone) with 3x speed
boost, that's an immediate win.

---

## Part 2: Direct Weight Manipulation Without Retraining

### The Core Question

Can we create domain-specialized RWKV variants by mathematically manipulating
weights based on our evaluation metrics, without gradient-based retraining?

### Research Findings: 6 Technique Families

#### 1. Neural Thickets + RandOpt (ICML 2026) — MOST PROMISING

**Discovery**: Around pre-trained weights exists a dense "thicket" of specialized
solutions. Perturbing weights with random Gaussian noise and evaluating variants
generates specialized experts without any gradient computation.

**Algorithm**:
```
For each variant i = 1..N:
  W_i = W_base + epsilon_i    (epsilon ~ N(0, sigma^2))
  score_i = eval(W_i, data_subset)
Select top-K by score
Ensemble: prediction = average/vote of top-K
```

**Why it applies to us**:
- We already have eval: `hybrid-eval --log` measures per-bit, per-domain BPB
- Could generate 50-100 RWKV variants, eval each on 1KB binary data (~30s each)
- Select variants that best predict binary patterns
- No backprop, no gradients, only forward passes + perturbation
- Caveat: works better at larger scale. At 100M params, thicket density may be
  lower than at 7B. Needs empirical validation.

**Cost**: 100 variants x 30s = ~50 min. Feasible on CPU.

Ref: https://arxiv.org/abs/2603.12228

#### 2. Task Arithmetic (Editing Models with Task Arithmetic)

**Concept**: A "task vector" is tau = W_finetuned - W_pretrained. Given a
fine-tuned model, extract tau and add/subtract it to change capabilities.

```
tau_binary = W_binary_expert - W_base
W_new = W_base + alpha * tau_binary
```

**For us**: Requires W_finetuned (needs GPU fine-tuning first). BUT if we find
a community RWKV-7 fine-tuned on binary data, we could extract tau and apply
it without GPU.

**Variant without fine-tuning**: Approximate tau analytically. If byte-token
embeddings (0-255) are the weak point, we could move them in directions that
reduce BPB, computed via finite differences.

Ref: https://arxiv.org/pdf/2212.04089

#### 3. ROME / MEMIT — Surgical Association Editing

**Concept**: Identify which layer/neurons are responsible for a prediction,
apply a rank-one update to MLP weights.

```
W_new = W_old + Delta    where Delta = (v_new - v_old) * k* / (C * k*)
```

**For compression**: Instead of editing "Paris is the capital of France", we'd
edit "after 0xFF 0xFE, the distribution should be X". But this is far more
complex — not discrete factual associations but continuous probability
distributions over 65K tokens.

**Verdict**: Elegant but NOT directly applicable. ROME works for discrete
factual knowledge, not probability distribution recalibration.

Ref: https://arxiv.org/pdf/2202.05262

#### 4. SVD Weight Surgery — Spectral Direction Editing

**Concept**: Decompose W = U Sigma V^T. Each singular vector captures a
"semantic direction". We can prune, amplify, or rotate.

**Application to RWKV embedding matrix** (65536 x 768):
```
1. SVD of E = U Sigma V^T
2. Identify singular vectors activated by byte-tokens (0-255)
3. Identify singular vectors activated by text-tokens (256-65535)
4. Amplify byte vectors: sigma_byte *= alpha
5. Optionally attenuate text: sigma_text *= beta (beta < 1)
```

This redistributes representational capacity toward bytes without touching
architecture or retraining. SVD of 65536x768 takes <1s.

**Risk**: Depends on whether singular vectors align cleanly with byte/text
dichotomy. Needs empirical verification.

#### 5. Embedding Table Surgery — Most Direct

**Concept**: The embedding table E has 65536 rows. Rows 0-255 correspond to
individual byte tokens. We can directly modify these 256 vectors.

**Methods**:

a) **Neighbor interpolation**: If byte 0x41 ('A') has a good embedding but
   0xC0 (binary byte) doesn't, interpolate from contextually similar bytes.

b) **Output calibration**: Measure average output distribution when input is
   binary bytes. If biased toward text (it is), correct input embeddings to
   compensate.

c) **Least-squares fit**: For each byte b, we have data on what distribution
   P(next|context) RWKV produces vs what CM produces. Solve:
   ```
   E_b_new = argmin_e ||f(e, state) - P_correct||^2
   ```
   Convex if we linearize f. Solvable analytically.

#### 6. Heritage: edge-lm Analytical Weights

From our own heritage.md: "Analytical weights (SVD/PPMI): 90% of weights are
computable without backprop."

edge-lm demonstrated that for a WHT model, most weights can be derived
analytically from corpus statistics. The question is whether this scales to
RWKV-7's 100M params with nonlinear recurrence.

**Key difference**: edge-lm had 400K params and simple WHT. RWKV-7 has 100M
with temporal recurrence. Modifying internal weights (not embeddings) causes
cascading effects through the temporal state — much harder to predict.

### Feasibility Matrix

| Technique | No GPU? | No fine-tune? | Est. impact | Risk | Priority |
|---|---|---|---|---|---|
| RandOpt (thickets) | YES | YES | Medium-high | Medium (100M may be small) | **1** |
| Embedding surgery | YES | YES | Medium | Low (only 256 vectors) | **2** |
| SVD spectral edit | YES | YES | Low-medium | Medium (alignment dependent) | **3** |
| Task arithmetic | YES | NO (need tau) | High | Low (if tau exists) | **4** |
| ROME/MEMIT | YES | YES | Low | High (wrong use case) | Skip |

### Recommended Exploration Path

1. **RandOpt first**: Generate N perturbed RWKV copies, eval on Silesia binary
   subset. If thicket exists at 100M scale, we get domain-specialized variants
   for free. ~50 min experiment.

2. **Embedding surgery second**: Analyze byte-token embeddings (0-255) vs
   text-token embeddings. Measure representational capacity allocated to bytes.
   Modify surgically if imbalanced. ~10 min analysis.

3. **SVD of embedding matrix**: Understand if singular vectors separate cleanly
   between byte and text domains. If yes, amplify byte directions. ~5 min.

4. **Task arithmetic if available**: Search HuggingFace for RWKV-7 0.1B
   fine-tuned on binary/mixed data. Extract tau, apply. ~30 min.

---

## Part 3: Connections and Synthesis

### The Unified Vision

The user's insight connects three ideas into one:

1. **Dynamic MoE** (Part 1): Don't use all components equally — route by regime
2. **Weight manipulation** (Part 2): Create specialized variants without training
3. **Hardware efficiency**: Only activate what helps, bypass what doesn't

These three combine into a system where:
- Multiple RWKV variants exist (base + RandOpt-derived specialists)
- A regime detector selects which variant to activate
- CM + match model always run (cheap, universal)
- The LSTM mixer weighs everything adaptively
- Total compute is LOWER than current (bypass saves 97% when RWKV doesn't help)

### Why Not Just Add More CM Models?

Adding CM models (orders 9-16, indirect, sparse, word) would help but has
diminishing returns. Heritage documents this: "diminishing returns after ~50
models." cmix has 2077 models but still uses neural predictors because CM alone
has a ceiling (~1.25-1.35 BPB, PAQ8px territory).

The neural predictor is the multiplicative amplifier. CM provides diverse inputs;
the neural predictor provides the generalization that makes them useful. A BAD
neural predictor (RWKV on binary) actively hurts by confusing the mixer. A GOOD
neural predictor (MambaByte on binary, RWKV on text) amplifies CM's contributions.

### Risk Assessment

| Risk | Mitigation |
|---|---|
| RandOpt fails at 100M scale | Cost is ~50 min. Kill criteria: if no variant beats baseline by >0.01, stop |
| MambaByte integration is complex | Start with Phase B (RWKV bypass) which needs no new model |
| Regime detector misclassifies | Use soft routing (blend, don't switch) + let LSTM mixer correct |
| Multiple models slow down | Only one neural model active per regime — faster than current |

## Key References

- OmniZip (CVPR 2026): https://arxiv.org/html/2602.22286v1
- DualComp: https://arxiv.org/pdf/2505.16256
- Nacrith: https://arxiv.org/pdf/2602.19626
- MambaByte: https://arxiv.org/pdf/2401.13660v2
- MambaByte checkpoints: https://huggingface.co/collections/JunxiongWang/mambabyte-66de59f9ecc44bd637946442
- Fast BLT (ICML 2026): https://arxiv.org/pdf/2605.08044
- StateSMix: https://arxiv.org/pdf/2605.02904
- cmix: https://www.byronknoll.com/cmix.html
- hifi-rata-cmix: https://github.com/axfrgo/hifi-rata-cmix
- fx2-cmix: https://github.com/kaitz/fx2-cmix
- PAQ8px: https://github.com/Dark1510/paq8px
- Neural Thickets (ICML 2026): https://arxiv.org/abs/2603.12228
- RandOpt analysis: https://kindxiaoming.github.io/blog/2026/randopt/
- Modular Norm RandOpt: https://arxiv.org/html/2609.25745v2
- Task Arithmetic: https://arxiv.org/pdf/2212.04089
- ROME: https://arxiv.org/pdf/2202.05262
- SVD of Transformer weights: https://www.lesswrong.com/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight
- Runtime Dynamic MoE Compression (ICLR 2026): https://timdettmers.com/papers/runtime-dynamic-compression.pdf
- RWKV-7 Goose: https://arxiv.org/pdf/2503.14456
- A Survey of RWKV: https://arxiv.org/pdf/2412.14847
- Mamba-130M: https://huggingface.co/state-spaces/mamba-130m-hf
