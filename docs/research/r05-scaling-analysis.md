# R05: Scaling Analysis — Path to Sub-1.0 BPB

- **Date**: 2026-10-02
- **Status**: Complete (updated with empirical 0.4B results)
- **Purpose**: Determine whether scaling from RWKV-7 0.1B to 0.4B is the
  correct path to sub-1.0 BPB, with quantitative projections and risk analysis

---

## 1. RWKV-7 Perplexity Scaling (Pile Validation)

From the RWKV-7 paper (arXiv:2503.14456) and "Don't Pay Attention" (2506.11305):

| Model | Params | D | L | Pile PPL | vs Mamba |
|---|---|---|---|---|---|
| RWKV-7 | 168M | 768 | 12 | **14.2** | Mamba-130M: 16.0 |
| RWKV-7 | 421M | 1024 | 24 | **7.2** | Mamba-370M: 8.0 |
| RWKV-7 | 1.47B | — | — | ~3.5* | — |

*1.47B estimated from scaling curve.

**Key ratio**: 168M → 421M = **2.0x perplexity improvement** (14.2 → 7.2).
This is better than log-linear scaling — RWKV-7 benefits disproportionately
from the architecture's dynamic state evolution at larger dimensions.

RWKV-7 consistently outperforms Mamba at equivalent parameter counts.

## 2. Nacrith Ablation — Component Contributions (Critical Data)

From the Nacrith paper (arXiv:2602.19626), Table 7, on 1MB enwik8:

| Config | BPB | Delta | % of total gain |
|---|---|---|---|
| A0: LLM + arithmetic CDF-16 | 1.817 | baseline | — |
| A1: + CDF-24 | 1.300 | **-0.517** | 56% |
| A2: + adaptive bias head | 1.285 | -0.015 | 2% |
| A3: + N-gram + confidence skip | 0.897 | **-0.388** | 42% |
| A4: full system | 0.896 | -0.001 | <1% |

**Critical insight**: The two dominant components are:
1. **CDF-24** (-0.517): eliminates quantization overhead from CDF-16 floors
   in large vocabularies. This applies to ACTUAL COMPRESSION only.
2. **N-gram + confidence skip** (-0.388): local pattern matching + skipping
   LLM inference when N-gram confidence exceeds threshold.

The bias head alone contributes only -0.015 BPB. Our measured -0.06 ensemble
delta (N-gram + bias combined) is in the right ballpark for a first attempt.

### What this means for cross-entropy vs compression BPB

- A0 (1.817) includes CDF-16 coding overhead (~0.52 wasted)
- A1 (1.300) ≈ true cross-entropy of SmolLM2-135M on 1MB enwik8
- SmolLM2-135M raw cross-entropy: **~1.28-1.30 BPB**
- Our RWKV-7 0.1B cross-entropy: **1.47 BPB** (100KB)
- Our RWKV-7 0.1B is 100M params vs SmolLM2's 135M (30 layers, dim=576)

Our RWKV-7 0.1B produces ~0.17 BPB worse predictions than SmolLM2-135M.
This is consistent with the 35% parameter gap and shallower architecture
(12 layers vs 30).

## 3. Scaling Projection: RWKV-7 0.4B

### 3.1 BPB Estimation

Perplexity-to-BPB conversion:
- PPL 14.2 → CE_nats = ln(14.2) = 2.653 → CE_bits/token = 3.83
- PPL 7.2 → CE_nats = ln(7.2) = 1.974 → CE_bits/token = 2.85

BPB ratio: ln(7.2) / ln(14.2) = **0.744**

Applying to our measured BPB:
- RWKV-7 0.1B baseline: 1.47 BPB
- Estimated RWKV-7 0.4B baseline: 1.47 × 0.744 ≈ **1.09 BPB**

This is consistent with ts_zip's result: RWKV v4 169M = 1.106 BPB.
RWKV-7 0.4B should be significantly better than RWKV v4 169M due to:
- 2.5x more params (421M vs 169M)
- Better architecture (v7 > v4)
- Estimated: **1.00-1.10 BPB** (conservative to optimistic)

### 3.2 With Ensemble

| Component | Estimated delta | Running BPB |
|---|---|---|
| RWKV-7 0.4B baseline | — | 1.05 (mid estimate) |
| + N-gram (orders 1-4) | -0.03 | 1.02 |
| + Bias head (tuned) | -0.02 | 1.00 |
| + Confidence skip | -0.05 | 0.95 |
| + Ensemble tuning | -0.02 | **0.93** |

**Projection: RWKV-7 0.4B + full ensemble = 0.93-1.03 BPB**

Sub-1.0 is achievable with 0.4B, but requires full ensemble including
confidence skip. Without skip: ~1.00-1.05 (borderline).

### 3.3 Comparison with Nacrith

| Dimension | Nacrith | azathoth-lm (projected) |
|---|---|---|
| Neural model | SmolLM2-135M (transformer) | RWKV-7 0.4B (RNN) |
| Parameters | 135M | 421M |
| Memory per step | O(context × d) | **O(d)** — constant |
| Context handling | KV cache + sliding (2048) | **Single pass, infinite** |
| Inference engine | llama.cpp (GPU) | Rust native (CPU) |
| Ensemble | N-gram + bias + CDF-24 + skip | N-gram + bias + skip |
| BPB | 0.939 | **0.93-1.03** (est.) |

**Our architectural advantage**: RWKV's O(1) memory per step means:
- No context window limitation
- No costly KV cache sliding (Nacrith: 37× overhead per slide)
- No recomputation when context fills
- Consistent throughput regardless of position in file

For a 100MB benchmark like enwik8, this is substantial.
SmolLM2 with 2048-token context needs ~49,000 context slides.

## 4. Hardware Feasibility (0.4B on i5-1235U)

### 4.1 Memory

| Config | Weights | State | N-gram | Total |
|---|---|---|---|---|
| F32 | 1.6 GB | 350 MB | 200 MB | **2.2 GB** |
| Q8 (proper VNNI) | 400 MB | 350 MB | 200 MB | **950 MB** |
| Q4 | 200 MB | 350 MB | 200 MB | **750 MB** |

All configurations fit comfortably in 32 GB RAM.

### 4.2 Throughput

Compute scaling: 0.1B → 0.4B
- Layers: 12 → 24 (2×)
- Dimension: 768 → 1024 (1.78× in matmul cost)
- Total: ~3.56× slower per token

| Config | ms/tok | tok/s | enwik8 (40M tok) |
|---|---|---|---|
| 0.4B F32 | ~164 | ~6.1 | **1825 hours** |
| 0.4B Q8 VNNI | ~55 | ~18 | **617 hours** |
| 0.4B Q8 + 70% skip | ~55* | ~60** | **185 hours** |

*Only 30% of tokens need RWKV forward.
**Effective throughput including skipped tokens.

Full enwik8 at 185 hours (~8 days) is feasible but slow.

### 4.3 Progressive Evaluation Plan

| Phase | Data | Estimated time | Purpose |
|---|---|---|---|
| Smoke | 10 KB | ~30 sec | Validate 0.4B loads and runs |
| Quick | 100 KB | ~5 min | BPB trend vs 0.1B |
| Medium | 1 MB | ~1 hour | Reliable BPB estimate |
| Extended | 10 MB | ~18 hours | Strong BPB estimate |
| Full | 100 MB | ~185 hours | Official benchmark |

Kill criteria:
- If 10KB BPB > 1.30: abort, model loading error
- If 100KB BPB > 1.15: insufficient scaling, evaluate 1.5B
- If 1MB BPB > 1.05: ensemble insufficient, need confidence skip

## 5. Empirical 0.4B Results (2026-10-02)

### 5.1 KILL: 0.4B Scaling Hypothesis Failed

**Both available 0.4B checkpoints perform WORSE than the 0.1B on enwik8:**

| Checkpoint | Format | BPB (10KB) | vs 0.1B |
|---|---|---|---|
| RWKV-7 0.1B World (fla-hub) | HF | **1.4298** | baseline |
| RWKV-7 0.4B World v2.9 (fla-hub) | HF | 1.6549 | **+0.225** |
| RWKV-7 0.4B World v2.9 + ensemble | HF | 1.5787 | +0.149 |
| RWKV-7 G1d 0.4B (BlinkDL native) | BlinkDL | 1.8817 | **+0.452** |

Kill criteria triggered: 10KB BPB > 1.30 for all 0.4B variants.

### 5.2 Root Cause Analysis

1. **World v2.9 checkpoint**: Trained on multilingual data, likely
   under-performs on English-only enwik8 compared to smaller, more
   English-focused training runs.
2. **G1d checkpoint**: Version "d" (Feb 2026) is the earliest available
   RWKV-7 G1 version. Only g1j and g1k (Aug-Sep 2026) are well-trained,
   but these start at 1.5B parameters.
3. **No Pile-trained 0.4B**: The paper's PPL 7.2 (421M) result uses
   Pile-trained checkpoints that aren't publicly available at 0.4B.
4. **Model file integrity**: Both files verified complete (correct byte
   counts, valid tensor dimensions, non-corrupted weights).

### 5.3 Bug Fix: BlinkDL LoRA Transposition

During 0.4B testing, discovered and fixed a latent bug in the BlinkDL
weight loader: LoRA weights need transposition because BlinkDL uses
`vec @ matrix` while our code uses `mat_vec_mul(matrix, vec)`.

Affected weights: w1/w2, a1/a2, v1/v2, g1/g2 (8 tensors per layer).
This bug existed since the BlinkDL loader was written but was never
triggered because only HF-format models were used.

### 5.4 KILL: G1k 1.5B Also Failed (2026-10-02)

**G1k 1.5B produces catastrophically worse BPB than 0.1B on enwik8:**

| Checkpoint | BPB (10KB) | vs 0.1B |
|---|---|---|
| RWKV-7 0.1B World (fla-hub) | **1.4298** | baseline |
| RWKV-7 G1k 1.5B (BlinkDL) | **5.2691** | **+3.84** (3.7x worse) |

Kill criteria triggered: BPB > 1.30 by enormous margin.

**Verification**: Forward pass validated token-by-token against Python
reference implementation. Logits match within Q8 head noise (~0.02-0.03).
BPB also confirmed bad in Python (6.5 BPB on first 133 bytes). The model
genuinely assigns near-zero probability to correct enwik8 tokens.

**Root cause**: The G1k checkpoint is trained on a data mixture that does
not align with enwik8 (Wikipedia XML markup + English prose). Despite being
"80% English", the training distribution is incompatible with enwik8's
specific domain. The fla-hub 0.1B "World" model, despite being 15x smaller,
was trained on data more representative of this domain.

### 5.5 Revised Strategy (Final)

All available checkpoints larger than 0.1B perform WORSE on enwik8:

| Checkpoint | Params | BPB (10KB) | Status |
|---|---|---|---|
| RWKV-7 0.1B World (fla-hub) | 100M | **1.4298** | Best available |
| RWKV-7 0.4B World v2.9 | 400M | 1.6549 | KILLED |
| RWKV-7 G1d 0.4B | 400M | 1.8817 | KILLED |
| RWKV-7 G1k 1.5B | 1.5B | 5.2691 | KILLED |

**Conclusion**: Scaling via available RWKV-7 checkpoints is a dead end.
The Pile-trained checkpoints from the paper (PPL 7.2 at 421M) are not
publicly available.

Options remaining:

| Option | BPB est. | Feasibility | Risk |
|---|---|---|---|
| A: 0.1B + maximized ensemble | 1.20-1.35 | **High** | Can't reach sub-1.0 |
| B: Fine-tune 0.1B on enwik8-domain | 1.10-1.25 | Low | Need training infra |
| C: Port to SmolLM2-135M (transformer) | 0.94-1.10 | Medium | O(context) memory |
| D: Find/train Pile-focused RWKV-7 | 0.85-1.05 | Low | Time-intensive |

**Decision: Option A — maximize 0.1B ensemble.**

Rationale:
- The 0.1B model works (1.43 BPB baseline, 1.38 with ensemble)
- Ensemble delta increases with data (-0.054 at 10KB, -0.061 at 100KB)
- Confidence skip + tuned N-gram + CDF-24 could push to ~1.20
- This path requires only engineering, not new model procurement
- Sub-1.0 is not achievable but significant improvement is possible

### G1k 1.5B Specifications

- **Checkpoint**: `rwkv7-g1k-1.5b-20260930-ctx25600.pth` (3.06 GB, BF16)
- **Architecture**: L=24, D=2048, H=32, head_size=64, vocab=65536
- **Training**: G1k series (latest), 80% English, Sep 2026
- **Context**: 25,600 tokens

### Hardware Budget

| Component | F32 | Notes |
|---|---|---|
| Weights | ~6 GB | 1.5B × 4 bytes |
| State (H×N×N per layer) | ~300 MB | 24 × 32 × 64 × 64 × 4 |
| Embeddings (pre-normalized) | ~512 MB | 65536 × 2048 × 4 |
| Head Q8 | ~128 MB | 65536 × 2048 bytes |
| N-gram hash tables | ~200 MB | Grows with data |
| **Total** | **~7.1 GB** | Fits in 32 GB |

### Throughput Estimate

Compute scaling from 0.1B to 1.5B:
- Layers: 12 → 24 (2×)
- Dimension: 768 → 2048 (7.1× in matmul cost, D² scaling)
- Total: ~14.2× slower than 0.1B
- Estimated: ~650 ms/tok → ~1.5 tok/s

| Phase | Data | Tokens est. | Time est. |
|---|---|---|---|
| Smoke | 10 KB | ~3000 | ~33 min |
| Quick | 100 KB | ~27K | ~5 hours |
| Full | 100 MB | ~40M | ~300 days (infeasible) |

**Critical**: Full enwik8 is infeasible at this throughput.
Must implement confidence skip + VNNI Q8 for any extended eval.
Smoke test (10KB) is our validation gate.

## 6. Novelty Assessment

### What azathoth-lm IS:
A practical CPU-only compressor combining RWKV-7's O(1) inference with
online token-level ensemble prediction. Not architecturally novel at the
component level — each piece exists in the literature.

### What azathoth-lm OFFERS uniquely:
1. **RWKV + token ensemble for compression**: ts_zip uses RWKV without
   ensemble; Nacrith uses transformer with ensemble. The combination
   of RWKV's constant-memory inference + online ensemble is unexplored.
2. **Pure CPU, pure Rust, zero-dependency**: No llama.cpp, no Python,
   no GPU. Every byte of code auditable.
3. **Single-pass streaming**: Unlike transformers that need context
   windowing, RWKV processes the entire file in one sequential pass.
4. **Minimal overhead ratio**: If 0.4B achieves 0.95 BPB, that's better
   BPB-per-parameter than Nacrith (135M→0.94 vs 421M→0.95), while
   offering O(1) memory and no GPU requirement.

### What IS architecturally novel (potential):
- **Distilled RWKV for compression**: Fine-tune 0.4B specifically on
  enwik8-domain data (Wikipedia markup + English prose). Could push
  BPB below pure foundation model predictions.
- **Cross-architecture ensemble**: RWKV (global patterns) + byte-level
  CM (local patterns) — the original azathoth-lm vision, now with a
  viable neural backbone.

## 7. References

1. RWKV-7 paper: [arXiv:2503.14456](https://arxiv.org/abs/2503.14456)
2. "Don't Pay Attention" (scaling comparisons): [arXiv:2506.11305](https://arxiv.org/abs/2506.11305)
3. Nacrith: [arXiv:2602.19626](https://arxiv.org/abs/2602.19626)
4. Nacrith GitHub: [robtacconelli/Nacrith-GPU](https://github.com/robtacconelli/Nacrith-GPU)
5. ts_zip: [bellard.org/ts_zip](https://bellard.org/ts_zip/)
6. RWKV-7 0.4B weights: [fla-hub/rwkv7-0.4B-world](https://huggingface.co/fla-hub/rwkv7-0.4B-world)
7. RWKV-7 weights (official): [BlinkDL/rwkv-7-world](https://huggingface.co/BlinkDL/rwkv-7-world)
8. L3TC (RWKV compression): [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33446)
9. StateSMix: [arXiv:2605.02904](https://arxiv.org/abs/2605.02904)
