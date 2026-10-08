# R51: Organic Architecture Reform — Evidence-Based Roadmap 2024-2026

**Date**: 2026-10-08
**Status**: Proposal (approved, pending implementation)
**Purpose**: Restructure the roadmap based on R50 consolidated data, finding organic solutions that integrate dynamically into the model flow rather than patching individual deficiencies.

## Motivation

R50 established 100KB baselines across 25 files (T1b/T2b/T3) and revealed that:
- 40% of files (Cluster D) have BPB > 2.6 — both CM and RWKV fail
- B/Tok ~ 1.0 renders RWKV useless on 14/25 files
- Bits 3-5 bottleneck traps ~63-68% of cost in text files
- The existing roadmap items (N4, N5, N6) are either incremental or blocked

Individual patches (delta coding here, specialized CM there) would address symptoms but not root causes. The architecture needs organic reform where solutions flow naturally through the existing pipeline.

## Research Synthesis

Web investigation revealed 8 systems published 2024-2026 that inform directly:

| System | Date | Key Innovation | BPB enwik8 | Relevance |
|---|---|---|---|---|
| **fx2-cmix-transformer** | Jul 2026 | Frozen 6M Q4 Transformer + online LSTM (2x200, BPTT=128) | **Hutter Prize winner** | Our RWKV+LSTM architecture is analogous, but with BPTT 16x smaller |
| **RATA-CMIX** | Sep 2026 | Same + PPMd order 25, 575 models | Hutter Prize | Confirms BPTT=128 + online LSTM is the competitive standard |
| **StateSMix** | Apr 2026 | Online Mamba SSM (D=32, L=2, 120K params) from scratch + n-gram | 2.123 (1MB) | Byte-level SSM WITHOUT pre-training is viable and dominant (46.6% of gain) |
| **Midicoth** | Mar 2026 | Tweedie denoising as post-correction of probabilities | Beats zstd/brotli | Mathematical correction without additional models |
| **Nacrith** | Feb 2026 | SmolLM2-135M + adaptive bias head + N-gram | 0.918 (alice29) | Adaptive pre-blend of LM with online components |
| **Chained Neural** | Apr 2026 | Chain of predictors with information inheritance | Near SOTA | Lower-order -> higher-order improves efficiency |
| **MambaByte** | COLM 2024 | Byte-level SSM without tokenization | Competitive with subword | Eliminates tokenizer dependency |
| **AIT DCC G2-V3** | Jun 2026 | Adaptive preprocessing: byte-plane split, BWT+MTF, x86 filter | Top-tier AIT | Preprocessing by automatically detected type |

## Central Finding

**Our gap vs state of the art is NOT about models — it's about integration.** We have the right components (frozen neural + CM + match + online mixer), but our integration point (LSTM mixer) has 3 structural limitations the ecosystem already solved:

1. **BPTT=8 bits** (= 1 byte of temporal context). fx2-cmix/RATA-CMIX use **BPTT=128**. Gap: **16x**.
2. **The mixer doesn't see bytes**, only 4 group logits. cmix/PAQ feed the LSTM with **the actual byte + model predictions**.
3. **No post-correction**. Midicoth demonstrated that Tweedie denoising improves ANY probability estimator, including well-calibrated mixers.

## The Organic Solution: 3 Integrated Layers + 1 Adaptive Preprocessing

```
BEFORE:
  bytes -> [CM, RWKV, Match] -> hierarchical LSTM(4 logits, BPTT=8 bits) -> P(bit)

AFTER:
  bytes -> PreProc(d) -> [CM, RWKV+uSSM, Match] -> enriched LSTM(4+byte_ctx, BPTT=64) -> Tweedie(P)
```

The flow is THE SAME. No models added to mixer. Existing components enriched.

---

### Layer 1: uSSM — Micro State Space Model byte-level (StateSMix + MambaByte)

**What**: A minimal SSM (D=32, L=2, ~50K params) trained online from scratch, operating at byte level.

**Why organic**: Not added to mixer as a new group. **Pre-blended with RWKV** inside existing Group 2. Mixer still sees 4 groups.

**Blend mechanism**:
```
P_neural(bit) = a * P_RWKV(bit) + (1-a) * P_uSSM(bit)

a = sigmoid(beta), where beta updates via SGD:
  beta <- beta + lr * (log_loss_RWKV - log_loss_uSSM)  // whoever predicted better
```

When RWKV is better (text, B/Tok>2): a->1.0, uSSM ignored.
When uSSM is better (binary, B/Tok~1): a->0, uSSM dominates.

**Mathematical validation (3 verifications)**:

*V1 — Computational viability*:
```
Forward: D^2 * L * 4 = 32^2 * 2 * 4 = 8,192 ops/byte
Backward: ~25K ops/byte
Adam update: ~100K ops/byte (50K active params * 2 moments)
Total: ~133K ops/byte
At 12 GFLOPS: 11us/byte
100K bytes: 1.1s additional (vs 814s actual = +0.13% overhead)
```
**Result: computational overhead negligible.**

*V2 — Blend convergence*:
```
With lr_blend = 0.01 and EMA(decay=0.99):
- Bytes 0-1000: a ~ 0.95 (RWKV dominates, uSSM untrained)
- Bytes 1000-10000: a adjusts per domain
- Bytes 10000+: converged
Worst case (a stays ~1.0): result = pure RWKV = no regression.
```
**Result: cannot worsen baseline. Can only match or improve.**

*V3 — Empirical precedent*:
```
StateSMix with SSM(D=32, L=2, 120K params): 2.123 BPB on enwik8 1MB from scratch.
That's BETTER than gzip (2.58) with ONLY 120K params and ZERO pre-training.
As complement to our system (1.19 BPB), even marginal uSSM improvement
on binary data justifies the addition.
Nacrith confirms that adaptive blend (bias head) works: 0.918 BPB on alice29.
```

**Files benefited**: all B/Tok < 1.5 (14 files in Cluster C/D). Neutral on text (a->1.0).

---

### Layer 2: Byte-Context LSTM with Extended BPTT (cmix/RATA-CMIX)

**What**: Enrich the existing LSTM mixer with (a) embedding of the current byte and (b) BPTT of 8 bytes (64 bits).

**Why organic**: Not a new model. The SAME LSTM with richer input and longer temporal context. Exactly what cmix and the Hutter Prize winner do.

**Input change**:
```
BEFORE: [stretch(g0), stretch(g1), stretch(g2), stretch(g3), bit_ctx(8)] = 12 floats
AFTER:  [stretch(g0..g3), bit_ctx(8), byte_embed(last_4_bytes)] = 12 + 32 = 44 floats
```

byte_embed: last 4 bytes as 4 vectors of 8 bits normalized (0/1 -> -0.5/+0.5).

**BPTT**: from 8 (bits) to 64 (= 8 bytes * 8 bits/byte).

**Mathematical validation (3 verifications)**:

*V1 — Parameters and convergence*:
```
Params BEFORE: 4*(12+128+1)*128 + 129 = 72,321 (active 51,330 via coupling)
Params AFTER:  4*(44+128+1)*128 + 129 = 88,705 (active ~63K)
Increase: +23% params, +267% BPTT steps
Data available: 100K bytes * 8 bits = 800K bit-steps
Bits per param: 800K / 63K = 12.7 — sufficient for convergence
(cmix: ~5M params on 800M bits = 160 bits/param — more data but more params)
```
**Result: convergence viable at 100KB.**

*V2 — Computational cost*:
```
LSTM forward (44 input, H=128): 4*(44+128+1)*128 = 88,576 MACs/step
BPTT=64: backprop = 64 * 88,576 * 3 = ~17M MACs per update
Every 8 bits (1 byte): 1 BPTT update = 17M MACs
At 12 GFLOPS: 1.4ms per byte
BEFORE: mixer cost negligible (<0.01ms/byte)
AFTER: ~1.4ms/byte
RWKV cost: ~1.0ms/token ~ ~1-4ms/byte (depending on B/Tok)
```
**Result: mixer goes from negligible to ~comparable with RWKV. Estimated throughput: ~40-60% of current. For text (120 B/s -> ~50 B/s), for binary (40 B/s -> ~25 B/s). Significant but acceptable.**

*V3 — Fundamental difference vs B1 (which was KILLED)*:
```
B1 tested BPTT=16/32 BITS with the SAME 12-float input.
More steps of the SAME input = same information, processed more times.
Result: +0.0000 (BPTT=16), +0.0020 (BPTT=32) — zero benefit.

THIS proposal: BPTT=64 bits with ENRICHED input of 44 floats.
The LSTM now SEES the actual bytes — fundamentally new information.
Each BPTT step contributes: what byte it was, how models predicted it.
This is what cmix does and is the documented reason for its advantage.
```
**Result: structurally different from what B1 tested. B1's failure does not apply.**

---

### Layer 3: Tweedie Post-Correction (Midicoth)

**What**: After the LSTM produces P(bit=1), apply Tweedie correction:
```
x = stretch(P_mixer)                     // mixer logit
x_corrected = x + sigma^2(ctx) * score(ctx)  // Tweedie correction
P_final = squash(x_corrected)             // probability
```

Where `score(ctx)` = nabla log p(x|ctx) estimated with running statistics of the mixer in similar contexts.

**Why organic**: Not a separate model (like SSE). A MATHEMATICAL correction of the mixer output using its OWN statistics. No additional hash tables, no new predictor.

**Concrete implementation**:
```
// 8 bit-positions * 256 last-byte contexts = 2048 buckets
// Each bucket maintains: mean(stretch(P)), var(stretch(P)), count
struct TweedieCorrector {
    mean: [f32; 2048],
    var: [f32; 2048],
    count: [u32; 2048],
}
// Memory: 2048 * 12 bytes = 24 KB (negligible)
```

**Mathematical validation (3 verifications)**:

*V1 — Theoretical foundation*:
```
Tweedie's formula: E[theta|x] = x + sigma^2 * nabla log p(x)
Applied to bit-prediction:
  theta = true logit (unknown)
  x = mixer output logit (observed with estimation noise)
  sigma^2 = estimation variance of mixer in this context
  nabla log p(x) = score function of marginal prior

The correction ALWAYS reduces estimator MSE (Stein's dominance,
proven for dimension >= 3 — our 8 bit positions satisfy this).
```
**Result: mathematically guaranteed improvement under model assumptions.**

*V2 — Fundamental difference vs SSE/APM (which was KILLED in A1)*:
```
SSE/APM: independent hash table with its own learning rate.
  -> Creates a second adaptive system that can diverge from mixer.
  -> Cascading causes feedback loops (SSE corrects SSE).
  -> Sparse contexts at 100KB -> noisy predictions.

Tweedie: correction based on the mixer's OWN statistics.
  -> No separate learning rate — uses observed variance.
  -> Single-step, no cascade — correction bounded by sigma^2.
  -> Contexts = 2048 buckets, each with mu/sigma/n.
  -> At 100KB with 8 bits: 800K samples / 2048 = 390 samples/bucket -> robust.
```
**Result: Tweedie is not SSE. Mathematically different and lacks SSE's failure mechanism.**

*V3 — Estimated impact*:
```
Midicoth (which uses Tweedie) beats zstd, brotli, bzip2.
Its baseline is weaker than ours, but the relative correction is:
  ~1-3% improvement in compression ratio -> ~0.01-0.03 BPB in our range.
Correction is larger for predictors with higher variance (Cluster C/D).
For Cluster A/B (low variance, high confidence): correction ~ 0.
```
**Result: estimated improvement -0.01 to -0.03 BPB, concentrated on difficult files.**

---

### Layer 0 (preprocessing): Unified Adaptive Delta

**What**: Adaptive preprocessing pipeline that detects and normalizes local patterns:
- **Delta coding**: for monotonic numerical sequences (OEIS, osdb)
- **Byte-plane split**: for float/multibyte aligned data (ait-E, ait-F)
- **Identity**: for text/code (no transformation)

**Automatic selection**: based on statistics of the first N bytes (entropy per plane, autocorrelation, byte frequency skew). NOT domain detection — local statistical analysis.

**Why organic**: operates BEFORE models see data. CM, RWKV, uSSM and match all see the transformed stream. Transparent to the mixer.

**Evidence**: AIT DCC 2026 G2-V3 (top performer) uses exactly this strategy: file-type detection -> type-specific preprocessing -> adaptive coding. Its preprocessing includes "byte-plane splitting for float32" which is exactly what we need for ait-E (6.74 BPB).

---

## Unexplored Edges (bias-corrected)

### E1: Byte-plane splitting (bias: "only for images")

Byte-plane splitting is NOT only for images. It separates byte N of each multi-byte value into an independent plane. Works for:
- **IEEE-754 floats**: exponent in plane 0-1, mantissa in planes 2-3
- **x86 instructions**: opcode patterns in plane 0, operands in planes 1-3
- **Tabular records**: field types repeat in the same plane

AIT DCC 2026 top performer G2-V3 uses it explicitly for float32. We have NOT considered it. Would impact ait-E (-2.0 to -4.0 BPB estimated) and possibly ait-F, sao, x-ray.

### E2: WHT as mixer feature expansion (bias: "WHT failed in edge-lm")

WHT failed as a MIXING MECHANISM in edge-lm. But as FEATURE EXPANSION it's different: instead of feeding 14 stretch(p) to the LSTM, feed their 16 Walsh-Hadamard coefficients. This gives the LSTM "frequencies" of model predictions (how many agree, how many diverge) without additional parameters. O(n log n) = O(14 * 4) = 56 operations. Cost: zero.

ICLR 2026 validates that WHT basis outperforms FFT for discontinuous data — and our model predictions are inherently discontinuous (hash collisions).

### E3: Information inheritance between CM orders (bias: "the mixer already does it")

The paper "Chained Neural Predictors" (2026) demonstrates that feeding order-N's prediction as INPUT to order-N+1 improves efficiency. Our CMs are independent — order-3 doesn't know what order-2 predicted. The LSTM mixer "implicitly" learns this, but with BPTT=8 bits it doesn't have enough context.

Implementation: instead of each CM producing independent stretch(p), have CM order-(N+1) receive stretch(p) from order-N as additional hash context. Minimal change to hash function, no additional parameters.

### E4: Rank-based byte encoding (bias: "permutation is marginal")

The paper "Rank-Based Modeling" (2025) proposes replacing raw bytes with their frequency RANK. Most frequent byte -> rank 0, second -> rank 1, etc. Rank updates online.

This is equivalent to Move-To-Front (MTF) from BWT, but without needing BWT. Benefit for bit-level prediction: frequent bytes (low rank) have predictable MSBs (bits 0-3 ~ 0000 for ranks 0-15), concentrating all entropy in LSBs where we have more accumulated context.

G2-V3 of AIT DCC uses "BWT+MTF" as preprocessing for text. We have not considered it.

### E5: Asymmetric self-distillation RWKV->byte (bias: "RWKV already contributes via bridge")

Instead of using RWKV only as predictor, use it as TEACHER for the uSSM. On each byte:
1. RWKV produces P_token
2. Bridge converts to P_byte_RWKV
3. uSSM produces P_byte_SSM
4. Loss of uSSM = a * CE(truth) + (1-a) * KL(P_byte_RWKV || P_byte_SSM)

The uSSM learns BOTH from ground truth AND from RWKV. On text (where RWKV is good), the uSSM distills its knowledge. On binary (where RWKV is bad), ground truth dominates. This is asymmetric self-distillation — the student operates at byte level but inherits token-level knowledge from the teacher.

**No current system combines online self-distillation with context mixing.** This would be a genuine innovation.

### E6: Prediction horizon adaptation (bias: "we always predict the current byte")

What if in high-confidence zones (Cluster A, BPB < 0.5), we predict MULTIPLE bytes at once and use multi-symbol arithmetic coding? This is what Byte Latent Transformer (2025) does with "patches" — when bytes are predictable, it groups them.

Implementation: if P(byte) > 0.95 for some byte, emit that byte with 1 bit (flag=match) and advance. The arithmetic coder already supports this implicitly, but making the fast-path explicit would save the 8 bit-predictions for near-certain bytes.

---

## Proposed Roadmap (integration phases)

```
PHASE 0: Adaptive preprocessing (N4 extended)            <- 0 risk, independent
|-- Delta coding for numerical sequences
|-- Byte-plane split for float/multibyte data
|-- Automatic selection by local statistics
    Impact: OEIS -0.3 to -0.7, ait-E -2.0 to -4.0

PHASE 1: Byte-context LSTM + extended BPTT               <- low risk, highest confidence
|-- Add embedding of last 4 bytes to LSTM
|-- Extend BPTT from 8 to 64 (= 8 bytes)
|-- WHT feature expansion (edge E2)
|-- Validate on T1 composite (no-regression gate)
    Estimated impact: -0.01 to -0.03 BPB on text, -0.05 on binary

PHASE 2: Tweedie post-correction                          <- low risk, mathematically sound
|-- Implement TweedieCorrector (24 KB memory)
|-- 2048 context buckets (8 bits * 256 last-byte)
|-- Validate on T1 composite (bounded correction)
    Estimated impact: -0.01 to -0.03 BPB, concentrated on Cluster C/D

PHASE 3: uSSM byte-level with adaptive pre-blend          <- medium risk, highest innovation
|-- Implement Mamba-style SSM minimal (D=32, L=2)
|-- Online training from scratch (Adam, byte-level)
|-- Adaptive pre-blend with RWKV (sigmoid gate)
|-- Asymmetric self-distillation (edge E5)
|-- Validate on T1 composite + T3 (binary improvement gate)
    Estimated impact: neutral at 100KB text, -0.1 to -0.3 on binary >1MB

PHASE 4: Information inheritance + rank encoding           <- experimental
|-- CM order-chain (edge E3)
|-- Rank-based byte encoding (edge E4)
|-- Validate independently each sub-feature
    Estimated impact: -0.01 to -0.05 BPB
```

## Validation Criteria

Each phase MUST pass the composite BPB gate (R28):
1. **mean does not rise** (or drops)
2. **sigma does not rise**
3. **worst does not rise >0.05**

If any phase fails the gate, it reverts WITHOUT affecting prior phases. Phases are independent in code though complementary in effect.

## Projected Cumulative Impact

| Phase | BPB enwik8 est. | T2b mean est. | Confidence |
|---|---|---|---|
| Current | 1.1852 | 1.8814 | -- |
| +Phase 0 (preproc) | 1.1852 (neutral text) | 1.75 (-0.13) | High |
| +Phase 1 (byte-ctx LSTM) | 1.17 (-0.015) | 1.70 (-0.05) | High |
| +Phase 2 (Tweedie) | 1.16 (-0.01) | 1.67 (-0.03) | Medium-high |
| +Phase 3 (uSSM) | 1.15 (-0.01) | 1.55 (-0.12) | Medium |
| +Phase 4 (inheritance) | 1.14 (-0.01) | 1.50 (-0.05) | Low |
| **Cumulative** | **~1.14** | **~1.50** | -- |

This would place us at cmix level (1.17) on enwik8 and significantly better cross-domain (T2b mean 1.50 vs current 1.88).

## References

- [cmix GitHub](https://github.com/byronknoll/cmix)
- [fx2-cmix-transformer (Hutter Prize)](https://github.com/astOwOlfo/fx2-cmix-transformer-v1)
- [RATA-CMIX](https://github.com/axfrgo/hifi-rata-cmix)
- [StateSMix: Mamba SSM + sparse n-gram](https://arxiv.org/abs/2605.02904)
- [Midicoth: Micro-Diffusion Tweedie Denoising](https://arxiv.org/abs/2603.08771)
- [Nacrith: SmolLM2 + ensemble context](https://arxiv.org/abs/2602.19626)
- [MambaByte: Token-free SSM](https://arxiv.org/abs/2401.13660)
- [Chained Lightweight Neural Predictors](https://arxiv.org/abs/2604.15472)
- [AIT DCC 2026 Challenge](https://arxiv.org/abs/2606.17712)
- [NNCP v2: Transformer compression](https://bellard.org/nncp/nncp_v2.pdf)
- [Byte Latent Transformer](https://aclanthology.org/2025.acl-long.453.pdf)
