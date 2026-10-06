# R34: Roadmap Audit — 5-Thread Exhaustive Research

**Date**: 2026-10-06
**Status**: Complete
**Purpose**: Validate or invalidate every roadmap item using exhaustive web research.

## Methodology

Five parallel research threads covering:
1. APM/SSE state of the art
2. CM scaling effectiveness
3. LSTM mixer improvements and alternatives
4. Novel compression techniques 2025-2026
5. SA-PPM and suffix-based compression

Total: ~200 web searches, 5 structured reports, cross-referenced findings.

## Executive Summary

**The roadmap was fundamentally misordered.** The #1 gap vs competition is not
CM model count (A1) or SA-PPM (D1) — it is **LSTM mixer depth** (old B1).

| Item | Old Priority | New Priority | Old Estimate | Revised Estimate |
|---|---|---|---|---|
| CM scaling 9→25+ | **Tier A (#1)** | **KILLED** | -0.05 to -0.10 | -0.01 to -0.03 |
| APM/SSE | Tier A (#2) | Tier A | -0.01 to -0.04 | -0.005 to -0.020 |
| LSTM improvements | **Tier B** | **Tier S (#1)** | -0.01 to -0.03 | **-0.02 to -0.10** |
| Hedge mixer | Tier B | **KILLED** | -0.005 to -0.02 | Neutral/negative |
| ISSE chains | Tier C | Tier B | -0.02 to -0.04 | -0.01 to -0.025 |
| Micro-diffusion | Tier C | Tier A | -0.01 to -0.03 | -0.01 to -0.03 (confirmed) |
| SA-PPM | **Tier D (#1)** | **KILLED** | **-0.10 to -0.30** | -0.01 to -0.05 |

Three items NOT in the old roadmap should be added:
- **WordModel** (every top compressor uses it, we don't have it)
- **Match model multi-input** (replaces SA-PPM at 5% effort)
- **WRT preprocessing** (fx2-cmix Hutter Prize winner uses it)

## Thread 1: CM Scaling Effectiveness

### Key Findings

**Model count is not the bottleneck.** Evidence:

1. **R33 empirical result**: Adding 3 models (sparse + ICM) gave +0.0027 BPB
   (neutral) on enwik8 100KB. The estimate of -0.05 to -0.10 was 5x optimistic.

2. **Midicoth ablation** (March 2026): On enwik8, Match model = 5.56%,
   Word model = 0.75%, higher-order = 2.49%, Tweedie = 2.77%. Most models
   contribute marginally. A small core drives the vast majority.

3. **FP8 (Fast PAQ8)**: Removed majority of paq8px models, achieved only
   marginally worse compression while being 8x faster.

4. **G9-V1**: 10 models with cascaded mixers = competitive at AIT 2026.
   Quality and diversity > quantity.

5. **Gleipnir's 27 models**: Only ~15 are text-relevant. The critical ones
   beyond our set are **WordModel** (case-folded) and **word-pair context**.

### WordModel

The ONE model we're missing that matters. Every top compressor has it.

- Case-folds words: "The", "the", "THE" share statistics
- Gleipnir: "prevents halving the evidence behind every word context"
- Word-pair patterns ("of the", "in the") span boundaries that order models waste hash space on
- Midicoth ablation: 0.75-1.06% improvement after PPM+Match
- Our RWKV partially captures word semantics via tokenization, but CM WordModel
  helps during early bytes and for rare/OOV words

### Verdict on A1

**KILL raw CM scaling to 25+. Replace with targeted WordModel addition.**

## Thread 2: APM/SSE State of the Art

### Key Findings

1. **cmix uses SSE after LSTM** — empirical proof it's not redundant
2. **paq8px v217**: 3-layer mixer → 2 SSE stages with distinct contexts
3. **Gleipnir**: 11 ISSE stages + 6 SSE stages. ISSE is more expressive than APM.
4. **Heritage SSE failure** was context-reuse, not fundamental. Solved by using
   distinct contexts per stage (confirmed by cbloom analysis).
5. **Nacrith**: No SSE/APM — bias head serves similar function

### ISSE Mechanism (Gleipnir)

```
p0 = stretch(order-1 prediction)
for each stage k:
  s = bit_history_state[context_k]
  w = IW[k][s]  // two int32 weights
  p_k = clamp((w0 * p_{k-1} + w1 * 512) >> 16)
```

Maps context → bit history → weight pair → mix(input, 0.5). More expressive
than table lookup, cheaper than LSTM.

### Tweedie Interference

Midicoth finding: "When combined with SSE, they compete for the same signal
and produce interference rather than synergy." Do NOT combine Tweedie + SSE.
Choose one or the other as post-mixer refinement.

### Verdict on A2

**Keep but lower estimate to -0.005 to -0.020. Start with 1-2 APM stages,
consider ISSE if APM shows gain. Do not combine with Tweedie.**

## Thread 3: LSTM Mixer Improvements

### The Real Gap

```
                    cmix                    azathoth-lm
Layers:             2 × 200                 1 × 128
BPTT:               100                     1
Optimizer:          Adam                    SGD
LayerNorm:          Yes                     No
Coupled gates:      Yes (i=1-f)             No
Gradient clip:      10.0                    No
Params:             ~160K                   ~67K
```

Our LSTM with BPTT=1 is essentially a **feedforward network with persistent
state**. It cannot learn multi-step dependencies like "model X is reliable
after seeing pattern Y two bytes ago."

### Coupled Gates (i = 1 - f)

- Validated by "LSTM: A Search Space Odyssey" (Greff et al., 5400 experiments)
- cmix uses it in production
- -25% params, prevents cell state from growing unboundedly
- Est. impact: neutral to slightly positive BPB, but enables stability for BPTT>1

### LayerNorm

- cmix applies per-gate with learnable gamma/beta
- No temporal dependency → streaming-compatible
- With BPTT=1 benefit is small; critical prerequisite for BPTT>1

### BPTT>1

- Our heritage says "BPTT>1 during eval: +0.10 BPB regression"
- BUT: that was BPTT during eval-only (without training). In compression,
  train+predict is unified — BPTT>1 is safe and standard (cmix uses it).
- BPTT=8 (one full byte): first real temporal learning
- BPTT=100 (cmix-level): requires Adam + LayerNorm + gradient clipping
- Est. impact: -0.02 to -0.10 BPB (the single biggest lever)

### Hedge Mixer (B2 old)

**KILL.** Nacrith ablation: Hedge converges to w_llm ≈ 1.0 when one model
dominates. In our case with comparable-strength models, LSTM's interaction
modeling matters more than Hedge's regret bounds.

### Verdict on B1/B2

**B1 is the new #1 priority (Tier S). B2 is KILLED.**

Implementation order: coupled gates → LayerNorm → Adam → BPTT=8 → scale BPTT.

## Thread 4: Novel Compression 2025-2026

### Major Discoveries

**fx2-cmix-T** (July 2026, Hutter Prize winner):
- 6M Transformer Q4 frozen + 2000+ CM + WRT preprocessing
- 205-symbol alphabet after Word Reducing Transform
- Pre-trained on 8x RTX 5090. Domain-optimized.
- Key lesson: domain-fit matters more than model size

**RATA-CMIX** (September 2026):
- Builds on fx2-cmix-T with entity clustering + lexicographic permutation
- 2x200 online LSTM as parallel expert (generates predictions, not just mixes)

**Midicoth** (March 2026):
- Tweedie empirical Bayes denoising on PPM predictions
- Binary tree byte decomposition aligns with our bit-level architecture
- 2.3-2.7% improvement post-blend. Zero runtime cost.
- 1.753 BPB without any neural network or GPU

**StateSMix** (May 2026):
- Online Mamba SSM + 9 sparse n-gram hash tables
- 2.130 BPB on enwik8 full. Pure C + AVX2.
- N-gram logit-bias mechanism could inform our CM pipeline

**2026 AIT Challenge**:
- xEnc3 beat paq8px in both size and speed (17x faster)
- G9-V1: 10 GLN models with cascaded mixers — competitive

### Items to Add to Roadmap

1. **WRT preprocessing**: fx2-cmix reduces 256→205 symbols. Concentrates
   probability mass. Our system does zero preprocessing.
2. **Tweedie denoising**: Midicoth paper, zero cost, post-blend correction.
3. **Online LSTM expert**: RATA-CMIX style — LSTM as predictor not mixer.

## Thread 5: SA-PPM / Suffix Array

### Key Findings

**SA-PPM should be KILLED.** Evidence:

1. **No top compressor uses suffix arrays.** fx2-cmix, Nacrith, cmix, Gleipnir,
   PAQ8px — all use hash-table CM + match models.

2. **ppmonstr (order-64 suffix tree) = PPMd (order-16 hash) ± 0.003 BPB.**
   Going longer doesn't help. The escape mechanism and statistics estimation
   matter more than context length.

3. **Original estimate was conflated.** The -0.10 to -0.30 range confused the
   gap between standalone PPM (~1.50) and modern CM (~1.20) with suffix-based
   matching gain. That gap is explained by model diversity + LSTM mixing.

4. **Online SA is Very High effort** (~1000-2000 lines of careful code for
   Ukkonen-style incremental construction).

5. **Our match model already gets most of the benefit** (-0.0095 BPB).
   A full SA-PPM might add -0.01 to -0.05 more.

### Alternative: Match Model Improvements

Capture 50-80% of SA-PPM benefit at 5% effort:
- More context lengths: 3, 5, 10, 12, 20, 48 (fill gaps)
- 8-way or 16-way associative hash (current: 4-way)
- Multiple match predictions as separate mixer inputs (like PAQ8px)
- Match extension: verify how far match continues beyond hash context

## Cross-Thread Synthesis

### What Actually Drives Sub-1.0 BPB

Looking at the systems that achieve sub-1.0:

| System | BPB | Neural | CM | Mixer | Key Innovation |
|---|---|---|---|---|---|
| Nacrith | 0.94 | 135M SmolLM2 | ~4 | Hedge | N-gram confidence skip = 30% of gain |
| fx2-cmix-T | 0.97 | 6M TF Q4 | 2000+ | LSTM 2×200 | Domain-trained TF + WRT preprocessing |

Both require either a larger neural model or domain-tuned training (GPU).
Our CPU-only ceiling is ~1.05-1.10 BPB with all improvements.

### Revised Priority Ranking (impact per engineering hour)

1. **Coupled gates** — 30 min code change, -25% LSTM params, enables stability
2. **WordModel** — 2-4h, orthogonal information no other model provides
3. **APM 1-2 stages** — 2-4h, well-understood technique, distinct contexts
4. **LayerNorm** — 2-4h, prerequisite for BPTT>1
5. **Match multi-input** — 1-2h, fill context length gaps
6. **BPTT=8** — 4-8h, needs Adam + grad clip, highest single delta
7. **Tweedie denoising** — 4-8h, zero runtime cost, but cannot combine with APM
8. **Higher-order CM 12/16** — 30 min, small tables, marginal gain

## References

### Papers
- Midicoth: Micro-Diffusion Compression (arXiv:2603.08771, March 2026)
- Nacrith: Neural Lossless Compression (arXiv:2602.19626, Feb 2026)
- StateSMix: Mamba + Sparse N-gram (arXiv:2605.02904, May 2026)
- Chained Lightweight Neural Predictors (arXiv:2604.15472, April 2026)
- 2026 AIT Data Compression Challenge (arXiv:2606.17712)
- LSTM: A Search Space Odyssey (Greff et al., 2015)
- OmniZip (CVPR 2026, arXiv:2602.22286)
- L3TC: RWKV for Lossless Compression (AAAI 2025)

### Systems
- fx2-cmix-T: github.com/kaitz/fx2-cmix (Hutter Prize Jul 2026)
- RATA-CMIX: github.com/axfrgo/hifi-rata-cmix (Sep 2026)
- Nacrith: github.com/st4ck/nacrith
- Midicoth: github.com/robtacconelli/midicoth
- cmix v21: github.com/byronknoll/cmix
- paq8px v217: github.com/hxim/paq8px
- Gleipnir: github.com/ValisSowilo/Gleipnir

### Forums
- encode.su: CM design discussions, PAQ8PX modelling threads
- cbloom rants: Secondary Estimation analysis
- mattmahoney.net: Large Text Compression Benchmark
