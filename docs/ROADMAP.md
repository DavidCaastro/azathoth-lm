# Roadmap — azathoth-lm

**Date**: 2026-10-02
**Baseline**: 1.3238 BPB (100KB enwik8, mixer eta=0.01, lr=0.30, scale=0.5)
**Target**: < 1.0 BPB on enwik8

## Current Position

```
1.32  azathoth-lm   (0.1B RWKV + N-gram + bias + mixer, 100KB eval)
1.27  PAQ8px        (200+ context models)
1.19  NNCP v3       (199M Transformer-XL)
1.17  cmix          (2077 models + LSTM)
1.11  ts_zip        (RWKV-169M v4 Q8, pure LM)
0.97  fx2-cmix      (6M Transformer + 2000+ CM)
0.94  Nacrith       (135M SmolLM2 + CM)
```

Gap to target: ~0.33 BPB. Gap to PAQ8px: ~0.05 BPB.

## Phase 1 — Maximize Current Architecture

Priority: extract remaining BPB from 0.1B RWKV + ensemble without new components.

### P1.1: CDF-24 Arithmetic Coder

- **Impact**: est. -0.05 to -0.10 BPB (eliminates quantization loss in probability → bits)
- **Effort**: Medium (arithmetic coder implementation in Rust)
- **Risk**: Low — well-understood algorithm, no model changes
- **Rationale**: Current BPB is measured via cross-entropy on softmax output.
  Real compression requires mapping probabilities to bit streams via arithmetic coding.
  CDF-24 (24-bit precision) is standard in PAQ8px/cmix. The gap between cross-entropy
  BPB and actual compressed BPB can be 0.05-0.10 depending on tail distribution handling.
- **Heritage**: No prior attempt in analytic-lm or edge-lm. Clean implementation.
- **Kill criteria**: If compressed BPB > cross-entropy BPB + 0.01, implementation is buggy.

### P1.2: N-gram Order Expansion (5-6)

- **Impact**: est. -0.01 to -0.03 BPB
- **Effort**: Low (extend existing TokenNgram, increase hash table)
- **Risk**: Low — RAM cost ~2x per order, but well within 32 GB budget
- **Rationale**: Current orders 1-4 capture short patterns. Orders 5-6 would capture
  common multi-word phrases in enwik8 (article titles, template text, dates).
  N-gram weight is already the dominant mixer component (w_ng=1.85), suggesting
  the N-gram signal is valuable and more of it may help.
- **Heritage**: analytic-lm used match tables up to order 8 as separate components.
- **Kill criteria**: If BPB unchanged on 100KB with orders 5-6 added, revert.

### P1.3: Full enwik8 Benchmark (100MB)

- **Impact**: Official BPB number for comparison with literature
- **Effort**: High wall-clock time (~327h at 85 B/s current throughput)
- **Risk**: None to BPB — but blocks machine for ~2 weeks
- **Rationale**: 100KB eval is a proxy. All published numbers use full enwik8.
  Must run at least once for credible comparison. Can run after P1.4 for speed.
- **Dependencies**: Ideally after P1.4 (confidence skip) to reduce wall time.
- **Mixer eta for 100MB**: est. ~0.001 based on scaling law eta ~ O(1/sqrt(N_tokens)).
  With ~25M tokens: eta = 0.01 * sqrt(25K/25M) = 0.0003. Sweep [0.0001, 0.0003, 0.001].

### P1.4: Confidence Skip

- **Impact**: 2-5x throughput improvement, est. ~0 BPB loss
- **Effort**: Medium (threshold tuning, skip logic in main loop)
- **Risk**: Medium — aggressive skipping degrades BPB
- **Rationale**: When RWKV prediction confidence is very high (top-1 prob > threshold),
  the N-gram and bias head corrections are negligible. Skipping ensemble computation
  for high-confidence tokens saves ~46 ms/tok overhead for those tokens.
  At 85 B/s, full enwik8 takes ~327h. At 2x, ~164h. At 5x, ~65h.
- **Heritage**: Listed in heritage.md Tier 2 as untried. No known failure mode.
- **Kill criteria**: If BPB increases > 0.005 at any skip threshold, too aggressive.

## Phase 2 — New Components

Priority: add new prediction sources to the ensemble.

### P2.1: Context Mixing Models (hash-based)

- **Impact**: est. -0.05 to -0.15 BPB (based on PAQ8px architecture)
- **Effort**: High (implement CM infrastructure from analytic-lm)
- **Risk**: Medium — hash table memory pressure, collision management
- **Rationale**: The biggest architectural gap vs PAQ8px/cmix/Nacrith is the number
  of context models. We have 2 (N-gram + bias head). PAQ8px has 200+. cmix has 2077.
  Even Nacrith has multiple secondary models beyond the LLM.
  heritage.md confirms: multi-order match tables (orders 2-8), recency decay=0.90,
  4-way associative hash all work. Diminishing returns after ~50 models.
- **Heritage**: analytic-lm achieved 1.58 BPB with 54 CM + LSTM. Porting the best
  subset (top ~20 by contribution) could yield significant gains.
- **Kill criteria**: If 10 CM models add < 0.01 BPB improvement, architecture ceiling.

### P2.2: LSTM Mixer

- **Impact**: est. -0.05 to -0.22 BPB (analytic-lm's biggest single win)
- **Effort**: High (LSTM implementation, BPTT, gradient management)
- **Risk**: Medium — BPTT>1 during eval is known to overfit (heritage.md)
- **Rationale**: heritage.md documents LSTM mixing as +0.22 BPB over linear mixers.
  This is the largest known architectural win. Current logistic additive mixing is
  equivalent to a 1-layer linear mixer. An LSTM can capture temporal dependencies
  between component contributions.
- **Heritage**: BPTT>1 during eval = +0.10 BPB (FAIL). Must use BPTT=1 during eval.
  HID=128+ recommended. SGD > Adam for online single-sample.
- **Dependencies**: Benefits most with multiple CM models (P2.1) to mix.
- **Kill criteria**: If LSTM mixer BPB > logistic mixer BPB on 100KB, architecture mismatch.

### P2.3: Byte-level Prediction Path

- **Impact**: est. -0.02 to -0.05 BPB
- **Effort**: Medium (parallel byte-level predictions alongside token-level)
- **Risk**: Low-Medium — token/byte alignment complexity
- **Rationale**: Current architecture is purely token-level. heritage.md shows
  bit-level > byte-level (1.58 vs 1.645 BPB). A byte-level prediction path
  could capture sub-token patterns (XML tags, numbers, punctuation) that
  the World tokenizer handles poorly.
- **Heritage**: analytic-lm was byte-level (bit-level). The token-level shift
  was driven by RWKV's tokenizer. A hybrid could get both benefits.

## Phase 3 — Advanced Techniques

Priority: techniques with higher complexity but potentially large gains.

### P3.1: Domain-Matched Neural Checkpoint

- **Impact**: est. -0.10 to -0.20 BPB
- **Effort**: Very High (fine-tuning infrastructure, GPU access needed)
- **Risk**: High — requires GPU access we don't have locally
- **Rationale**: R05 showed all available RWKV-7 checkpoints >0.1B perform worse
  on enwik8 due to domain mismatch. A checkpoint fine-tuned on Wikipedia/enwik8
  data would dramatically improve base model predictions. ts_zip achieves 1.11 BPB
  with RWKV-169M trained on English text.
- **Options**: Fine-tune 0.1B on enwik8 subset (if GPU available), or find/request
  an English-domain RWKV-7 checkpoint from the community.
- **Kill criteria**: If fine-tuned 0.1B < 1.20 BPB standalone, massive win. If > 1.40, insufficient.

### P3.2: SA-PPM / Suffix Array Predictor

- **Impact**: est. -0.30 to -0.60 BPB (heritage.md Tier 1)
- **Effort**: Very High (suffix array construction, PPM integration)
- **Risk**: High — memory cost, O(n) construction, enwik8 = 100MB suffix array
- **Rationale**: Unifies all context-matching into a single optimal structure.
  PPM alone achieves ~1.50 BPB on enwik8. Combined with neural predictions,
  could be the single largest win available.
- **Heritage**: Listed as Tier 1 in heritage.md, never attempted. est. -0.30 to -0.60.
- **Dependencies**: RAM budget — 100MB enwik8 suffix array needs ~400-800 MB.

### P3.3: Hierarchical/Multi-Stage Mixer

- **Impact**: est. -0.02 to -0.05 BPB
- **Effort**: Medium-High
- **Risk**: Medium — SSE overcorrection documented in heritage.md
- **Rationale**: Group models by type (exact match, neural, statistical) and
  mix within groups before final mixing. Reduces interaction noise.
- **Heritage**: Cascaded SSE always overcorrects. Hierarchical by type is untried.

## Priority Matrix

| Action | Est. Delta BPB | Effort | Risk | Priority |
|---|---|---|---|---|
| P1.1 CDF-24 | -0.05 to -0.10 | Medium | Low | **1 (next)** |
| P1.2 N-gram 5-6 | -0.01 to -0.03 | Low | Low | **2** |
| P1.4 Confidence skip | ~0 BPB, 2-5x speed | Medium | Medium | **3** |
| P2.1 Context models | -0.05 to -0.15 | High | Medium | **4** |
| P2.2 LSTM mixer | -0.05 to -0.22 | High | Medium | **5** |
| P1.3 Full enwik8 | official number | Time | None | **6 (after P1.4)** |
| P3.1 Domain checkpoint | -0.10 to -0.20 | Very High | High | 7 (needs GPU) |
| P3.2 SA-PPM | -0.30 to -0.60 | Very High | High | 8 (research) |
| P2.3 Byte-level path | -0.02 to -0.05 | Medium | Medium | 9 |
| P3.3 Hierarchical mixer | -0.02 to -0.05 | Medium-High | Medium | 10 |

## Projected Trajectory

Optimistic (all Phase 1-2 succeed):
```
1.3238  current
1.27    + CDF-24 (-0.05)
1.25    + N-gram 5-6 (-0.02)
1.15    + CM models (-0.10)
1.00    + LSTM mixer (-0.15)
```

Conservative (Phase 1 only, partial gains):
```
1.3238  current
1.29    + CDF-24 (-0.03)
1.28    + N-gram 5-6 (-0.01)
1.26    + confidence skip (no BPB change, 3x speed)
```

The < 1.0 target requires Phase 2 components (CM + LSTM mixer) or Phase 3
(domain-matched checkpoint or SA-PPM). Phase 1 alone reaches ~1.25-1.28.

## Constraints

- **CPU-only**: i5-1235U, 32 GB DDR5, no GPU
- **RAM budget**: ~16 GB for inference (RWKV ~350 MB, CM hash tables est. ~6-8 GB)
- **Throughput**: 85 B/s current, full enwik8 = ~327h without speed optimization
- **Max 1 heavy task**: concurrent RWKV evaluations cause CPU thrashing
- **Zero external deps**: all code must compile with rustc + stdlib only
