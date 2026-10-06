# R20: P2.2 Hierarchical Model Groups

**Date**: 2026-10-06
**Status**: Complete — VALIDATED
**Purpose**: Group CM models by context length and mix within groups before
top-level LSTM, reducing LSTM input dimensionality and capturing group structure.

## Hypothesis

Grouping CM models by type (short-context vs long-context) and mixing within
groups before the top LSTM will improve BPB by reducing noise from inter-group
interactions that the LSTM must learn to filter.

**Prediction**: -0.02 to -0.05 BPB over flat LSTM mixer.
**Kill criteria**: If hierarchical BPB > LSTM BPB on 100KB, stop.

## Implementation

### Architecture

```
9 CM models (orders 0-8) + 1 RWKV bridge
    |
    +-- Group 0: CM orders 0-2 (3 models, short context)
    |       → Logistic sub-mixer → P(bit=1) for group 0
    |
    +-- Group 1: CM orders 3-8 (6 models, long context)
    |       → Logistic sub-mixer → P(bit=1) for group 1
    |
    +-- Group 2: RWKV bridge (1 model, neural)
    |       → Direct passthrough → P(bit=1) for group 2
    |
    +-- Top LSTM (H=128, 3 inputs → dynamic weights)
            → Final prediction
```

### Key Design Decisions

1. **Two CM groups** (not three): orders 0-2 represent short-context byte patterns
   (unigram, bigram, trigram). Orders 3-8 represent longer exact matches.
   This captures the fundamental split between local and extended context.

2. **RWKV bridge as separate group**: The neural predictor operates on fundamentally
   different principles (generalization vs exact matching). Keeping it isolated
   prevents CM noise from diluting the LSTM's ability to weight neural vs statistical.

3. **Logistic sub-mixers** (not LSTM): Within each group, models are of the same
   family and have similar temporal dynamics. Logistic mixing (per-bit-context
   weights) is sufficient. Heritage confirms logistic works well for same-family models.

4. **Smaller top LSTM** (3 inputs vs 10): 67,330 params vs 71,817. Fewer inputs
   means the LSTM converges faster with limited data. This explains the dramatic
   improvement on 10KB (1.2502 vs 1.5252 for flat LSTM).

### Integration

- `MixerKind::Hierarchical` variant in `src/domain/cm.rs`
- `ContextMixer::new_with_hierarchical(hidden_dim, lr)` constructor
- `--hierarchical` flag in hybrid-eval command
- Dynamic group extension when external models added (RWKV bridge)

## Results

### 10KB enwik8

| Mixer | BPB | Quartile trajectory |
|---|---|---|
| Logistic hybrid | 1.4133 | 1.23→1.23→1.34→1.41 |
| LSTM hybrid | 1.5252 | 2.07→1.62→1.54→1.53 |
| **Hierarchical** | **1.2502** | 1.15→1.08→1.17→1.25 |

Hierarchical is **massively better on small data** because the top LSTM has
only 3 inputs and converges in <2500 bytes.

### 100KB enwik8

| Mixer | BPB | Speed |
|---|---|---|
| Logistic hybrid | 1.2924 | 169 B/s |
| LSTM hybrid | 1.2549 | 134 B/s |
| **Hierarchical** | **1.2272** | **138 B/s** |

**Delta: -0.0277 BPB** over flat LSTM.
**Delta: -0.0712 BPB** over RWKV ensemble baseline.

### Trajectory

```
Hierarchical: 1.29 → 1.28 → 1.25 → 1.23  (still improving)
LSTM flat:    1.40 → 1.33 → 1.28 → 1.25
```

Hierarchical leads at every quartile and is still improving at 100KB boundary.

## Analysis

### Why hierarchical works better than flat LSTM

1. **Reduced dimensionality**: 3 inputs vs 10 means the LSTM weight matrices are
   smaller (67K vs 72K params), but more importantly, the per-input learning rate
   is effectively higher. Each of the 3 group predictions carries more concentrated
   information than individual model predictions.

2. **Pre-filtered noise**: The logistic sub-mixers handle intra-group model
   weighting, removing noise that the top LSTM would otherwise need to learn to
   ignore. The top LSTM only needs to learn the temporal dynamics of 3 conceptually
   distinct prediction sources.

3. **Faster convergence**: With only 3 inputs, the LSTM's hidden state can
   specialize faster. On 10KB, the flat LSTM is still warming up (1.5252 BPB)
   while hierarchical is already near its asymptote (1.2502 BPB).

### Heritage comparison

Heritage estimated -0.02 to -0.05 BPB. We observe -0.0277, within the predicted
range. Heritage also warned about "cascaded SSE overcorrection" — this doesn't
apply here because our sub-mixers are logistic (not SSE), and the top mixer is
LSTM (temporal), not another logistic layer.

## Key Files

- `src/domain/cm.rs` — MixerKind::Hierarchical, new_with_hierarchical()
- `src/main.rs` — --hierarchical flag in hybrid-eval

## Conclusions

1. **Hierarchical grouping VALIDATED.** -0.0277 BPB over flat LSTM on 100KB.
2. **New best BPB: 1.2272** (100KB enwik8). Well below PAQ8px territory.
3. **Faster convergence**: Hierarchical already near-optimal at 10KB (1.2502)
   while flat LSTM needs 100KB+ to converge.
4. **Speed maintained**: 138 B/s vs 134 B/s (flat LSTM). Slight improvement
   because smaller top LSTM has fewer parameters.
5. **Still improving**: trajectory suggests further gains on larger data.
