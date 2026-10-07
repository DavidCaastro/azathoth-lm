# R44: C1 — Online LSTM Expert

**Date**: 2026-10-07
**Status**: Complete — KILLED (+0.0127 on 100KB)
**Purpose**: LSTM as independent byte predictor (not mixer), learning online.

## Hypothesis

A small LSTM (emb 32, hidden 64, ~44K params) operating at byte level
will provide genuinely new predictions via online adaptation — a signal
source that RWKV (frozen) cannot provide.

**Prediction**: -0.01 to -0.03 BPB.

**Kill criteria**: Regression > 0.01 on 100KB.

## Implementation

New module `src/domain/lstm_expert.rs`:
- Byte embedding: 256 × 32 = 8,192 params
- LSTM cell: coupled gates (i=1-f), LayerNorm, H=64 = 18,624 params
- Output: 64 × 256 + 256 = 16,640 params + 384 LN params
- Total: 43,840 params
- Training: SGD per byte, grad clip = 5.0

Integration: expert prediction → byte_probs_to_bit_preds → external input
to hierarchical mixer (creates Group 4: expert).

CLI: `--expert [--expert-lr F]`, default lr=0.01.

## Results

| Config | 10KB BPB | 100KB BPB | Speed |
|---|---|---|---|
| Baseline (no expert) | 1.1680 | 1.1852 | 148 B/s |
| + Expert lr=0.01 | 1.1919 (+0.0239) | 1.1979 (+0.0127) | 108 B/s |
| + Expert lr=0.001 | 1.1894 (+0.0214) | (skipped) | 86 B/s |

Regression decreasing with more data but still positive at 100KB.

## Analysis

### Root cause: hierarchical mixer group overhead

The regression is NOT from the expert itself — it's from the hierarchical
mixer creating a new group for each external input. With the expert:
- Groups: 0 (orders 0-2), 1 (orders 3-8+), 2 (RWKV), 3 (match), **4 (expert)**
- Top LSTM: 5 inputs instead of 4
- Extra parameters in top LSTM that don't converge at 100KB

This is the SAME mechanism behind:
- A2 (match multi-input): +0.013 with multiple match externals
- S4 in separate group: +0.013 when word models had own group
- B1 (2-layer LSTM): neutral at 100KB despite more capacity

### Why the expert signal is weak

1. **RWKV dominance**: RWKV-7 0.1B with 100M pretrained params already
   provides excellent byte predictions. A 44K param LSTM trained online
   from scratch cannot compete on text data.

2. **Online learning paradox**: At 100KB, the expert has seen 100K bytes.
   With 44K params, the ratio is only 2.3:1 (bytes:params). Insufficient
   for convergence. cmix's LSTM expert works because it processes 100MB
   (2300:1 ratio).

3. **Speed cost**: Expert adds ~30% overhead (148 → 108 B/s) from the
   256-way softmax computation per byte.

### Fundamental insight

**Adding any new external to the hierarchical mixer hurts at 100KB.**
The top LSTM has too few inputs (3-5 groups) and too little data to
learn optimal weighting for additional sources. This applies to ALL
potential new predictors, not just the LSTM expert.

The only ways to benefit from new predictors at this scale are:
1. Replace an existing predictor (not add a new one)
2. Merge predictions before the mixer (pre-blend into existing group)
3. Scale to larger datasets (>1MB) where the mixer can learn

## Verdict

**KILLED.** Expert code and `--expert` flag retained for future use
on larger datasets. The LSTM expert is architecturally sound but the
hierarchical mixer group overhead dominates at 100KB scale.

## References

- A2 (R40): same regression pattern from adding externals
- R34: cmix uses 2×200 LSTM at 100MB scale (2300:1 ratio)
- Heritage: "more models of same type: diminishing returns"
