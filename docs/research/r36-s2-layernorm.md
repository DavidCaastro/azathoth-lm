# R36: S2 — Per-Gate LayerNorm

**Date**: 2026-10-06
**Status**: Complete — CONFIRMED (-0.0057 BPB 100KB, strong early learning boost)
**Purpose**: Add per-gate LayerNorm to LSTM mixer as prerequisite for BPTT>1.

## Hypothesis

Per-gate LayerNorm will:
1. Stabilize pre-activation distributions across time
2. Improve early learning (faster convergence)
3. Maintain or slightly improve BPB on longer sequences
4. Be prerequisite for BPTT>1 where gradient stability is critical

**Prediction**: -0.005 to -0.015 BPB.

**Kill criteria**: Regression > 0.02 BPB on 100KB enwik8, or speed < 100 B/s.

## Background

- cmix applies per-gate LN with learnable gamma/beta
- No temporal dependency → streaming-compatible
- Standard in modern LSTMs for stable training
- R34: identified as S2 priority, prerequisite for S3 (BPTT>1)

## Implementation

### LayerNorm per gate

For each of the 3 gates (forget, candidate, output):
1. Compute raw pre-activation: `val = bias + w_ih*x + w_hh*h_prev`
2. LayerNorm across hidden dim: `x_hat = (val - mean) / sqrt(var + eps)`
3. Scale and shift: `ln_out = gamma * x_hat + beta`
4. Apply nonlinearity: sigmoid (f, o) or tanh (g)

New parameters: `ln_gamma` (3×H, init 1.0) and `ln_beta` (3×H, init 0.0).
Total: 2 × 3 × 128 = 768 new params (+1.5%).

### Forward restructuring

Changed from element-wise gate computation to gate-wise:
- Old: `for gh in 0..(3*H)` — one element at a time
- New: compute all pre-acts → for each gate: normalize → nonlinearity

This is necessary because LN needs the full hidden vector for mean/variance.

### Backward restructuring

LN backward requires per-gate processing (mean/var are gate-level statistics):
1. Compute d_ln_out for all gates (gradient through nonlinearities)
2. For each gate: recompute mean/var from cached_pre_act
3. Standard LN backward: d_x = inv_std * (d_x_hat - mean(d_x_hat) - x_hat * mean(d_x_hat * x_hat))
4. Update gamma, beta, then w_ih, w_hh, bias using d_pre_act

## Results

| Config | BPB (10KB) | BPB (100KB) | Params | Speed |
|---|---|---|---|---|
| S1 (coupled, no LN) | 1.2196 | 1.1955 | 50,562 | 158 B/s |
| **S2 (coupled + LN)** | **1.1686** | **1.1898** | **51,330** | **154 B/s** |
| Delta | **-0.0510** | **-0.0057** | +768 (+1.5%) | -2.5% |

### Progressive BPB with LN

| Progress | Without LN | With LN | Delta |
|---|---|---|---|
| 25% (25KB) | 1.2635 | 1.2456 | -0.0179 |
| 50% (50KB) | 1.2537 | 1.2455 | -0.0082 |
| 75% (75KB) | 1.2178 | 1.2114 | -0.0064 |
| 100% (100KB) | 1.1955 | 1.1898 | -0.0057 |

The delta narrows as more data is seen. LN's main benefit is faster
initial convergence — the LSTM reaches good mixing weights sooner.

### 10KB vs 100KB discrepancy

The 10KB improvement (-0.0510) is much larger than 100KB (-0.0057).
This is consistent with LN's known effect: it normalizes the gradient
landscape, allowing faster learning. With more data, the un-normalized
LSTM eventually catches up, but the early advantage persists.

For BPTT>1 (S3), this gradient normalization is critical — without it,
gradients through multiple timesteps explode or vanish.

## Verdict

**CONFIRMED.** LayerNorm delivers:
- **-0.0057 BPB on 100KB** (modest but real improvement)
- **-0.0510 BPB on 10KB** (strong early learning boost)
- **Only +768 params** (+1.5%), negligible speed impact (-2.5%)
- **Gradient stability** prerequisite for BPTT>1

The 10KB result is particularly important: azathoth-lm now at **1.1686 BPB**
on 10KB, which is a new best by a wide margin.

## Next: S3 (BPTT=8)

With coupled gates (S1) stabilizing cell state and LayerNorm (S2)
normalizing gradients, the prerequisites for BPTT>1 are met.
S3 will implement BPTT=8 (one full byte of temporal context) with
Adam optimizer and gradient clipping. Estimated -0.02 to -0.05 BPB.

## References

- Ba et al., "Layer Normalization" (2016)
- cmix v21: per-gate LN with learnable gamma/beta
- R34: Roadmap audit, S2 priority
- R35: S1 coupled gates (prerequisite confirmed)
