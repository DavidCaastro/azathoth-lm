# R54: Phase 2 — Tweedie Post-Correction

**Date**: 2026-10-08
**Status**: KILLED
**Purpose**: Test Tweedie/calibration-based post-correction of mixer output

## Hypothesis

The LSTM mixer has systematic, context-dependent prediction biases that
can be detected and corrected using per-context statistics. Two approaches:

1. **Tweedie shrinkage** (James-Stein): shrink mixer logits toward
   context-specific mean. Math-guaranteed improvement for dim >= 3.
2. **Bias calibration**: track residual (prediction - outcome) per context
   and subtract accumulated bias.

Context buckets: bit_pos(0-7) x last_byte(0-255) = 2048 buckets, 16-24 KB.

## Prediction

- Estimated -0.01 to -0.03 BPB, concentrated on high-entropy domains (C/D)
- Should NOT hurt text domains where mixer is already well-calibrated

## Kill Criteria

- Any regression > +0.01 on T1 mean
- sigma increase

## Implementation

### Approach 1: Tweedie Logit Shrinkage

Tracked per-bucket: mean(logit), var(logit), count. Formula:
```
tau2_prior = 16.0
lambda = sigma^2 / (sigma^2 + tau2_prior)
correction = lambda * (mu_logit - prediction_logit)
logit_corrected = prediction_logit + clamp(correction, -0.3, 0.3)
```

### Approach 2: Bias Calibration

Tracked per-bucket: EMA(prediction - bit), count. Formula:
```
bias_ema = EMA(prediction - actual_bit)
corrected = prediction - clamp(bias_ema, -0.05, 0.05)
```

## Results

### Approach 1 — Tweedie Shrinkage (various MIN_COUNT)

All results: enwik8 10KB, `hybrid-eval --weights weights/rwkv7-0.1b --bytes 10000`

| MIN_COUNT | tau2_prior | enwik8 BPB | Delta vs Phase 1 (1.1666) |
|---|---|---|---|
| 8 | 4.0 | 1.1959 | **+0.0293** |
| 64 | 16.0 | 1.1767 | **+0.0101** |
| 200 | 16.0 | 1.1725 | **+0.0059** |
| 1000 | 16.0 | 1.1676 | **+0.0010** |

Monotonic improvement only by disabling the correction (higher threshold).

### Approach 2 — Bias Calibration (gradient-isolated)

Mixer update uses raw prediction (not corrected) to avoid gradient contamination.

| Config | enwik8 10KB | Delta |
|---|---|---|
| MIN_COUNT=32, MAX=0.08, EMA=0.01, no warmup | 1.2455 | **+0.0789** |
| MIN_COUNT=32, MAX=0.05, EMA=0.005, warmup=10KB | 1.1666 | 0 (no-op at 10KB) |
| Same config, enwik8 100KB | 1.2305 | **+0.0453** vs 1.1852 |

### Critical Bug Found

Initial implementation passed Tweedie-corrected prediction to mixer update,
contaminating LSTM gradients. After fixing (mixer learns from own raw output),
regression reduced from 1.8154 to 1.2455 at 10KB, but still harmful.

## Root Cause Analysis

### Why Tweedie Fails Here

1. **James-Stein assumptions violated**: Requires independent observations of
   FIXED parameters. Our mixer is online-adaptive — parameters change every step.
   Shrinking toward the mean of a non-stationary process is meaningless.

2. **Bias calibration fights adaptation**: The mixer ALREADY corrects its own
   biases via online gradient descent. Tracking past biases and correcting for
   them is circular — the mixer has already moved past those biases.

3. **Same root cause as SSE/APM**: Adding a second adaptation loop on top of
   the first creates interference. Heritage: "Cascaded SSE: siempre overcorrects."
   Single-stage SSE is gentler but still harmful because it assumes the mixer
   has stable biases, which it doesn't.

4. **Bucket granularity trap**: 2048 buckets x 8 bits = data-hungry. At 10KB
   (80K bits), ~39 obs/bucket average but very skewed. Common bytes (space, 'e')
   get 1000+ while rare bytes get <10. No MIN_COUNT solves both problems.

### When Post-Correction COULD Work

- With a FIXED (non-adaptive) mixer (not our architecture)
- At much larger scales (>10MB) where bias patterns stabilize
- As a LOOKUP table trained offline on a separate calibration set
- In a two-pass architecture: first pass learns, second pass corrects

## Verdict

**KILLED**. Post-correction of an online-adaptive mixer is fundamentally
counterproductive. The mixer's own gradient descent is a superior calibration
mechanism. Adding any form of post-hoc correction creates interference.

## Heritage Entry

Added to `.claude/rules/heritage.md`:
- Tweedie/bias post-correction on adaptive mixer: KILLED (R54)
- Root cause: second adaptation loop interferes with mixer's online learning

## Commands Used

```bash
# Phase 1 baseline (no Tweedie)
cargo run --release -- hybrid-eval --weights weights/rwkv7-0.1b --bytes 10000 --input data/enwik8
# BPB: 1.1666

# Tweedie shrinkage (MIN_COUNT=8, tau2=4.0)
# BPB: 1.1959 (+0.0293)

# Bias calibration (MIN_COUNT=32, MAX=0.08)
# BPB: 1.2455 (+0.0789)

# With gradient isolation + 10KB warmup, 100KB eval
cargo run --release -- hybrid-eval --weights weights/rwkv7-0.1b --bytes 100000 --input data/enwik8
# BPB: 1.2305 (+0.0453 vs 1.1852 baseline)
```
