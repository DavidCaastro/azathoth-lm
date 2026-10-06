# R39: A1 — APM/SSE Post-LSTM Correction

**Date**: 2026-10-06
**Status**: Complete — KILLED (+0.10 to +0.19 BPB regression on 10KB)
**Purpose**: Add post-mixer APM stages for probability recalibration.

## Hypothesis

Post-LSTM APM (Adaptive Probability Map) stages will:
1. Correct systematic biases in LSTM output per bit position
2. Recalibrate predictions based on last byte context
3. Add -0.005 to -0.020 BPB

**Prediction**: -0.005 to -0.020 BPB.

**Kill criteria**: Regression > 0.02 BPB on 10KB.

## Background

- cmix uses SSE post-LSTM (validates the concept at scale)
- Gleipnir uses 11 APM stages with distinct contexts
- Heritage: "cascaded SSE: always overcorrects" — due to context reuse
- Heritage: "block-type detection + SSE 8192 ctx: +0.105 BPB (dilute data)"

## Implementation

### APM struct

Each APM stage is a table of stretched probabilities:
- Table: `[n_contexts][33 bins]` — 33 quantization bins for input probability
- Initialization: identity function (output = input)
- Prediction: quantize input P, interpolate between adjacent bins
- Update: SGD on adjacent bins proportional to interpolation weight

### Stages tested

1. **Stage 0**: bit position context (8 entries x 33 bins = 1.1 KB)
2. **Stage 1**: last byte context (256 entries x 33 bins = 33.8 KB)

### Integration point

Post-mixer, pre-bit-cost: mixer prediction → APM → refined prediction.
APM updates in reverse order. Mixer updates use pre-APM prediction.

## Results

| Config | BPB (10KB) | Delta vs S4 (1.1680) |
|---|---|---|
| S4 baseline (no APM) | 1.1680 | — |
| 2 stages, lr=0.008/0.004 | 1.3318 | **+0.1638** |
| 2 stages, lr=0.001/0.0005 | 1.3572 | **+0.1892** |
| 1 stage, lr=0.0003 | 1.2686 | **+0.1006** |

All configurations show massive regression. Kill criteria (+0.02) violated.

## Analysis

### Why APM fails here

1. **Data sparsity**: At 10KB (80K bits), each (bin, context) combination
   is visited too few times for the APM to learn meaningful corrections.
   With 33 bins x 8 contexts = 264 entries, each bin gets ~300 updates.
   With 33 bins x 256 contexts = 8448 entries, most get <10 updates.

2. **Well-calibrated LSTM**: Our LSTM mixer with BPTT=8 + LayerNorm
   already produces well-calibrated probabilities. The APM has nothing
   useful to correct and instead adds noise.

3. **Heritage confirmed**: "Cascaded SSE: always overcorrects" and
   "SSE 8192 ctx: +0.105 BPB" — both confirmed. The data volume
   required for APM to converge exceeds our 100KB eval window.

4. **cmix's context**: cmix processes full enwik8 (100MB = 800M bits).
   At that scale, even the 256-context APM gets ~3M updates per context.
   Our 100KB eval gives each context ~3K updates — 1000x less.

### When APM might work

APM/SSE becomes viable when:
- Processing full enwik8 (100MB+)
- After significant LSTM improvements (more miscalibration to correct)
- With very few contexts (e.g., 4-8) and many bins

## Verdict

**KILLED.** APM/SSE post-LSTM causes +0.10 to +0.19 BPB regression.
The LSTM is already well-calibrated at 100KB scale, and APM bins
are too sparse to learn useful corrections. Code retained (disabled)
for potential use on full enwik8 evaluation.

## References

- cmix v21: SSE post-LSTM (works at 100MB scale)
- Gleipnir: 11 APM stages (CM-only, no LSTM)
- Heritage: SSE overcorrection documented
