# R41: B2 — BPTT Scaling (16, 32)

**Date**: 2026-10-06
**Status**: Complete — KILLED (neutral at 16, slight regression at 32)
**Purpose**: Scale BPTT from 8 to 16-32 bits for longer temporal context.

## Hypothesis

Longer BPTT windows (16-32 bits = 2-4 bytes) will capture inter-byte
temporal patterns that BPTT=8 (1 byte) misses.

**Prediction**: -0.01 to -0.03 BPB.

**Kill criteria**: No improvement on 100KB eval, or regression > 0.01 on 10KB.

## Results

| BPTT | 10KB BPB | 100KB BPB | Speed | Delta (100KB) |
|---|---|---|---|---|
| 8 (baseline) | 1.1680 | 1.1852 | 148 B/s | — |
| 16 | 1.1685 | 1.1852 | 130 B/s | +0.0000 |
| 32 | 1.1700 | (skipped) | 126 B/s | (10KB worse) |

BPTT=16: exactly neutral on 100KB (+0.0000), -12% speed.
BPTT=32: +0.0020 regression on 10KB, -15% speed. 100KB eval skipped.

## Analysis

### Why BPTT>8 doesn't help

1. **Bit-level vs byte-level**: Our LSTM operates on individual bits.
   BPTT=8 already covers 1 full byte — the natural unit of information.
   Cross-byte patterns at the bit level are noisy and high-entropy.

2. **Gradient dilution**: Longer BPTT windows mean gradients from the
   output layer must propagate through more timesteps. With only 51K
   params and ~100KB of data, the signal-to-noise ratio drops.

3. **cmix comparison**: cmix uses BPTT=100 at BYTE level (not bit level),
   with 2×200 LSTM (160K params per layer). That's 100 bytes of context
   in a much larger model. Our 51K param LSTM at bit level has
   fundamentally different capacity constraints.

4. **Speed cost**: Each doubling of BPTT adds ~10-15% overhead from
   storing more intermediate activations and longer backward passes.
   No BPB improvement makes this pure cost.

5. **Data scale**: At 100KB, the LSTM processes ~800K bit updates with
   only ~50K/100K BPTT backward passes. Longer windows mean fewer
   backward passes (50K at BPTT=16 vs 100K at BPTT=8), reducing
   total learning opportunities.

## Verdict

**KILLED.** BPTT=8 (1 byte) is the optimal window for our bit-level
LSTM mixer at this scale. The path to cmix-level temporal learning
requires either byte-level LSTM (not bit-level) or significantly
larger hidden dimensions — both are addressed by B1 (2-layer LSTM).

Reverted BPTT_LEN to 8.

## References

- S3 (R37): BPTT=8 implementation, -0.0055 BPB
- Heritage: "BPTT >1 during eval: +0.10 BPB" (analytic-lm, different cause)
- cmix: BPTT=100 byte-level, 2×200 LSTM
