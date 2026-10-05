# R12: Q4 Group Quantization Experiment

- **Date**: 2026-10-05
- **Status**: Complete (KILLED)
- **Purpose**: Test Q4 weights (group_size=64) with Q8 activations for further memory/speed gains

---

## Hypothesis

Q4 weight quantization with group quantization (one scale per 64 elements
instead of one per row) will:
- Reduce weight memory by ~50% vs Q8 (~65 MB vs ~130 MB)
- Maintain BPB within ±0.02 of Q8 baseline (1.2984)
- Maintain or improve speed (less memory bandwidth)

## Design

### Q4 format: Q4g64
- Each weight stored as 4 bits (two weights per byte, packed)
- One f32 scale per group of 64 elements
- Scale overhead: 4 bytes per 64 weights = 6.25%
- Memory per weight: 0.5 bytes + 0.0625 bytes = 0.5625 bytes

### What gets Q4
- 6 large matrices per layer (key, value, receptance, output, FFN key, FFN value)
- Head (V×D) stays Q8 (already working well)
- Small matrices (LoRA, norms, mixing vectors) stay f32

### Activation quantization
- Input vector stays Q8 (i8) — activations are more sensitive to quantization
- Accumulation: i8 (activation) × i4→i8 (weight dequant) → i32

### Kill criteria
- BPB > 1.35 on 10KB → abort (>0.07 regression from Q8's 1.2797)
- Speed < 80 B/s → no benefit over Q8

## Results — 10KB smoke test

| Metric | Q8 int-accum | Q4g64 | Delta | Kill? |
|---|---|---|---|---|
| BPB | 1.2797 | **1.4234** | **+0.1437** | YES (> 1.35) |
| Speed | 91 B/s | **17 B/s** | **-81%** | YES (< 80) |
| RAM (weights) | ~130 MB | ~94 MB | -28% | — |

Both kill criteria triggered. Q4 reverted, Q8 restored.

## Analysis

### Why BPB regressed (+0.1437)

Q4 has only 15 quantization levels (-8 to 7) vs Q8's 255 levels.
For a 0.1B model with D=768, each weight carries significant information.
The quantization error accumulates across 12 layers, compounding at each
step. The group_size=64 helps with outliers but cannot compensate for the
fundamental loss of 4 bits of precision per weight.

For comparison: Q8 actually improved BPB (-0.0254) via regularization.
Q4 crossed the line from beneficial noise to destructive noise.

### Why speed regressed (-81%)

The Q4 kernel requires per-element nibble unpacking:
```
let byte = packed[idx / 2];
let nibble = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
let val = sign_extend(nibble);
```

This is branch-heavy and not SIMD-friendly. Unlike Q8 where each weight
is a contiguous i8 that maps directly to SIMD lanes, Q4 requires bit
extraction that serializes the inner loop. The overhead of unpacking
exceeds the bandwidth savings from smaller data.

Additionally, the group-level accumulation (one scale per 64 elements)
prevents the 4-way row ILP optimization that Q8 uses.

### When Q4 might work

- Larger models (1B+) where individual weights carry less information
- With calibration-based quantization (GPTQ, AWQ) instead of naive min/max
- With optimized SIMD kernels that handle nibble unpacking in bulk
- When combined with mixed precision (Q4 for FFN, Q8 for attention)

None of these apply to our current setup (0.1B model, naive quantization,
scalar nibble unpacking).

## Verdict

**KILLED.** Q4 is not viable for 0.1B RWKV-7 with our current kernel.
Q8 with integer accumulation remains the optimal quantization level.
The Q4Tensor implementation is kept in tensor.rs for potential future use
with larger models.

## Lessons

- Q8 is a sweet spot for 0.1B: enough precision to regularize, not enough
  to destroy signal
- Nibble-packed Q4 requires specialized SIMD kernels to be competitive
- Memory savings alone don't justify quantization — speed and quality must
  both hold
