# R12: Q4 Group Quantization Experiment

- **Date**: 2026-10-05
- **Status**: In Progress
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

## Results

(To be filled after experiment)
