# R52: Adaptive Preprocessing — Phase 0 Implementation

**Date**: 2026-10-08
**Status**: In Progress (code complete, pending full eval)
**Purpose**: Implement R51 Phase 0 — adaptive preprocessing with delta coding, byte-plane splitting, and auto-detection.

## Hypothesis

Pre-transforming binary/numerical data before models see it will reduce BPB on Cluster C/D files (high BPB, low bytes/token) without affecting text performance.

**Prediction**: -0.3 to -1.5 BPB on binary files (ait-E, ait-F, ait-G, sao, x-ray), neutral on text.

**Kill criteria**: If any text file BPB rises >0.01, revert.

## Implementation

### Transforms

1. **Delta coding** (`delta:N`): `out[i] = data[i] - data[i-stride]` (wrapping u8). Strides: 1, 2, 4, 8.
   - Inverse: `data[i] = delta[i] + decoded[i-stride]`
   - Best for: correlated consecutive values (audio, images, sensors)

2. **Byte-plane split** (`byteplane:N`): Rearrange so plane k = {data[k], data[k+stride], ...}. Strides: 2, 4, 8.
   - Inverse: reverse interleave
   - Best for: multi-byte aligned records (float32, structs, executables)

3. **Identity**: No transform (text, code, JSON).

### Auto-detection

Analyzes first 8192 bytes:
- Compute raw byte entropy H0
- For each delta stride: compute entropy of delta-encoded sample
- For each byteplane stride: compute weighted mean per-plane entropy (min 64 samples/plane)
- Select transform with lowest entropy if it beats H0 by >0.15 bits/byte
- Fallback: identity

### Integration

New CLI flag: `--preprocess auto|identity|delta:N|byteplane:N`

Applied AFTER E8/E9 (if enabled), BEFORE tokenization and evaluation.
All models (CM, RWKV, match) see the transformed stream.

## Auto-detection Results (8KB sample)

### Silesia corpus

| File | Type | Transform | H raw | H best | Delta |
|---|---|---|---|---|---|
| dickens | text | identity | 4.99 | 4.99 | 0 |
| mozilla | x86 binary | byteplane(4) | 3.98 | 3.80 | -0.18 |
| nci | chemical | identity | 2.31 | 2.31 | 0 |
| ooffice | MS Office | byteplane(8) | 4.17 | 3.98 | -0.19 |
| osdb | database | byteplane(4) | 6.57 | 6.41 | -0.16 |
| samba | x86 code | identity | 4.55 | 4.55 | 0 |
| sao | astronomy | byteplane(4) | 7.33 | 6.69 | -0.64 |
| webster | text | identity | 4.99 | 4.99 | 0 |
| x-ray | medical imaging | delta(2) | 6.48 | 4.75 | -1.73 |
| xml | text | identity | 5.25 | 5.25 | 0 |

### T3 modern data

| File | Type | Transform | H raw | H best | Delta |
|---|---|---|---|---|---|
| ait-A | unknown | identity | 4.19 | 4.19 | 0 |
| ait-B | unknown | identity | 5.44 | 5.44 | 0 |
| ait-C | unknown | identity | 5.23 | 5.23 | 0 |
| ait-D | binary | byteplane(8) | 7.98 | 7.81 | -0.17 |
| ait-E | float data | byteplane(4) | 7.03 | 6.17 | -0.86 |
| ait-F | binary | byteplane(2) | 7.01 | 6.31 | -0.70 |
| ait-G | correlated | delta(2) | 4.63 | 3.12 | -1.51 |
| ait-H | structured | byteplane(8) | 3.27 | 2.84 | -0.43 |
| ml-weights | safetensors | identity | 4.99 | 4.99 | 0 |
| modern-pe | x64 PE | byteplane(8) | 5.74 | 5.57 | -0.17 |
| jsonl | text | identity | 4.75 | 4.75 | 0 |

### Other

| File | Type | Transform | H raw | H best | Delta |
|---|---|---|---|---|---|
| enwik8 | text (wiki) | identity | 5.11 | 5.11 | 0 |
| OEIS | text (numbers) | identity | 3.42 | 3.42 | 0 |

### Summary

- **Text files**: all correctly identified as identity (zero risk)
- **Binary structured**: byteplane detected at correct stride
- **Correlated binary**: delta detected (x-ray stride 2, ait-G stride 2)
- **Largest entropy reduction**: x-ray (-1.73), ait-G (-1.51), ait-E (-0.86)
- **False positives**: none observed

## Tests

22 unit tests covering:
- Roundtrip correctness for all transforms (all strides)
- Edge cases: empty data, unaligned lengths, wrapping arithmetic
- Auto-detection: text→identity, ramp→delta, float-like→byteplane
- Argument parsing

## Next Steps

1. Full eval: run T1b/T2b/T3 with `--preprocess auto` to measure actual BPB impact
2. Compare baseline (no preprocess) vs preprocessed on all 25 files
3. If confirmed, make `--preprocess auto` the default

## Files Changed

- `src/domain/preprocess.rs` — expanded with delta, byteplane, auto-detect
- `src/main.rs` — added `--preprocess` flag to hybrid-eval
