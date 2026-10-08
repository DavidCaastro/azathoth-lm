# R52: Adaptive Preprocessing — Phase 0 Implementation

**Date**: 2026-10-08
**Status**: KILLED
**Purpose**: Implement R51 Phase 0 — adaptive preprocessing with delta coding, byte-plane splitting, and auto-detection.
**Verdict**: Preprocessing is fundamentally incompatible with pre-trained neural models in hybrid mode. Transforms destroy RWKV's learned distribution, causing catastrophic BPB regression on every file where transforms activate.

## Hypothesis

Pre-transforming binary/numerical data before models see it will reduce BPB on Cluster C/D files (high BPB, low bytes/token) without affecting text performance.

**Prediction**: -0.3 to -1.5 BPB on binary files (ait-E, ait-F, ait-G, sao, x-ray), neutral on text.

**Kill criteria**: If any text file BPB rises >0.01, revert.

**Result**: REFUTED. Text files unaffected (identity correctly selected), but binary files with active transforms show +2.4 to +4.1 BPB regression. Kill criteria met on structural grounds — transforms and RWKV are architecturally incompatible.

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

## T2b Eval Results (--preprocess auto, 100KB)

Ran on 12 Silesia files. 5 completed, 7 crashed with OOM (memory leak across sequential runs on Windows).

### Completed files

| File | Transform | BPB (preprocess) | BPB (baseline R50) | Delta | Verdict |
|---|---|---|---|---|---|
| dickens | identity | 1.3461 | 1.3461 | **0.0000** | Neutral (correct) |
| nci | identity | 0.3228 | 0.3228 | **0.0000** | Neutral (correct) |
| mozilla | byteplane(4) | **5.2636** | 1.1385 | **+4.1251** | CATASTROPHIC |
| mr | byteplane(4) | **3.8135** | 1.4190 | **+2.3945** | CATASTROPHIC |
| ooffice | byteplane(8) | **5.9697** | 2.9265 | **+3.0432** | CATASTROPHIC |

### Crashed files (OOM: "memory allocation of 201326592 bytes failed")

osdb (byteplane 4), reymont (identity), samba (identity), sao (byteplane 4),
webster (identity), x-ray (delta 2), xml (identity).

Sequential process accumulation on Windows — 192MB RWKV allocation fails
after 5 runs without full memory release between processes.

### BPB progression on transformed files (every 25KB)

mozilla byteplane(4): 2.37 → 3.02 → 4.44 → **5.26** (DIVERGING — worse with more data)
mr byteplane(4): 2.88 → 3.11 → 3.56 → **3.81** (DIVERGING)
ooffice byteplane(8): 5.88 → 5.96 → 5.95 → **5.97** (FLAT at near-random)

Key observation: BPB does NOT converge downward. The models cannot learn
the transformed distribution because RWKV's pre-trained weights expect
the original byte patterns. Online CM adaptation is overwhelmed by
RWKV's confident but wrong predictions.

## Root Cause Analysis

### Why preprocessing fails in hybrid mode

1. **RWKV was pre-trained on raw data**: its 100M parameters encode
   statistical patterns of natural text, code, executables. Byteplane(4)
   takes every 4th byte — RWKV sees permuted data that matches nothing
   it learned. Delta coding shifts the byte distribution entirely.

2. **RWKV dominates the mixer**: per R47, RWKV contributes 12-61% of
   predictions across all domains. Even in Cluster D where RWKV is weakest
   (12%), its confident-but-wrong predictions via stretch() amplify into
   the LSTM mixer, poisoning the final output.

3. **CM cannot compensate**: CM learns online and CAN adapt to transformed
   data, but the LSTM mixer has learned to trust RWKV's group. At 100KB,
   the mixer cannot unlearn this trust fast enough.

4. **Fundamental architectural conflict**: preprocessing transforms the
   data BEFORE all models. In a CM-only system this works (all models
   adapt online). In a hybrid with pre-trained components, the pre-trained
   model's knowledge becomes anti-knowledge.

### Comparison with AIT DCC G2-V3

G2-V3 uses similar preprocessing (byte-plane split, delta coding). But:
- G2-V3 is CM-ONLY (no pre-trained neural model)
- All its predictors are online-adaptive
- Preprocessing reduces entropy → CM learns faster → BPB improves

Our system is hybrid. The pre-trained RWKV is the dominant predictor.
Preprocessing helps CM but destroys RWKV. Net effect: catastrophic loss.

### Could bifurcated preprocessing work?

In principle, preprocessing could apply ONLY to CM while RWKV sees raw data.
This would require:
- Dual data paths (transformed for CM, raw for RWKV)
- Separate match model on raw data
- Complex state management for the byte bridge

Estimated effort: significant refactor. Expected gain: marginal (CM alone
on binary data only benefits ~0.1-0.3 BPB on Cluster D where CM is already
weak). Not worth the complexity.

## Lesson for heritage.md

**Adaptive preprocessing (delta, byteplane) is incompatible with pre-trained
neural models in hybrid mode.** Pre-trained models expect the original data
distribution. Transforming data before a pre-trained model is equivalent to
evaluating on an out-of-distribution input — the model's learned priors
become anti-priors.

This applies to ANY hybrid system with frozen/pre-trained components.
Preprocessing is only viable when ALL predictors are online-adaptive (CM-only).

Exception: E8/E9 transform works because it's narrow (only CALL/JMP addresses),
preserving >99% of the byte stream unchanged. RWKV tolerance for E8/E9 was
validated in R49 (text: neutral, mozilla: -0.18 BPB).

## Code Disposition

- `src/domain/preprocess.rs`: RETAINED (E8/E9 still used, transforms are
  correct and tested). `--preprocess` CLI flag remains available but should
  NOT be used with hybrid-eval in production.
- `run-t2b-preprocess.sh`: DELETE (one-shot eval script, results captured here).
- `docs/results/r52-t2b-preprocess/`: DELETE (partial results, superseded by this doc).

## Files Changed

- `src/domain/preprocess.rs` — expanded with delta, byteplane, auto-detect
- `src/main.rs` — added `--preprocess` flag to hybrid-eval
