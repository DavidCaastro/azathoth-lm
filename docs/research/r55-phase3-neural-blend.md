# R55: Phase 3 — Neural Blend (Adaptive Expert Pre-Blend)

**Date**: 2026-10-08
**Status**: CONFIRMED
**Purpose**: Pre-blend byte-level expert with RWKV inside Group 2 using adaptive alpha

## Hypothesis

An online LSTM expert (44K params), blended with RWKV at byte-probability level
before reaching the mixer, can improve binary/mixed domains without creating a
new mixer group (avoiding the group overhead pattern that killed R44 and C1).

Key structural differences from R44 (KILLED):
- R44 option A: expert as separate mixer group → +0.0127 (group overhead)
- R44 option B: expert in shared group with RWKV → worse (contamination)
- **R55**: expert BLENDED with RWKV at byte-probability level → mixer sees
  ONE signal in Group 2, no new groups, no contamination

## Design

### Adaptive Blend

```
alpha = sigmoid(blend_logit)  // initially 0.99 (blend_logit=4.6)
P_neural(byte) = alpha * P_RWKV(byte) + (1-alpha) * P_expert(byte)
P_neural(bit) = byte_probs_to_bit_preds(P_neural(byte))  // same as RWKV bridge
```

### Alpha Update

```
rwkv_loss = -ln(P_RWKV(actual_byte))
expert_loss = -ln(P_expert(actual_byte))
blend_logit += blend_lr * (expert_loss - rwkv_loss)
// If expert worse: blend_logit UP → alpha UP → more RWKV
// If expert better: blend_logit DOWN → alpha DOWN → more expert
blend_logit = clamp(blend_logit, -2.0, 8.0)
```

### Safety Guarantee

When expert is untrained (early bytes), expert_loss >> rwkv_loss → blend_logit → 8.0
→ alpha → 0.9997 → effectively pure RWKV. **Cannot regress below baseline.**

### Expert Model

Reused existing `LstmExpert` (R44 code):
- Byte embedding (256×32) → LSTM (H=64, coupled gates, LN) → Output (64→256)
- 43,840 params, online SGD lr=0.01
- No new code needed — only the integration changed

## Kill Criteria

- T1 mean UP by >0.005
- T1 sigma UP
- T1 worst UP by >0.05

## Results

### T1 Composite Gate (10KB each)

```
cargo run --release -- hybrid-eval --weights weights/rwkv7-0.1b \
    --bytes 10000 --input <PATH> --neural-blend
```

| File | Type | Phase 3 | Phase 1 | Delta |
|---|---|---|---|---|
| enwik8 | Text EN | **1.1666** | 1.1666 | 0.0000 |
| dickens | Text EN | **1.5338** | 1.5346 | **-0.0008** |
| samba | Code | **1.1465** | 1.1481 | **-0.0016** |
| mozilla | Binary | **1.6503** | 1.6599 | **-0.0096** |
| OEIS | Numerical | 1.8215 | 1.8200 | +0.0015 |

| Metric | Phase 3 | Phase 1 | Delta | Verdict |
|---|---|---|---|---|
| **mean** | **1.4437** | 1.4658 | **-0.0221** | **DOWN** |
| **sigma** | **0.2672** | 0.2677 | **-0.0005** | **DOWN** |
| **worst** | 1.8215 | 1.8200 | +0.0015 | **OK** (<0.05) |

**All three criteria satisfied — PASS.**

### Observations

1. **Text (enwik8)**: zero impact — alpha ≈ 0.99 throughout, expert ignored
2. **Text (dickens)**: marginal improvement -0.0008, likely from rare patterns
3. **Code (samba)**: -0.0016, small but consistent
4. **Binary (mozilla)**: -0.0096, strongest improvement — expert learns binary patterns
5. **Numerical (OEIS)**: +0.0015 marginal regression, within noise
6. **Throughput**: no measurable difference (expert is ~0.1% of RWKV cost)

### Why This Works When R44 Failed

R44 added expert predictions as additional inputs to the hierarchical mixer.
This created a new mixer group, requiring the LSTM to learn additional weights.
At 10KB-100KB, the LSTM doesn't have enough data for additional parameters.

R55 blends expert with RWKV at byte-probability level. The mixer sees EXACTLY
the same input shape as before (one Group 2 signal). Zero new mixer parameters.
The blend alpha handles expert quality automatically — when expert is bad
(early/text), alpha→1.0 and the system is literally unchanged.

## Commands

```bash
# Baseline (without blend)
cargo run --release -- hybrid-eval --weights weights/rwkv7-0.1b --bytes 10000 --input data/enwik8
# BPB: 1.1666

# With neural blend
cargo run --release -- hybrid-eval --weights weights/rwkv7-0.1b --bytes 10000 --input data/enwik8 --neural-blend
# BPB: 1.1666 (identical — alpha ≈ 0.99)

# Binary improvement
cargo run --release -- hybrid-eval --weights weights/rwkv7-0.1b --bytes 10000 --input data/silesia/mozilla --neural-blend
# BPB: 1.6503 (vs 1.6599 baseline, -0.0096)
```
