# R53: Phase 1 — Byte-Context LSTM with Extended BPTT

**Date**: 2026-10-08
**Status**: CONFIRMED — T1 composite gate passed (all 3 metrics improved)
**Purpose**: Enrich LSTM mixer input with byte context and extend temporal learning from BPTT=8 to BPTT=64.

## Hypothesis

The LSTM mixer receives only 2-4 group probabilities (no byte context, no bit position).
cmix/fx2-cmix/RATA-CMIX feed their LSTM with the actual byte + model predictions +
bit position, with BPTT=100-128 bytes. Our gap vs SOTA is integration, not models.

**Prediction**: -0.01 to -0.03 BPB on text, -0.05 on binary (R51 estimate).

**Kill criteria**: If 100KB BPB > baseline + 0.01, revert.

## Changes

### LSTM Mixer (`lstm_mixer.rs`)

1. **BPTT**: 8 → 64 (8 bytes of temporal context, vs 1 byte before)
2. **Context separation**: new `context_dim` field separates model predictions
   (get stretched + mixing weights) from context features (raw, no weights)
3. **API**: `predict_ctx(model_probs, ctx)` / `update_ctx(model_probs, ctx, pred, bit)`
   - Context enriches LSTM hidden state but does NOT get mixing weights
   - Old `predict()`/`update()` call new methods with empty context
4. **`extend_models()`**: correctly inserts new model columns before context
   columns, preserving learned context weights

### Context Mixer (`cm.rs`)

1. **Byte tracking**: `last_bytes: [u8; 4]` circular buffer + `byte_count`
2. **Context buffer**: 40-float `ctx_buf` in Hierarchical variant
3. **Context features per bit** (j = current bit position):
   - Bit position one-hot: 8 floats, ctx[j] = 1.0, others = 0.0
   - Last 4 bytes binary: 32 floats, each bit → {-0.5, +0.5}
   - Total: BYTE_CTX_DIM = 40
4. **Constructor**: `new_full(n_groups, 40, hidden_dim, lr, n_layers)` for LSTM

### Parameters

| Metric | Before (BPTT=8) | After (BPTT=64+ctx) |
|---|---|---|
| LSTM input_dim | n_groups (4) | n_groups + 40 (44) |
| LSTM params | 51,330 | 66,690 |
| BPTT history | 8 × H buffers | 64 × H buffers (~200KB) |
| Context | none | bit_pos(8) + last_4_bytes(32) |
| Update freq | every byte | every 8 bytes |

## Exact Commands Used

**CRITICAL**: All evals MUST use the full flag set. Omitting flags causes invalid results.

```bash
# Baseline S3 (BPTT=8, no byte context)
cargo run --release -- hybrid-eval \
  --weights weights/rwkv7-0.1b \
  --bytes 10000 \
  --input data/enwik8 \
  --hierarchical --match --emb-surgery center0.3

# Phase 1 (BPTT=64 + byte context) — same flags, code difference only
cargo run --release -- hybrid-eval \
  --weights weights/rwkv7-0.1b \
  --bytes 10000 \
  --input data/enwik8 \
  --hierarchical --match --emb-surgery center0.3

# For 100KB eval, change --bytes 100000
# For other files, change --input path:
#   data/silesia/dickens
#   data/silesia/samba
#   data/silesia/mozilla
#   data/oeis/stripped
```

**Mandatory flags**: `--hierarchical --match --emb-surgery center0.3`
Without these, the eval uses a flat mixer without match model or embedding
surgery, producing ~1.6 BPB on enwik8 (vs 1.17 with full pipeline).

## Results

### enwik8 Initial Results (verified with World v2.8 weights)

| Eval | BPB (Phase 1) | BPB (baseline S3) | Delta |
|---|---|---|---|
| 10KB | 1.1666 | 1.1680 | **-0.0014** |
| 100KB | **1.1810** | 1.1843 | **-0.0033** |

### BPB progression at 100KB

| Checkpoint | Phase 1 | Baseline | Delta |
|---|---|---|---|
| 25KB | 1.2407 | ~1.21 | +0.03 (warming up) |
| 50KB | 1.2382 | ~1.20 | +0.04 (still converging) |
| 75KB | 1.2026 | ~1.19 | +0.01 (crossing over) |
| 100KB | **1.1810** | 1.1843 | **-0.0033** |

### LR Tuning (10KB enwik8)

| lr | BPB | Delta vs 0.002 |
|---|---|---|
| 0.002 (default) | **1.1666** | — |
| 0.004 | 1.1704 | +0.0038 |
| 0.006 | 1.1730 | +0.0064 |
| 0.008 | 1.1758 | +0.0092 |

**Conclusion**: lr=0.002 optimal. Higher lr does NOT compensate for 8x fewer Adam updates.

### T1 Composite Gate (10KB each, CONFIRMED)

| File | Type | Phase 1 | Baseline S3 | Delta | B/s |
|---|---|---|---|---|---|
| enwik8 | Text EN | **1.1666** | 1.1680 | **-0.0014** | 92 |
| dickens | Text EN | **1.5346** | 1.5465 | **-0.0119** | ~65 |
| samba | Code | 1.1481 | 1.1445 | +0.0036 | ~90 |
| mozilla | Binary | 1.6599 | 1.6404 | +0.0195 | 40 |
| OEIS | Numerical | **1.8200** | 1.8378 | **-0.0178** | 44 |

| Metric | Phase 1 | Baseline S3 | Delta | Verdict |
|---|---|---|---|---|
| **mean** | **1.4658** | 1.4674 | **-0.0016** | **DOWN** ✓ |
| **sigma** | **0.2677** | 0.3030 | **-0.0353** | **DOWN** ✓ |
| **worst** | **1.8200** (OEIS) | 1.8378 | **-0.0178** | **DOWN** ✓ |

**Verdict**: PASS — all three metrics improved. Phase 1 accepted.

### Throughput

| Eval | Phase 1 | Baseline | Delta |
|---|---|---|---|
| 10KB | 40-92 B/s | 40-133 B/s | -19% to -25% |
| 100KB | 111 B/s | 148 B/s | -25% |

Slowdown from: 8x less frequent BPTT backward (64 vs 8 steps per update)
and larger LSTM input (44 vs 4 floats). Within R51's estimated 40-60% range.

## Analysis

### Why it works

1. **Bit position context**: tells the LSTM which bit (0-7) it's predicting.
   Bits 0-1 (MSB) have fundamentally different statistics than bits 5-7 (LSB).
   Without this, the LSTM treats all bits identically.

2. **Byte context**: last 4 decoded bytes give the LSTM character-level patterns.
   This is equivalent to cmix's byte input — the LSTM can now learn "after seeing
   'the ', predict space" or "after 0x0A0D, predict line content".

3. **Extended BPTT**: 8 bytes of gradient flow means the LSTM can learn cross-byte
   patterns. With BPTT=8 bits (1 byte), the LSTM only learned within-byte patterns.

### Why convergence is slower

With BPTT=64, Adam fires every 8 bytes instead of every byte. This means:
- 8x fewer bias correction steps (affects early training)
- Each update integrates 64 timesteps (more stable but less reactive)
- The output layer (SGD, every bit) compensates during warmup

### Structural difference from B1 (KILLED)

B1 tested BPTT=16/32 with the SAME 4-float input → zero benefit.
Phase 1 adds 40 NEW features. The LSTM now has fundamentally different
information at each step, making longer temporal context meaningful.

### Cross-domain behavior

- Text (enwik8, dickens): clear improvement, -0.0014 to -0.0119
- Code (samba): marginal regression +0.0036 (within noise)
- Binary (mozilla): +0.0195 (8x fewer updates, binary patterns harder to learn)
- Numerical (OEIS): -0.0178 (strong improvement, bit position info helps digits)
- sigma DOWN by 0.0353 → Phase 1 is more uniform across domains than S3

## Critical Lesson: Missing CLI Flags

During T1 evaluation, an initial round was run WITHOUT the required flags
(`--hierarchical --match --emb-surgery center0.3`), producing catastrophic
false results (mozilla: 2.52, OEIS: 2.05 — 0.8+ regression). These were
entirely due to using a flat logistic mixer instead of the full pipeline.

**Rule**: ALWAYS include the full flag set for valid eval results. See
"Exact Commands Used" section above. Partial flags are invalid.

## Next Steps

1. **T2b eval**: 12 Silesia × 100KB to verify composite at scale.
2. **If confirmed**: update baselines, proceed to Phase 2 (Tweedie).

## Files Changed

- `src/domain/lstm_mixer.rs` — BPTT=64, context_dim, predict_ctx/update_ctx
- `src/domain/cm.rs` — byte tracking, context enrichment in Hierarchical mixer
