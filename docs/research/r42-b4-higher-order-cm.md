# R42: B4 — Higher-Order CM (Orders 12, 16)

**Date**: 2026-10-06
**Status**: Complete — KILLED (neutral, +0.0004 on 100KB)
**Purpose**: Add longer context order models to CM for extended pattern capture.

## Hypothesis

Higher-order context models (12, 16 bytes) will capture patterns
beyond order 8, providing new information to the mixer.

**Prediction**: -0.005 to -0.01 BPB.

**Kill criteria**: Neutral or regression on 100KB.

## Implementation

Added two OrderModel instances:
- Order 12, table_bits=16 (1.5 MB)
- Order 16, table_bits=15 (0.75 MB)

Total memory increase: +2.2 MB (97.6 → 99.8 MB).
Both placed in Group 1 (long context) of hierarchical mixer.

## Results

| Config | 10KB BPB | 100KB BPB | Speed | Memory |
|---|---|---|---|---|
| Baseline (orders 0-8) | 1.1680 | 1.1852 | 148 B/s | 97.6 MB |
| + Orders 12, 16 | 1.1689 | 1.1856 | 146 B/s | 99.8 MB |
| **Delta** | **+0.0009** | **+0.0004** | **-1%** | **+2.2 MB** |

## Analysis

### Why higher-order CM doesn't help with RWKV

1. **Redundant with RWKV**: RWKV-7 0.1B has 12 layers processing
   entire token sequences. Its attention-like mechanism (WKV) already
   captures 12-16+ byte contexts much better than hash-table models.

2. **Hash collision saturation**: At order 12-16, the context hash
   becomes essentially unique per occurrence. With 64K-32K table
   entries and 4-way associativity, most contexts will only be seen
   once — the model predicts 0.5 (no information).

3. **Consistent with R33**: Previous CM scaling experiment (9→14 models)
   also showed +0.0027 (neutral). The pattern is clear: adding more CM
   models of higher order does not improve compression when RWKV is present.

4. **Different without RWKV**: In CM-only compressors (PAQ8px, cmix),
   higher orders help because there's no neural model capturing long
   context. In our hybrid, RWKV already fills this role.

## Verdict

**KILLED.** Orders 12, 16 add +2.2 MB and -1% speed for +0.0004 BPB
(noise). Reverted. This confirms R33's finding that CM model count
is not the bottleneck in our hybrid architecture.

## References

- R33: CM scaling A1 Phase 1, +0.0027 (neutral on text)
- R34: "Model count not bottleneck"
- Gleipnir: uses up to order 16 but is CM-only (no neural)
