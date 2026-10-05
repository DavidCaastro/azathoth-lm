# R18: P1.4 Confidence Skip

**Date**: 2026-10-05
**Status**: Complete — KILLED
**Purpose**: Skip byte-level CM+bridge computation for high-confidence RWKV
predictions to improve throughput with minimal BPB loss.

## Hypothesis

When RWKV top-1 probability exceeds a threshold, the byte-level CM corrections
are negligible. Skipping CM+bridge processing for those tokens should give
2-5x throughput with ~0 BPB loss.

**Prediction**: At threshold=0.95, skip ~5-10% of tokens with < 0.005 BPB loss.
**Kill criteria**: BPB increase > 0.005 at any threshold, or throughput gain < 10%.

## Implementation

Added `--skip THRESHOLD` to `hybrid-eval`:
1. After RWKV softmax, check top-1 probability
2. If top-1 >= threshold: use RWKV token-level cross-entropy (skip bridge+CM prediction)
3. Call `cm.observe_byte()` for skipped bytes to maintain CM context (history + hash tables)
4. Track skip rate and throughput

Key design: `observe_byte()` updates CM hash tables without measuring prediction cost,
so CM doesn't lose context for future non-skipped bytes.

## Results (100KB enwik8)

| Threshold | BPB | Delta vs no-skip | Tokens skipped | Bytes skipped | Speed |
|---|---|---|---|---|---|
| 0.00 (no skip) | 1.2924 | — | 0% | 0 | 169 B/s |
| 0.95 | 1.2950 | **+0.0026** | 5.6% | 3,509 | 169 B/s |
| 0.80 | 1.3059 | **+0.0135** | 14.5% | 10,107 | 170 B/s |
| 0.50 | 1.3343 | **+0.0419** | 32.4% | 26,600 | 174 B/s |

### BPB degradation at all thresholds

Even the most conservative threshold (0.95) loses +0.0026 BPB. More aggressive
thresholds degrade rapidly: 0.50 loses +0.0419 BPB (3.2% relative degradation).

### Throughput gain is negligible

- No skip: 169 B/s
- Skip 32%: 174 B/s (+3%)
- Skip 6%: 169 B/s (+0%)

## Root Cause Analysis

### Why throughput doesn't improve

```
Time budget per token (~6ms total):
  RWKV forward pass:     ~5.8ms  (97%)  ← sequential, can't skip
  Softmax + bridge:      ~0.01ms (0.2%)
  CM process (per byte): ~0.02ms (0.3%)
  Other overhead:        ~0.17ms (2.5%)
```

RWKV forward pass is 97% of compute and must run for every token (sequential
state updates). CM+bridge together are ~0.5% of compute. Skipping them saves
essentially nothing.

The 2-5x throughput estimate in the roadmap assumed a more balanced architecture
where ensemble compute was a significant fraction. In our hybrid architecture,
the bottleneck is entirely RWKV.

### Why BPB degrades

When we skip CM+bridge for a token, we use RWKV token-level cross-entropy
distributed uniformly across bytes. This loses:
1. **CM's byte-level patterns**: CM captures exact local byte patterns that RWKV misses
2. **Bridge's conditional byte predictions**: Within a token, bridge provides
   P(byte_k | byte_0..k-1) which is more precise than uniform distribution
3. **Mixer calibration**: Non-skipped bytes benefit from mixer weights learned
   from ALL bytes. Skipping removes training signal from 5-32% of bytes.

## Kill Criteria Evaluation

| Criterion | Result | Pass? |
|---|---|---|
| BPB increase < 0.005 | 0.0026 (skip=0.95) | PASS at 0.95 only |
| Throughput gain > 10% | +3% at best | **FAIL** |

Throughput gain fails at all thresholds. **P1.4 is KILLED.**

## Conclusions

1. **Confidence skip is architecture-dependent.** In systems with expensive ensemble
   (cmix: 2077 models), skipping saves significant compute. In our hybrid where
   RWKV dominates, it saves ~0%.
2. **RWKV forward pass is the sole throughput bottleneck.** Any speedup must come
   from optimizing the forward pass itself (quantization, SIMD, caching), not from
   skipping downstream components.
3. **CM contributes even for "easy" bytes.** The +0.0026 BPB degradation at 95%
   threshold shows CM adds value beyond RWKV for 5.6% of tokens. This validates
   the hybrid architecture.
4. **Concept may revive if CM model count grows.** With 50+ CM models (heritage
   target) or LSTM mixer, ensemble compute fraction increases. Confidence skip
   could become relevant at that stage.

## Key Files

- `src/main.rs` — hybrid-eval with `--skip THRESHOLD`
- `src/domain/cm.rs` — `observe_byte()` method for context-only updates
