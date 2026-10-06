# R21: P3.1 Longest-Match Predictor (Simplified SA-PPM)

**Date**: 2026-10-06
**Status**: Complete — VALIDATED
**Purpose**: Add a longest-match predictor that finds long-range byte context
matches beyond CM's fixed order 0-8, providing variable-length context matching
similar to SA-PPM but with simpler hash-based implementation.

## Hypothesis

A hash-based longest-match model that finds previous occurrences of the current
context (up to 128 bytes) and predicts the next byte based on what followed
those matches will add information that CM orders 0-8 cannot capture.

**Prediction**: -0.10 to -0.30 BPB (heritage SA-PPM estimate).
**Kill criteria**: If match model adds < 0.005 BPB improvement on 100KB, stop.

## Implementation

### Architecture

```
Input byte stream
    |
    +---> MatchModel (context lengths 4, 8, 16, 32, 64, 128)
    |     - Rolling FNV-1a hash per context length
    |     - 4-way associative hash tables (most recent 4 positions)
    |     - Verify match, predict from longest match
    |     - Convert byte probs → 8 bit predictions (MSB first)
    |     → Extra input to mixer alongside CM + RWKV bridge
```

### Hash Table Structure

| Context len | Table bits | Entries | Memory |
|---|---|---|---|
| 4 bytes | 20 | 1M | 16 MB |
| 8 bytes | 19 | 512K | 8 MB |
| 16 bytes | 18 | 256K | 4 MB |
| 32 bytes | 17 | 128K | 2 MB |
| 64 bytes | 16 | 64K | 1 MB |
| 128 bytes | 15 | 32K | 0.5 MB |
| **Total** | | | **~32 MB** |

### Prediction Flow

1. For each context length (longest first):
   - Hash current context → look up 4 candidate positions
   - Verify each candidate against actual data
   - Track longest verified match
2. Build byte distribution from what follows best matches
3. Weight by match length (longer = more confident)
4. Laplace smoothing (α=0.1) for unseen bytes
5. Convert byte probs to bit predictions via `byte_probs_to_bit_preds()`
6. Feed as additional input to hierarchical mixer

### Integration

- `MatchModel` in `src/domain/match_model.rs`
- `--match` flag in hybrid-eval command
- `process_byte_with_externals()` added to ContextMixer for multiple external inputs
- Match model updates after each byte (online)

## Results

### 10KB enwik8

| Config | BPB | Delta |
|---|---|---|
| Hierarchical only | 1.2502 | — |
| **Hierarchical + Match** | **1.2408** | **-0.0094** |

### 100KB enwik8

| Config | BPB | Speed | Delta |
|---|---|---|---|
| Hierarchical only | 1.2272 | 138 B/s | — |
| **Hierarchical + Match** | **1.2177** | **138 B/s** | **-0.0095** |

**Delta: -0.0095 BPB** with zero throughput loss.

### Full comparison table

| Config | BPB 100KB | vs baseline |
|---|---|---|
| RWKV ensemble | 1.2984 | — |
| Logistic hybrid | 1.2924 | -0.0060 |
| LSTM hybrid | 1.2549 | -0.0435 |
| Hierarchical | 1.2272 | -0.0712 |
| **Hierarchical + Match** | **1.2177** | **-0.0807** |

## Analysis

### Why the gain is smaller than heritage predicted

Heritage estimated -0.10 to -0.30 BPB for SA-PPM. We observe -0.0095. Reasons:

1. **Overlap with CM**: CM orders 1-8 already capture exact matches up to 8 bytes.
   The match model's unique contribution is matches >8 bytes, which are rarer in
   the first 100KB of enwik8 (mostly short Wikipedia articles and markup).

2. **Not full SA-PPM**: Our implementation uses hash tables with 4-way associative
   lookup, not a true suffix array. This means we may miss some valid matches due
   to hash collisions, especially for shorter contexts where the 4 slots fill fast.

3. **No PPM exclusion**: True PPM uses exclusion (if order-n predicts, exclude
   symbols from order-(n-1)). We simply weight by match length. PPM exclusion
   could improve precision.

4. **100KB is small**: Long-range matches become more valuable on larger data where
   repeated patterns (templates, boilerplate) recur at longer distances. At 100KB,
   the match model's history is limited.

### Expected gains on full enwik8

The match model should improve significantly on full enwik8 (100MB):
- Wikipedia articles reuse templates, infoboxes, categories
- Repeated markup patterns (`[[`, `{{`, `<ref>`) at distances >>128 bytes
- Longer history = more match candidates = better predictions

Conservative estimate: -0.03 to -0.08 BPB on full enwik8.

### Speed analysis

Zero throughput loss (138 B/s both with and without match model) because:
- Match model computation is O(1) per byte (hash lookups + verification)
- RWKV forward pass dominates (~97% of compute)
- 32 MB additional memory is well within budget

## Key Files

- `src/domain/match_model.rs` — MatchModel (hash tables, predict, observe)
- `src/domain/cm.rs` — process_byte_with_externals() for multiple external inputs
- `src/main.rs` — --match flag in hybrid-eval

## Conclusions

1. **Match model VALIDATED.** -0.0095 BPB on 100KB with zero speed loss.
2. **New best BPB: 1.2177** (100KB enwik8).
3. **Conservative implementation**: Hash-based, not full SA-PPM. Room for improvement
   with proper suffix array, PPM exclusion, or more hash table entries.
4. **Expected to scale**: Gains should increase on larger data (more match history).
5. **Low risk, low cost**: 32 MB memory, no throughput impact. Worth keeping enabled.

## Next Steps

- Full enwik8 evaluation to measure asymptotic match model contribution
- PPM exclusion mechanism (heritage Tier 1)
- Suffix array construction for optimal matching (original P3.1 scope)
- Combine with P2.3 multi-corpus validation
